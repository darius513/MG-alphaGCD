#include <iostream>
#include <mpi.h>
#include <unistd.h>

#include "../include/common.h"
#include "../include/graph/host_graph.h"
#include "../include/graph/gpu_graph.cuh"
#include "../include/louvain/louvain_baseline.cuh"
#include "../include/louvain/louvain_gl.cuh"
#include "../include/louvain/louvain.cuh"
#include "../include/partition/edge_partition.cuh"



int main(int argc, char* argv[]){
    int rank = 0, size = 1;
    int local_rank = -1;
    int local_size = 1;
    int num_device = 0;
    const std::string path_of_graph_str = get_argval<std::string>(argv, argv + argc, "-path_of_graph", "");
    const double threshold = get_argval<double>(argv, argv + argc, "-threshold", 1.0E-06);
    const int max_iter = get_argval<int>(argv, argv + argc, "-max_iter", 500);
    const int max_phases = get_argval<int>(argv, argv + argc, "-max_phases", 1000);
    const int random_vertex_num = get_argval<int>(argv, argv + argc, "-random_vertex_num", RANDOM_VERTEX_NUM);
    const double sparsity = get_argval<double>(argv, argv + argc, "-sparsity", 0.1);
    const int version = get_argval<int>(argv, argv + argc, "-version", 2);

    //----------------------------------------------------------------/
    //------------------------- init nvshmem -------------------------/
    CUDA_RT_CALL(cudaGetDeviceCount(&num_device));

    MPI_CALL(MPI_Init(&argc, &argv));
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                     &local_comm));
        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
        MPI_CALL(MPI_Comm_size(local_comm, &local_size));
        MPI_CALL(MPI_Comm_free(&local_comm));
    }
    if (1 < num_device && num_device < local_size) {
        fprintf(stderr,
                "ERROR Number of visible devices (%d) is less than number of ranks on the "
                "node (%d)!\n",
                num_device, local_size);
        MPI_CALL(MPI_Finalize());
        return 1;
    }
    if (1 == num_device) {
        // Only 1 device visible, assuming GPU affinity is handled via CUDA_VISIBLE_DEVICES
        CUDA_RT_CALL(cudaSetDevice(0));
    } else {
        CUDA_RT_CALL(cudaSetDevice(local_rank));
        int device_id;
        CUDA_RT_CALL(cudaGetDevice(&device_id));
    }
    CUDA_RT_CALL(cudaFree(0));

    //-------------------- load graph data on CPU --------------------/
    char *path_of_graph = const_cast<char *>(path_of_graph_str.c_str());
    bool random_graph = path_of_graph_str.empty();
    HostGraph *hostGraph = nullptr;
    if(random_graph){
        hostGraph = new HostGraph(random_vertex_num, sparsity, local_rank);
    }else{
        if (local_rank == 0) {
            printf("loading graph...\n");
        }
        hostGraph = new HostGraph(path_of_graph, local_rank);
    }
    hostGraph->compute_total_edge_weight();

    MPI_Comm mpi_comm;
    nvshmemx_init_attr_t attr;

    mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;

    // Its default value in nvshmem is 1 GB which is not sufficient
    auto total_vertices = hostGraph->get_total_vertices_();
    auto total_edges = hostGraph->get_total_edge_();
    edge_t len_edge_array = total_edges / local_size + (total_edges % local_size != 0) + total_vertices;
    long long unsigned int required_symmetric_heap_size = sizeof(vertex_t) * (4 * total_vertices + len_edge_array + 1) + sizeof(weight_t) * (2 * total_vertices + len_edge_array + 3);
    char *value = getenv("NVSHMEM_SYMMETRIC_SIZE");
    if (value) { /* env variable is set */
        long long unsigned int size_env = parse_nvshmem_symmetric_size(value);
        if (size_env < required_symmetric_heap_size) {
            fprintf(stderr,
                    "ERROR: Minimum NVSHMEM_SYMMETRIC_SIZE = %lluB, Current "
                    "NVSHMEM_SYMMETRIC_SIZE=%s\n",
                    required_symmetric_heap_size, value);
            MPI_CALL(MPI_Finalize());
            return -1;
        }
    } else {
        char symmetric_heap_size_str[100];
        sprintf(symmetric_heap_size_str, "%llu", required_symmetric_heap_size);

         if (rank == 0) {
             printf("Setting environment variable NVSHMEM_SYMMETRIC_SIZE = %llu bytes = %.3f GB\n",
                    required_symmetric_heap_size, (double)(required_symmetric_heap_size / 1000000000.0));
         }

        setenv("NVSHMEM_SYMMETRIC_SIZE", symmetric_heap_size_str, 1);
    }

    NVSHMEM_CHECK(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr));

    //----------------------------------------------------------------/
    int n_pes = nvshmem_n_pes();
    int my_pe = nvshmem_my_pe();
    GpuGraph* gpuGraph = new GpuGraph(n_pes, hostGraph);
    edge_partition::partitioner(hostGraph, gpuGraph, n_pes, my_pe);
    nvshmem_barrier_all();

    switch (version) {
        case 0:
            louvain_baseline::run(hostGraph, gpuGraph, threshold, max_iter, max_phases);
            break;
        case 1:
            louvain_gl::run(hostGraph, gpuGraph, threshold, max_iter, max_phases);
            break;
        case 2:
            louvain::run(hostGraph, gpuGraph, threshold, max_iter, max_phases);
            break;
        default:
            printf("the version is unsupported\n");
            break;
    }

    nvshmem_finalize();
    MPI_CALL(MPI_Finalize());
}
