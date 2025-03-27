#include "../../include/partition/edge_partition.cuh"

__global__ void
split(vertex_t *device_offset, edge_t *device_part_edge_offset, vertex_t *device_part_vertex_offset, int n_pes,
      vertex_t total_vertices){
    int my_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    edge_t begin_edge;
    if(my_id >= total_vertices) return ;
    edge_t lb = device_offset[my_id];
    edge_t rb = device_offset[my_id + 1];
    for(int i = 0; i < n_pes - 1; i++){
        begin_edge = device_part_edge_offset[i + 1];
        if(begin_edge >= lb && begin_edge < rb){
            device_part_vertex_offset[i + 1] = my_id;
        }
    }
}

__global__ void
sub_first_ele(vertex_t *offset, vertex_t first_ele, vertex_t len){
    for (int id = (blockIdx.x * blockDim.x) + threadIdx.x; id < len; id += blockDim.x * gridDim.x) {
        offset[id] -= first_ele;
    }
}

void edge_partition::partitioner(HostGraph *hostGraph, GpuGraph* gpuGraph, int n_pes, int my_pe) {
    auto total_vertices = hostGraph->get_total_vertices_();
    auto total_edges = hostGraph->get_total_edge_();
    edge_t *part_edge_offset = new edge_t[n_pes + 1];
    auto *part_vertex_offset = gpuGraph->get_part_vertex_offset_();
    auto *device_part_vertex_offset = gpuGraph->get_private_device_part_vertex_offset_();
    edge_t* device_edge;
    weight_t* device_edge_weight;
    auto* device_offset = gpuGraph->get_private_device_offset_();
    vertex_t *device_offset_copy;
    CUDA_RT_CALL(cudaMalloc((void **) &device_offset_copy, sizeof(vertex_t) * (total_vertices + 1)));
    CUDA_RT_CALL(cudaMemcpy(device_offset_copy, hostGraph->get_host_offset_(), sizeof(vertex_t) * (total_vertices + 1),
                            cudaMemcpyHostToDevice));

    edge_t *device_part_edge_offset;
    CUDA_RT_CALL(cudaMalloc((void **) &device_part_edge_offset, sizeof(edge_t) * (n_pes + 1)));

    part_vertex_offset[0] = 0;
    part_vertex_offset[n_pes] = total_vertices;
    CUDA_RT_CALL(cudaMemcpy(device_part_vertex_offset, part_vertex_offset, sizeof(vertex_t) * (n_pes + 1),
                            cudaMemcpyHostToDevice));

    part_edge_offset[0] = 0;
    for (int i = 0; i < n_pes; i++) {
        part_edge_offset[i + 1] = (part_edge_offset[i] + (total_edges / n_pes + (((total_edges % n_pes) > i) ? 1 : 0)));
    }
    CUDA_RT_CALL(cudaMemcpy(device_part_edge_offset, part_edge_offset, sizeof(edge_t) * (n_pes + 1),
                            cudaMemcpyHostToDevice));

    split<<<iDivUp(total_vertices, 256), 256>>>(device_offset_copy, device_part_edge_offset, device_part_vertex_offset,
                                                    n_pes, total_vertices);

    CUDA_RT_CALL(cudaMemcpy(part_vertex_offset, device_part_vertex_offset, sizeof(vertex_t) * (n_pes + 1),
                            cudaMemcpyDeviceToHost));

    edge_t max_local_edges = 0;
    for (int i = 0; i < n_pes; i++) {
        edge_t local_edges_pei = hostGraph->get_host_offset_()[part_vertex_offset[i + 1]] - hostGraph->get_host_offset_()[part_vertex_offset[i]];
        max_local_edges =  local_edges_pei > max_local_edges ? local_edges_pei : max_local_edges;
    }

    gpuGraph->set_len_edges_array_(max_local_edges);
    gpuGraph->set_local_vertices_(part_vertex_offset[my_pe + 1] - part_vertex_offset[my_pe]);
    gpuGraph->set_local_edges_(hostGraph->get_host_offset_()[part_vertex_offset[my_pe + 1]] - hostGraph->get_host_offset_()[part_vertex_offset[my_pe]]);

    device_edge = (edge_t *) nvshmem_malloc ((gpuGraph->get_len_edges_array_()) * sizeof(edge_t));
    device_edge_weight = (weight_t *) nvshmem_malloc ((gpuGraph->get_len_edges_array_()) * sizeof(weight_t));

    CUDA_RT_CALL(cudaMemset(device_edge, 0, sizeof(edge_t) * gpuGraph->get_len_edges_array_()));
    CUDA_RT_CALL(cudaMemset(device_edge_weight, 0, sizeof(weight_t) * gpuGraph->get_len_edges_array_()));
    CUDA_RT_CALL(cudaMemcpy(device_edge, hostGraph->get_host_edge_() + hostGraph->get_host_offset_()[part_vertex_offset[my_pe]], sizeof(edge_t) * (gpuGraph->get_local_edges_()),
                            cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(device_edge_weight, hostGraph->get_host_edge_weight_() + hostGraph->get_host_offset_()[part_vertex_offset[my_pe]], sizeof(weight_t) * (gpuGraph->get_local_edges_()),
                            cudaMemcpyHostToDevice));
    gpuGraph->set_private_device_edge_(device_edge);
    gpuGraph->set_private_device_edge_weight_(device_edge_weight);

    CUDA_RT_CALL(cudaMemcpy(device_offset, hostGraph->get_host_offset_() + part_vertex_offset[my_pe], sizeof(vertex_t) * (gpuGraph->get_local_vertices_() + 1),
                            cudaMemcpyHostToDevice));
    sub_first_ele<<<iDivUp((gpuGraph->get_local_vertices_() + 1), 256), 256>>>(device_offset, hostGraph->get_host_offset_()[part_vertex_offset[my_pe]], gpuGraph->get_local_vertices_() + 1);

    CUDA_RT_CALL(cudaFree(device_offset_copy));
}

void edge_partition::partitioner_intra_loop(HostGraph *hostGraph, GpuGraph* gpuGraph, int n_pes, int my_pe) {
    auto total_vertices = hostGraph->get_total_vertices_();
    auto total_edges = hostGraph->get_total_edge_();

    auto *private_device_offset = gpuGraph->get_private_device_offset_();
    auto *private_device_edge = gpuGraph->get_private_device_edge_();
    auto *private_device_edge_weight = gpuGraph->get_private_device_edge_weight_();

    edge_t *part_edge_offset = new edge_t[n_pes + 1];
    auto *part_vertex_offset = gpuGraph->get_part_vertex_offset_();
    auto *device_part_vertex_offset = gpuGraph->get_private_device_part_vertex_offset_();
    edge_t *device_part_edge_offset;

    CUDA_RT_CALL(cudaMalloc((void **) &device_part_edge_offset, sizeof(edge_t) * (n_pes + 1)));
    CUDA_RT_CALL(cudaMemcpy(private_device_offset, hostGraph->get_host_offset_(), sizeof(vertex_t) * (total_vertices + 1), cudaMemcpyHostToDevice));

    part_vertex_offset[0] = 0;
    part_vertex_offset[n_pes] = total_vertices;
    CUDA_RT_CALL(cudaMemcpy(device_part_vertex_offset, part_vertex_offset, sizeof(vertex_t) * (n_pes + 1), cudaMemcpyHostToDevice));
    part_edge_offset[0] = 0;
    for (int i = 0; i < n_pes; i++) {
        part_edge_offset[i + 1] = (part_edge_offset[i] + (total_edges / n_pes + (((total_edges % n_pes) > i) ? 1 : 0)));
    }
    CUDA_RT_CALL(cudaMemcpy(device_part_edge_offset, part_edge_offset, sizeof(edge_t) * (n_pes + 1), cudaMemcpyHostToDevice));

    split<<<iDivUp(total_vertices, 256), 256>>>(private_device_offset, device_part_edge_offset, device_part_vertex_offset,
                                                n_pes, total_vertices);
    CUDA_RT_CALL(cudaMemcpy(part_vertex_offset, device_part_vertex_offset, sizeof(vertex_t) * (n_pes + 1), cudaMemcpyDeviceToHost));

    gpuGraph->set_local_vertices_(part_vertex_offset[my_pe + 1] - part_vertex_offset[my_pe]);
    gpuGraph->set_local_edges_(hostGraph->get_host_offset_()[part_vertex_offset[my_pe + 1]] - hostGraph->get_host_offset_()[part_vertex_offset[my_pe]]);

    assert(gpuGraph->get_local_vertices_() > 0 && gpuGraph->get_local_edges_() > 0 && "error: local vertices or edges less than one");

    CUDA_RT_CALL(cudaMemcpy(private_device_edge, hostGraph->get_host_edge_() + hostGraph->get_host_offset_()[part_vertex_offset[my_pe]], sizeof(edge_t) * (gpuGraph->get_local_edges_()), cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(private_device_edge_weight, hostGraph->get_host_edge_weight_() + hostGraph->get_host_offset_()[part_vertex_offset[my_pe]], sizeof(weight_t) * (gpuGraph->get_local_edges_()), cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(private_device_offset, hostGraph->get_host_offset_() + part_vertex_offset[my_pe], sizeof(vertex_t) * (gpuGraph->get_local_vertices_() + 1),
                            cudaMemcpyHostToDevice));
    sub_first_ele<<<iDivUp((gpuGraph->get_local_vertices_() + 1), 256), 256>>>(private_device_offset, hostGraph->get_host_offset_()[part_vertex_offset[my_pe]], gpuGraph->get_local_vertices_() + 1);

}