#ifndef EDGE_PARTITION_CUH
#define EDGE_PARTITION_CUH
#include "../graph/host_graph.h"
#include "../graph/gpu_graph.cuh"
#include "../common.h"
namespace edge_partition {
    void partitioner(HostGraph* hostGraph, GpuGraph* gpuGraph, int n_pes, int my_pe);
    void partitioner_intra_loop(HostGraph* hostGraph, GpuGraph* gpuGraph, int n_pes, int my_pe);
}
#endif //EDGE_PARTITION_CUH
