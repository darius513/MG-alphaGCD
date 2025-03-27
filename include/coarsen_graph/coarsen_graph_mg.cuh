#ifndef MGL_COARSEN_GRAPH_MG_CUH
#define MGL_COARSEN_GRAPH_MG_CUH
#include "../graph/host_graph.h"
#include "../graph/gpu_graph.cuh"
#include "../common.h"
#include "../bin/BIN.cuh"

namespace coarsen_graph_mg {
    void coarsen_graph(HostGraph *hostGraph, GpuGraph *gpuGraph, int my_pe, int n_pes, cudaStream_t default_stream, cudaStream_t *streams, double& symbolic_time, double& numeric_time);
};


#endif //MGL_COARSEN_GRAPH_MG_CUH
