#ifndef MGL_LOUVAIN_BASELINE_CUH
#define MGL_LOUVAIN_BASELINE_CUH
#include "../graph/host_graph.h"
#include "../graph/gpu_graph.cuh"
#include "../common.h"
#include "../partition/edge_partition.cuh"
#include "../bin/BIN.cuh"
#include "../coarsen_graph/coarsen_graph_mg.cuh"

namespace louvain_baseline {
    void run(HostGraph *hostGraph, GpuGraph *gpuGraph, const double threshold, const int max_iter, const int max_phases);
};

#endif //MGL_LOUVAIN_BASELINE_CUH