#ifndef MGL_BIN_CUH
#define MGL_BIN_CUH
#include "../common.h"

#define BIN_NUM 10
#define BIN_NUM_COARSEN_GRAPH 11
#define BIN_NUM_COARSEN_GRAPH_NUMERIC 9

class BIN{
public:
    vertex_t *bin_size;
    vertex_t *bin_offset;
    vertex_t *device_bin_size;
    vertex_t *device_bin_offset;
    vertex_t *device_bin_permutation;
    vertex_t *device_max_degree;
    vertex_t max_degree;

    BIN(int bin_num, vertex_t vertex_num);
    ~BIN();

    void bin_create(vertex_t* private_device_offset, vertex_t vertex_num);
    void bin_create(vertex_t* com_degree, vertex_t lb, vertex_t rb, cudaStream_t default_stream, int my_pe);
    void bin_create(vertex_t* device_offset_tmp, vertex_t local_com_num, cudaStream_t default_stream, int my_pe);
};

#endif //MGL_BIN_CUH
