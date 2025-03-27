#include "../../include/bin/BIN.cuh"
#include "algorithm"

#define _CG_ABI_EXPERIMENTAL

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

BIN::BIN(int bin_num, vertex_t vertex_num)
{
    bin_size = new vertex_t [bin_num];
    bin_offset = new vertex_t [bin_num];
    CUDA_RT_CALL(cudaMalloc((void **) &device_bin_size, sizeof(vertex_t) * bin_num));
    CUDA_RT_CALL(cudaMalloc((void **) &device_bin_offset, sizeof(vertex_t) * bin_num));
    CUDA_RT_CALL(cudaMalloc((void **) &device_bin_permutation, sizeof(vertex_t) * vertex_num));
    vertex_t* bin_permutation = new vertex_t [vertex_num];
    for(vertex_t i = 0; i < vertex_num; i++) bin_permutation[i] = 0;
    CUDA_RT_CALL(cudaMemcpy(device_bin_permutation, bin_permutation, sizeof(vertex_t) * vertex_num, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMalloc((void **) &device_max_degree, sizeof(vertex_t)));
}

BIN::~BIN() {
    free(bin_size);
    free(bin_offset);
    CUDA_RT_CALL(cudaFree(device_bin_size));
    CUDA_RT_CALL(cudaFree(device_bin_offset));
    CUDA_RT_CALL(cudaFree(device_bin_permutation));
    CUDA_RT_CALL(cudaFree(device_max_degree));
}

template <int bin_num>
__global__ void bin_size_cuda(
    vertex_t *device_bin_size,
    vertex_t *private_device_offset,
    vertex_t vertex_num,
    vertex_t *d_max_degree
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    __shared__ vertex_t shared_bin_size[bin_num];
    __shared__ vertex_t max_degree_block[1];
    vertex_t max_degree_thread = 0;
    vertex_t thread_bin_size[bin_num];
    vertex_t lb;
    vertex_t rb;
    vertex_t neighbors;

    for(int i = 0; i < bin_num; i++) {
        thread_bin_size[i] = 0;
    }

    if (block.thread_rank() < bin_num) {
        shared_bin_size[block.thread_rank()] = 0;
    }

    if (block.thread_rank() == 0) {
        max_degree_block[0] = 0;
    }

    block.sync();

    for (int i = grid.thread_rank(); i < vertex_num; i += grid.num_threads()) {
        lb = private_device_offset[i];
        rb = private_device_offset[i + 1];
        neighbors = rb - lb;
        max_degree_thread = max_degree_thread >= neighbors ? max_degree_thread : neighbors;

        if(neighbors <= 2) {
            thread_bin_size[0] += 1;
        } else if(neighbors <= 4) {
            thread_bin_size[1] += 1;
        } else if(neighbors <= 8) {
            thread_bin_size[2] += 1;
        } else if(neighbors <= 16) {
            thread_bin_size[3] += 1;
        } else if(neighbors <= 32) {
            thread_bin_size[4] += 1;
        } else if(neighbors <= 128) {
            thread_bin_size[5] += 1;
        } else if (neighbors <= 512) {
            thread_bin_size[6] += 1;
        } else if (neighbors <= 2048) {
            thread_bin_size[7] += 1;
        } else if (neighbors <= 4094) {
            thread_bin_size[8] += 1;
        } else {
            thread_bin_size[9] += 1;
        }
    }

    tile32.sync();

    // tile32-wide reduce
#pragma unroll
    for (int i = 0; i < bin_num; i++) {
        thread_bin_size[i] = cg::reduce(tile32, thread_bin_size[i], cg::plus<vertex_t>());
    }
    max_degree_thread = cg::reduce(tile32, max_degree_thread, cg::greater<vertex_t>());

    tile32.sync();

    // block-wide reduce
    if (tile32.thread_rank() < bin_num) {
        atomicAdd(shared_bin_size + tile32.thread_rank(), thread_bin_size[tile32.thread_rank()]);
    }
    if (tile32.thread_rank() == 0) {
        atomicMax(max_degree_block, max_degree_thread);
    }
    block.sync();

    if (block.thread_rank() < bin_num) {
        atomicAdd(device_bin_size + block.thread_rank(), shared_bin_size[block.thread_rank()]);
    }
    if (block.thread_rank() == 0) {
        atomicMax(d_max_degree, max_degree_block[0]);
    }
}

template <int bin_num>
__global__ void bin_size_cuda(
    vertex_t *device_bin_size,
    vertex_t *com_degree,
    vertex_t *d_max_degree,
    vertex_t lb,
    vertex_t rb
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);

    __shared__ vertex_t shared_bin_size[bin_num];
    __shared__ vertex_t max_degree_block[1];
    vertex_t max_degree_thread = 0;
    vertex_t thread_bin_size[bin_num];
    vertex_t degree;
    vertex_t i;

    for(i = 0; i < bin_num; i++) {
        thread_bin_size[i] = 0;
    }
    if (block.thread_rank() < bin_num) {
        shared_bin_size[block.thread_rank()] = 0;
    }
    if (block.thread_rank() == 0) {
        max_degree_block[0] = 0;
    }

    block.sync();

    for (i = lb + grid.thread_rank(); i < rb; i += grid.num_threads()) {
        degree = com_degree[i + 1] - com_degree[i];
        max_degree_thread = max_degree_thread >= degree ? max_degree_thread : degree;

        if (degree <= 32) {
            thread_bin_size[0] += 1;
        } else if (degree <= 64) {
            thread_bin_size[1] += 1;
        } else if (degree <= 128) {
            thread_bin_size[2] += 1;
        } else if (degree <= 256) {
            thread_bin_size[3] += 1;
        } else if (degree <= 512) {
            thread_bin_size[4] += 1;
        } else if (degree <= 1024) {
            thread_bin_size[5] += 1;
        } else if (degree <= 2048) {
            thread_bin_size[6] += 1;
        } else if (degree <= 4096) {
            thread_bin_size[7] += 1;
        } else if (degree <= 8192) {
            thread_bin_size[8] += 1;
        } else if (degree <= 12288) {
            thread_bin_size[9] += 1;
        } else {
            thread_bin_size[10] += 1;
        }
    }

    tile32.sync();

    // tile32-wide reduce
#pragma unroll
    for (i = 0; i < bin_num; i++) {
        thread_bin_size[i] = cg::reduce(tile32, thread_bin_size[i], cg::plus<vertex_t>());
    }
    max_degree_thread = cg::reduce(tile32, max_degree_thread, cg::greater<vertex_t>());

    tile32.sync();

    // block-wide reduce
    if (tile32.thread_rank() < bin_num) {
        atomicAdd(shared_bin_size + tile32.thread_rank(), thread_bin_size[tile32.thread_rank()]);
    }
    if (tile32.thread_rank() == 0) {
        atomicMax(max_degree_block, max_degree_thread);
    }

    block.sync();

    if (block.thread_rank() < bin_num) {
        atomicAdd(device_bin_size + block.thread_rank(), shared_bin_size[block.thread_rank()]);
    }
    if (block.thread_rank() == 0) {
        atomicMax(d_max_degree, max_degree_block[0]);
    }
}

template <int bin_num>
__global__ void bin_size_cuda(
    vertex_t *device_bin_size,
    vertex_t *device_offset_tmp,
    vertex_t *d_max_degree,
    vertex_t local_com_num
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);

    __shared__ vertex_t shared_bin_size[bin_num];
    __shared__ vertex_t max_degree_block[1];
    vertex_t max_degree_thread = 0;
    vertex_t thread_bin_size[bin_num];
    vertex_t degree;
    vertex_t i;
    // init variable
    for(i = 0; i < bin_num; i++) {
        thread_bin_size[i] = 0;
    }
    if (block.thread_rank() < bin_num) {
        shared_bin_size[block.thread_rank()] = 0;
    }
    if (block.thread_rank() == 0) {
        max_degree_block[0] = 0;
    }

    block.sync();

    for (i = grid.thread_rank(); i < local_com_num; i += grid.num_threads()) {
        degree = device_offset_tmp[i + 1] - device_offset_tmp[i];
        max_degree_thread = max_degree_thread >= degree ? max_degree_thread : degree;

        if (degree <= 32) {
            thread_bin_size[0] += 1;
        } else if (degree <= 64) {
            thread_bin_size[1] += 1;
        } else if (degree <= 128) {
            thread_bin_size[2] += 1;
        } else if (degree <= 256) {
            thread_bin_size[3] += 1;
        } else if (degree <= 512) {
            thread_bin_size[4] += 1;
        } else if (degree <= 1024) {
            thread_bin_size[5] += 1;
        } else if (degree <= 2048) {
            thread_bin_size[6] += 1;
        } else if (degree <= 4095) {
            thread_bin_size[7] += 1;
        } else {
            thread_bin_size[8] += 1;
        }
    }

    tile32.sync();

    // tile32-wide reduce
#pragma unroll
    for (i = 0; i < bin_num; i++) {
        thread_bin_size[i] = cg::reduce(tile32, thread_bin_size[i], cg::plus<vertex_t>());
    }
    max_degree_thread = cg::reduce(tile32, max_degree_thread, cg::greater<vertex_t>());

    tile32.sync();

    // block-wide reduce
    if (tile32.thread_rank() < bin_num) {
        atomicAdd(shared_bin_size + tile32.thread_rank(), thread_bin_size[tile32.thread_rank()]);
    }
    if (tile32.thread_rank() == 0) {
        atomicMax(max_degree_block, max_degree_thread);
    }

    block.sync();

    if (block.thread_rank() < bin_num) {
        atomicAdd(device_bin_size + block.thread_rank(), shared_bin_size[block.thread_rank()]);
    }
    if (block.thread_rank() == 0) {
        atomicMax(d_max_degree, max_degree_block[0]);
    }
}

template <int bin_num>
__global__ void bin_permutation_cuda(
    vertex_t *device_bin_size,
    vertex_t *device_bin_offset,
    vertex_t *private_device_offset,
    vertex_t vertex_num,
    vertex_t *device_bin_permutation
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);

    vertex_t neighbors;

    block.sync();

    if (grid.thread_rank() < vertex_num) {
        neighbors = private_device_offset[grid.thread_rank() + 1] - private_device_offset[grid.thread_rank()];
    } else {
        return;
    }

    if (neighbors <= 2) {
        device_bin_permutation[device_bin_offset[0] + atomicAdd(device_bin_size + 0, 1)] = grid.thread_rank();
    } else if (neighbors <= 4) {
        device_bin_permutation[device_bin_offset[1] + atomicAdd(device_bin_size + 1, 1)] = grid.thread_rank();
    } else if (neighbors <= 8) {
        device_bin_permutation[device_bin_offset[2] + atomicAdd(device_bin_size + 2, 1)] = grid.thread_rank();
    } else if (neighbors <= 16) {
        device_bin_permutation[device_bin_offset[3] + atomicAdd(device_bin_size + 3, 1)] = grid.thread_rank();
    } else if (neighbors <= 32) {
        device_bin_permutation[device_bin_offset[4] + atomicAdd(device_bin_size + 4, 1)] = grid.thread_rank();
    } else if (neighbors <= 128) {
        device_bin_permutation[device_bin_offset[5] + atomicAdd(device_bin_size + 5, 1)] = grid.thread_rank();
    } else if (neighbors <= 512) {
        device_bin_permutation[device_bin_offset[6] + atomicAdd(device_bin_size + 6, 1)] = grid.thread_rank();
    } else if (neighbors <= 2048) {
        device_bin_permutation[device_bin_offset[7] + atomicAdd(device_bin_size + 7, 1)] = grid.thread_rank();
    } else if (neighbors <= 4094) {
        device_bin_permutation[device_bin_offset[8] + atomicAdd(device_bin_size + 8, 1)] = grid.thread_rank();
    } else {
        device_bin_permutation[device_bin_offset[9] + atomicAdd(device_bin_size + 9, 1)] = grid.thread_rank();
    }
}

template <int bin_num>
__global__ void bin_permutation_cuda(
    vertex_t *device_bin_size,
    vertex_t *device_bin_offset,
    vertex_t *com_degree,
    vertex_t *device_bin_permutation,
    vertex_t lb,
    vertex_t rb
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    vertex_t degree;

    if (grid.thread_rank() < (rb - lb)) {
        degree = com_degree[lb + grid.thread_rank() + 1] - com_degree[lb + grid.thread_rank()];
    } else {
        return;
    }

    if (degree <= 32) {
        device_bin_permutation[device_bin_offset[0] + atomicAdd(device_bin_size + 0, 1)] = grid.thread_rank();
    } else if (degree <= 64) {
        device_bin_permutation[device_bin_offset[1] + atomicAdd(device_bin_size + 1, 1)] = grid.thread_rank();
    } else if (degree <= 128) {
        device_bin_permutation[device_bin_offset[2] + atomicAdd(device_bin_size + 2, 1)] = grid.thread_rank();
    } else if (degree <= 256) {
        device_bin_permutation[device_bin_offset[3] + atomicAdd(device_bin_size + 3, 1)] = grid.thread_rank();
    } else if (degree <= 512) {
        device_bin_permutation[device_bin_offset[4] + atomicAdd(device_bin_size + 4, 1)] = grid.thread_rank();
    } else if (degree <= 1024) {
        device_bin_permutation[device_bin_offset[5] + atomicAdd(device_bin_size + 5, 1)] = grid.thread_rank();
    } else if (degree <= 2048) {
        device_bin_permutation[device_bin_offset[6] + atomicAdd(device_bin_size + 6, 1)] = grid.thread_rank();
    } else if (degree <= 4096) {
        device_bin_permutation[device_bin_offset[7] + atomicAdd(device_bin_size + 7, 1)] = grid.thread_rank();
    } else if (degree <= 8192) {
        device_bin_permutation[device_bin_offset[8] + atomicAdd(device_bin_size + 8, 1)] = grid.thread_rank();
    } else if (degree <= 12288) {
        device_bin_permutation[device_bin_offset[9] + atomicAdd(device_bin_size + 9, 1)] = grid.thread_rank();
    } else {
        device_bin_permutation[device_bin_offset[10] + atomicAdd(device_bin_size + 10, 1)] = grid.thread_rank();
    }
}

template <int bin_num>
__global__ void bin_permutation_cuda(
        vertex_t *device_bin_size,
        vertex_t *device_bin_offset,
        vertex_t *device_offset_tmp,
        vertex_t *device_bin_permutation,
        vertex_t local_com_num
) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    vertex_t degree;

    __shared__ vertex_t shared_bin_size[bin_num];
    __shared__ vertex_t shared_bin_offset[bin_num];

    if (block.thread_rank() < bin_num) {
        shared_bin_size[block.thread_rank()] = 0;
    }

    block.sync();

    if (grid.thread_rank() < local_com_num) {
        degree = device_offset_tmp[grid.thread_rank() + 1] - device_offset_tmp[grid.thread_rank()];
    } else {
        return;
    }

    if (degree <= 32) {
        atomicAdd(shared_bin_size + 0, 1);
    } else if (degree <= 64) {
        atomicAdd(shared_bin_size + 1, 1);
    } else if (degree <= 128) {
        atomicAdd(shared_bin_size + 2, 1);
    } else if (degree <= 256) {
        atomicAdd(shared_bin_size + 3, 1);
    } else if (degree <= 512) {
        atomicAdd(shared_bin_size + 4, 1);
    } else if (degree <= 1024) {
        atomicAdd(shared_bin_size + 5, 1);
    } else if (degree <= 2048) {
        atomicAdd(shared_bin_size + 6, 1);
    } else if (degree <= 4095) {
        atomicAdd(shared_bin_size + 7, 1);
    } else {
        atomicAdd(shared_bin_size + 8, 1);
    }

    block.sync();

    if (block.thread_rank() < bin_num) {
        shared_bin_offset[block.thread_rank()] = atomicAdd(device_bin_size + block.thread_rank(), shared_bin_size[block.thread_rank()]);
        shared_bin_offset[block.thread_rank()] += device_bin_offset[block.thread_rank()];
        shared_bin_size[block.thread_rank()] = 0;
    }

    block.sync();

    if (degree <= 32) {
        device_bin_permutation[shared_bin_offset[0] + atomicAdd(shared_bin_size + 0, 1)] = grid.thread_rank();
    } else if (degree <= 64) {
        device_bin_permutation[shared_bin_offset[1] + atomicAdd(shared_bin_size + 1, 1)] = grid.thread_rank();
    } else if (degree <= 128) {
        device_bin_permutation[shared_bin_offset[2] + atomicAdd(shared_bin_size + 2, 1)] = grid.thread_rank();
    } else if (degree <= 256) {
        device_bin_permutation[shared_bin_offset[3] + atomicAdd(shared_bin_size + 3, 1)] = grid.thread_rank();
    } else if (degree <= 512) {
        device_bin_permutation[shared_bin_offset[4] + atomicAdd(shared_bin_size + 4, 1)] = grid.thread_rank();
    } else if (degree <= 1024) {
        device_bin_permutation[shared_bin_offset[5] + atomicAdd(shared_bin_size + 5, 1)] = grid.thread_rank();
    } else if (degree <= 2048) {
        device_bin_permutation[shared_bin_offset[6] + atomicAdd(shared_bin_size + 6, 1)] = grid.thread_rank();
    } else if (degree <= 4095) {
        device_bin_permutation[shared_bin_offset[7] + atomicAdd(shared_bin_size + 7, 1)] = grid.thread_rank();
    } else {
        device_bin_permutation[shared_bin_offset[8] + atomicAdd(shared_bin_size + 8, 1)] = grid.thread_rank();
    }
}

void BIN::bin_create(vertex_t *private_device_offset, vertex_t vertex_num) {
    int block_size = 1024;
    int grid_size = iDivUp(vertex_num, block_size);

    for (int i = 0; i < BIN_NUM; i++) {
        bin_size[i] = 0;
        bin_offset[i] = 0;
    }
    max_degree = 0;
    CUDA_RT_CALL(cudaMemcpy(device_bin_size, bin_size, sizeof(vertex_t) * BIN_NUM, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(device_max_degree, &max_degree, sizeof(vertex_t), cudaMemcpyHostToDevice));
    bin_size_cuda<BIN_NUM><<<grid_size, block_size>>>(device_bin_size,
                                                      private_device_offset,
                                                      vertex_num,
                                                      device_max_degree);
    CUDA_RT_CALL(cudaMemcpy(bin_size, device_bin_size, sizeof(vertex_t) * BIN_NUM, cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(device_bin_size, bin_offset, sizeof(vertex_t) * BIN_NUM, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(&max_degree, device_max_degree, sizeof(vertex_t), cudaMemcpyDeviceToHost));

    bin_offset[0] = 0;
    for (int i = 0; i < BIN_NUM - 1; i++) {
        bin_offset[i + 1] = bin_offset[i] + bin_size[i];
    }

    CUDA_RT_CALL(cudaMemcpy(device_bin_offset, bin_offset, sizeof(vertex_t) * BIN_NUM, cudaMemcpyHostToDevice));
    bin_permutation_cuda<BIN_NUM><<<grid_size, block_size>>>(device_bin_size,
                                                            device_bin_offset,
                                                            private_device_offset,
                                                            vertex_num,
                                                            device_bin_permutation);
    CUDA_RT_CALL(cudaDeviceSynchronize());
}

void BIN::bin_create(vertex_t *com_degree, vertex_t lb, vertex_t rb, cudaStream_t default_stream, int my_pe) {
    vertex_t local_com_num = rb - lb;
    int block_num = 512;
    int grid_num = iDivUp(local_com_num, block_num);

    for (int i = 0; i < BIN_NUM_COARSEN_GRAPH; i++) {
        bin_size[i] = 0;
        bin_offset[i] = 0;
    }
    max_degree = 0;
    CUDA_RT_CALL(cudaMemcpy(device_bin_size, bin_size, sizeof(vertex_t) * BIN_NUM_COARSEN_GRAPH, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(device_max_degree, &max_degree, sizeof(vertex_t), cudaMemcpyHostToDevice));
    bin_size_cuda<BIN_NUM_COARSEN_GRAPH><<<grid_num, block_num>>>(device_bin_size,
                                                                 com_degree,
                                                                 device_max_degree,
                                                                 lb,
                                                                 rb);
    CUDA_RT_CALL(cudaDeviceSynchronize());

    CUDA_RT_CALL(cudaMemcpy(bin_size, device_bin_size, sizeof(vertex_t) * BIN_NUM_COARSEN_GRAPH, cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(device_bin_size, bin_offset, sizeof(vertex_t) * BIN_NUM_COARSEN_GRAPH, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(&max_degree, device_max_degree, sizeof(vertex_t), cudaMemcpyDeviceToHost));

    bin_offset[0] = 0;
    for (int i = 0; i < BIN_NUM_COARSEN_GRAPH - 1; i++) {
        bin_offset[i + 1] = bin_offset[i] + bin_size[i];
    }

    CUDA_RT_CALL(cudaMemcpy(device_bin_offset, bin_offset, sizeof(vertex_t) * BIN_NUM_COARSEN_GRAPH, cudaMemcpyHostToDevice));

    bin_permutation_cuda<BIN_NUM_COARSEN_GRAPH><<<grid_num, block_num>>>(device_bin_size,
                                                                        device_bin_offset,
                                                                        com_degree,
                                                                        device_bin_permutation,
                                                                        lb,
                                                                        rb);


    CUDA_RT_CALL(cudaDeviceSynchronize());
}

void BIN::bin_create(vertex_t *device_offset_tmp, vertex_t local_com_num, cudaStream_t default_stream, int my_pe) {
    int block_num = 1024;
    int grid_num = iDivUp(local_com_num, block_num);
    max_degree = 0;
    for (int i = 0; i < BIN_NUM_COARSEN_GRAPH_NUMERIC; i++) {
        bin_size[i] = 0;
        bin_offset[i] = 0;
    }
    CUDA_RT_CALL(cudaMemcpy(device_bin_size, bin_size, sizeof(vertex_t) * BIN_NUM_COARSEN_GRAPH_NUMERIC, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(device_max_degree, &max_degree, sizeof(vertex_t), cudaMemcpyHostToDevice));

    bin_size_cuda<BIN_NUM_COARSEN_GRAPH_NUMERIC><<<grid_num, block_num, 0, default_stream>>>(device_bin_size,
                                                                                             device_offset_tmp,
                                                                                             device_max_degree,
                                                                                             local_com_num);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

    CUDA_RT_CALL(cudaMemcpy(bin_size, device_bin_size, sizeof(vertex_t) * BIN_NUM_COARSEN_GRAPH_NUMERIC, cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(device_bin_size, bin_offset, sizeof(vertex_t) * BIN_NUM_COARSEN_GRAPH_NUMERIC, cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(&max_degree, device_max_degree, sizeof(vertex_t), cudaMemcpyDeviceToHost));

    bin_offset[0] = 0;
    for (int i = 0; i < BIN_NUM_COARSEN_GRAPH_NUMERIC - 1; i++) {
        bin_offset[i + 1] = bin_offset[i] + bin_size[i];
    }
    CUDA_RT_CALL(cudaMemcpy(device_bin_offset, bin_offset, sizeof(vertex_t) * BIN_NUM_COARSEN_GRAPH_NUMERIC, cudaMemcpyHostToDevice));

    bin_permutation_cuda<BIN_NUM_COARSEN_GRAPH_NUMERIC><<<grid_num, block_num, 0, default_stream>>>(device_bin_size,
                                                                                                    device_bin_offset,
                                                                                                    device_offset_tmp,
                                                                                                    device_bin_permutation,
                                                                                                    local_com_num);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
}


