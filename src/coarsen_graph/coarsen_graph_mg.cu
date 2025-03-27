#include "../../include/coarsen_graph/coarsen_graph_mg.cuh"

#define _CG_ABI_EXPERIMENTAL

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <numeric>

namespace cg = cooperative_groups;

template <typename T>
__global__ void copy(
    T* src,
    T* dst,
    vertex_t len)
{
    cg::grid_group grid = cg::this_grid();
    for (vertex_t id = grid.thread_rank(); id < len; id += grid.num_threads()) {
        dst[id] = src[id];
    }
}

__device__ __inline__ void locating_vertex(
    int &pe_dst,
    vertex_t &neighbor_id,
    vertex_t* private_device_part_vertex_offset,
    int n_pes)
{
#pragma unroll
    for (int i = 0; i < n_pes; i++) {
        if (neighbor_id < private_device_part_vertex_offset[i + 1]) {
            pe_dst = i;
            neighbor_id = neighbor_id - private_device_part_vertex_offset[i];
            break;
        }
    }
}

__global__ void set_array(
        vertex_t* array,
        vertex_t value,
        vertex_t len
)
{
    cg::grid_group grid = cg::this_grid();
    for (vertex_t i = grid.thread_rank(); i < len; i += grid.num_threads()) {
        array[i] = value;
    }
}

__global__ void __launch_bounds__(1024, 1)
gather_community_id_cuda(
    vertex_t* shared_device_community_ids_local,
    vertex_t* shared_device_community_ids_global,
    vertex_t local_vertices,
    vertex_t total_vertices,
    vertex_t* device_part_vertex_offset,
    int my_pe,
    int n_pes
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    vertex_t start_vertex_id = device_part_vertex_offset[my_pe];
    vertex_t nelems;
    int i;

    for (vertex_t block_offset = block.group_index().x * block.num_threads(); block_offset < local_vertices; block_offset += grid.num_threads()) {
        nelems = min(block.num_threads(), local_vertices - block_offset);
        for (i = 1; i < n_pes; i++) {
            block.sync();
            nvshmemx_uint32_put_block(shared_device_community_ids_global + start_vertex_id + block_offset, shared_device_community_ids_local + block_offset, nelems, (my_pe + i) % n_pes);
            block.sync();
        }
    }

    for (vertex_t vertex_id = grid.thread_rank(); vertex_id < local_vertices; vertex_id += grid.num_threads()) {
        shared_device_community_ids_global[start_vertex_id + vertex_id] = shared_device_community_ids_local[vertex_id];
    }
}

__global__ void __launch_bounds__(1024, 1)
gather_cuda(
    vertex_t* com_degree,
    vertex_t lb,
    vertex_t rb,
    int my_pe,
    int n_pes
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    vertex_t nelems;
    int i;

    for (vertex_t block_offset = block.group_index().x * block.num_threads(); block_offset < (rb - lb); block_offset += grid.num_threads()) {
        nelems = min(block.num_threads(), (rb - lb) - block_offset);
        for (i = 1; i < n_pes; i++) {
            nvshmemx_uint32_put_block(com_degree + lb + block_offset, com_degree + lb + block_offset, nelems, (my_pe + i) % n_pes);
        }
    }
}

__global__ void cal_degree_of_communities_cuda(
    vertex_t* shared_device_community_ids_global,
    vertex_t total_vertices,
    vertex_t* device_part_vertex_offset,
    vertex_t* com_degree,
    vertex_t* device_offset,
    int my_pe,
    int n_pes,
    vertex_t com_id_lb,
    vertex_t com_id_rb
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    vertex_t vertex_id;
    vertex_t community_id;
    int pe_dst;

    edge_t lb;
    edge_t rb;

    for (vertex_id = grid.thread_rank(); vertex_id < total_vertices; vertex_id += grid.num_threads()) {
        community_id = shared_device_community_ids_global[vertex_id];

        if (community_id >= com_id_lb && community_id < com_id_rb) {
            vertex_t tmp = vertex_id;
            locating_vertex(pe_dst, tmp, device_part_vertex_offset, n_pes);
            lb = nvshmem_uint32_g(device_offset + tmp, pe_dst);
            rb = nvshmem_uint32_g(device_offset + tmp + 1, pe_dst);
            atomicAdd(com_degree + community_id - com_id_lb, rb - lb);
        }
    }
}

__global__ void cal_vertices_num_of_communities_cuda(
    vertex_t* shared_device_community_ids_global,
    vertex_t total_vertices,
    vertex_t* com_size,
    vertex_t com_id_lb,
    vertex_t com_id_rb
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    vertex_t vertex_id;
    vertex_t community_id;

    for (vertex_id = grid.thread_rank(); vertex_id < total_vertices; vertex_id += grid.num_threads()) {
        community_id = shared_device_community_ids_global[vertex_id];
        if (community_id >= com_id_lb && community_id < com_id_rb) {
            atomicAdd(com_size + community_id - com_id_lb, 1);
        }
    }
}

__global__ void reorder_vertices_cuda(
    vertex_t* shared_device_community_ids_global,
    vertex_t total_vertices,
    vertex_t* com_size,
    vertex_t lb,
    vertex_t rb,
    vertex_t* reordered_vertices
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    vertex_t vertex_id;
    vertex_t loc;
    vertex_t community_id;

    for (vertex_id = grid.thread_rank(); vertex_id < total_vertices; vertex_id += grid.num_threads()) {
        community_id = shared_device_community_ids_global[vertex_id];
        if (community_id >= lb && community_id < rb) {
            loc = atomicAdd(com_size + community_id - lb, 1);
            reordered_vertices[loc] = vertex_id;
        }
    }
}

__global__ void com_size_is_empty_cuda(
    vertex_t* shared_device_community_ids_global,
    vertex_t* com_size,
    vertex_t total_vertices,
    vertex_t* device_part_vertex_offset,
    int my_pe
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    vertex_t vertex_id;
    vertex_t community_id;

    vertex_t begin_vertex_id = device_part_vertex_offset[my_pe];
    vertex_t end_vertex_id = device_part_vertex_offset[my_pe + 1];

    for (vertex_id = grid.thread_rank(); vertex_id < total_vertices; vertex_id += grid.num_threads()) {
        community_id = shared_device_community_ids_global[vertex_id];

        if (community_id >= begin_vertex_id && community_id < end_vertex_id) {
            com_size[community_id - begin_vertex_id] = 1;
        }
    }
}

__global__ void com_size_is_empty_all_range_cuda(
    vertex_t* shared_device_community_ids_global,
    vertex_t* com_size,
    vertex_t total_vertices
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    vertex_t vertex_id;
    vertex_t community_id;

    for (vertex_id = grid.thread_rank(); vertex_id < total_vertices; vertex_id += grid.num_threads()) {
        community_id = shared_device_community_ids_global[vertex_id];
        com_size[community_id] = 1;
    }
}

__global__ void new_global_community_id_mapping_cuda(
    vertex_t* com_size,
    vertex_t lb,
    vertex_t local_vertices
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    vertex_t vertex_id;

    for (vertex_id = grid.thread_rank(); vertex_id < local_vertices; vertex_id += grid.num_threads()) {
        com_size[vertex_id] += lb;
    }
}

__global__ void renew_community_id_cuda(
    vertex_t* shared_device_community_ids_global,
    vertex_t* com_size,
    vertex_t* device_part_vertex_offset,
    vertex_t total_vertices,
    int n_pes
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    vertex_t old_com_id;
    vertex_t new_com_id;
    int pe_dst;
    vertex_t vertex_id;

    for (vertex_id = grid.thread_rank(); vertex_id < total_vertices; vertex_id += grid.num_threads()) {
        old_com_id = shared_device_community_ids_global[vertex_id];
        locating_vertex(pe_dst, old_com_id, device_part_vertex_offset, n_pes);
        new_com_id = nvshmem_uint32_g(com_size + old_com_id, pe_dst);
        shared_device_community_ids_global[vertex_id] = new_com_id - 1;
    }
}

__global__ void renew_community_id_all_range_cuda(
    vertex_t* shared_device_community_ids_global,
    vertex_t* com_size,
    vertex_t total_vertices
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    vertex_t old_com_id;
    vertex_t new_com_id;
    vertex_t vertex_id;

    for (vertex_id = grid.thread_rank(); vertex_id < total_vertices; vertex_id += grid.num_threads()) {
        old_com_id = shared_device_community_ids_global[vertex_id];
        new_com_id = com_size[old_com_id];
        shared_device_community_ids_global[vertex_id] = new_com_id - 1;
    }

}

__global__ void edge_partition(
    vertex_t *com_degree,
    edge_t *device_part_edge_offset,
    vertex_t *device_part_community_offset,
    int n_pes,
    vertex_t community_num)
{

    int my_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (my_id == 0) {
        device_part_community_offset[0] = 0;
        device_part_community_offset[n_pes] = community_num;
    }
    edge_t begin_edge;
    if (my_id >= community_num) return ;
    edge_t lb = com_degree[my_id];
    edge_t rb = com_degree[my_id + 1];
    for(int i = 0; i < n_pes - 1; i++){
        begin_edge = device_part_edge_offset[i + 1];
        if(begin_edge >= lb && begin_edge < rb){
            device_part_community_offset[i + 1] = my_id;
        }
    }
}

template <int HASH_LEN>
__global__ void cal_exact_degree_of_communities_sh_tl_hash(
    vertex_t* com_to_vertex,
    vertex_t* reordered_vertices,
    vertex_t com_id_lb,
    vertex_t* shared_device_community_ids_global,
    vertex_t* device_offset,
    edge_t* device_edge,
    vertex_t* device_part_vertex_offset,
    int my_pe,
    int n_pes,
    vertex_t bin_size,
    vertex_t bin_offset,
    vertex_t* bin_permutation,
    vertex_t* com_degree
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    int tile_num_grid = grid.num_threads() / 32;      /* the total number of tile in the whole grid */
    int tile_id_grid = grid.thread_rank() / 32;      /* the global id of the tile in the grid */

    int hash_len_tile = 32;
    int hash_table_lb = (block.thread_rank() / 32) * hash_len_tile;
    int hash_table_rb = (block.thread_rank() / 32 + 1) * hash_len_tile;
    vertex_t i;
    vertex_t tile_id;
    vertex_t com_id;
    vertex_t vertex_id;
    int pe_dst;
    vertex_t lb;
    vertex_t rb;
    long long unsigned int tmp;
    vertex_t old_tmp;
    vertex_t hash;
    vertex_t degree;
    edge_t e;

    __shared__ vertex_t hash_table_key[HASH_LEN];

    for (tile_id = tile_id_grid; tile_id < bin_size; tile_id += tile_num_grid) {

        for (i = hash_table_lb + tile.thread_rank(); i < hash_table_rb; i += 32) {
            hash_table_key[i] = UINT32_MAX;
        }

        tile.sync();

        com_id = bin_permutation[bin_offset + tile_id];
        degree = 0;

        for (i = com_to_vertex[com_id]; i < com_to_vertex[com_id + 1]; i++) {
            vertex_id = reordered_vertices[i];
            locating_vertex(pe_dst, vertex_id, device_part_vertex_offset, n_pes);

            if (tile.thread_rank() == 0) {
                lb = nvshmem_uint32_g(device_offset + vertex_id, pe_dst);
                rb = nvshmem_uint32_g(device_offset + vertex_id + 1, pe_dst);
            }
            tile.sync();
            lb = tile.shfl(lb, 0);
            rb = tile.shfl(rb, 0);
            tile.sync();

            for (e = lb + tile.thread_rank(); e < rb; e += 32) {

                vertex_t nei_id = nvshmem_uint32_g(device_edge + e, pe_dst);         // access the global id of neighbor
                vertex_t nei_com_id = shared_device_community_ids_global[nei_id];   // access the community id of neighbor
                tmp = nei_com_id * 107;
                hash = tmp % (hash_table_rb - hash_table_lb);

                while (true) {
                    old_tmp = atomicCAS(hash_table_key + hash_table_lb + hash, UINT32_MAX, nei_com_id);
                    if (old_tmp == UINT32_MAX || old_tmp == nei_com_id) {
                        degree = old_tmp == UINT32_MAX ? degree + 1 : degree;
                        break;
                    }
                    hash = (hash + 1) % (hash_table_rb - hash_table_lb);
                }
            }
        }

        tile.sync();

        // count the number of neighboring community
        degree = cg::reduce(tile, degree, cg::plus<vertex_t>());
        if (tile.thread_rank() == 0) {
            com_degree[com_id_lb + com_id] = degree;
        }
        tile.sync();
    }
}

template <int HASH_LEN>
__global__ void cal_exact_degree_of_communities_sh_bk_hash(
    vertex_t* com_to_vertex,
    vertex_t* reordered_vertices,
    vertex_t com_id_lb,
    vertex_t* shared_device_community_ids_global,
    vertex_t* device_offset,
    edge_t* device_edge,
    vertex_t* device_part_vertex_offset,
    int my_pe,
    int n_pes,
    vertex_t bin_size,
    vertex_t bin_offset,
    vertex_t* bin_permutation,
    vertex_t* com_degree
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    int tile_id_block = block.thread_rank() / 32;
    int tile_num_block = block.num_threads() / 32;

    int block_id;
    int i;
    vertex_t com_id;
    long long unsigned int tmp;
    vertex_t old_tmp;
    vertex_t hash;
    vertex_t degree;
    vertex_t lb;
    vertex_t rb;
    vertex_t vertex_id;
    edge_t e;
    int pe_dst;

    __shared__ vertex_t hash_table_key[HASH_LEN];

    for (block_id = block.group_index().x; block_id < bin_size; block_id += grid.num_blocks()) {

        for (i = block.thread_rank(); i < HASH_LEN; i += block.num_threads()) {
            hash_table_key[i] = UINT32_MAX;
        }
        block.sync();
        com_id = bin_permutation[bin_offset + block_id];
        degree = 0;

        for (i = com_to_vertex[com_id] + tile_id_block; i < com_to_vertex[com_id + 1]; i += tile_num_block) {
            vertex_id = reordered_vertices[i];
            locating_vertex(pe_dst, vertex_id, device_part_vertex_offset, n_pes);

            if (tile.thread_rank() == 0) {
                lb = nvshmem_uint32_g(device_offset + vertex_id, pe_dst);
                rb = nvshmem_uint32_g(device_offset + vertex_id + 1, pe_dst);
            }
            tile.sync();
            lb = tile.shfl(lb, 0);
            rb = tile.shfl(rb, 0);
            tile.sync();

            for (e = lb + tile.thread_rank(); e < rb; e += 32) {
                vertex_t nei_id = nvshmem_uint32_g(device_edge + e, pe_dst);         // access the global id of neighbor
                vertex_t nei_com_id = shared_device_community_ids_global[nei_id];   // access the community id of neighbor
                tmp = nei_com_id * 107;
                hash = tmp % HASH_LEN;

                while (true) {
                    old_tmp = atomicCAS(hash_table_key + hash, UINT32_MAX, nei_com_id);
                    if (old_tmp == UINT32_MAX || old_tmp == nei_com_id) {
                        degree = old_tmp == UINT32_MAX ? degree + 1 : degree;
                        break;
                    }
                    hash = (hash + 1) % HASH_LEN;
                }
            }
        }

        block.sync();

        // count the number of neighboring community
        // tile-wide reduce
        degree = cg::reduce(tile, degree, cg::plus<vertex_t>());
        block.sync();
        // block-wide reduce
        if (block.thread_rank() == 0) {
            hash_table_key[0] = 0;
        }
        block.sync();
        if (tile.thread_rank() == 0) {
            atomicAdd(hash_table_key, degree);
        }
        block.sync();
        if (block.thread_rank() == 0) {
            com_degree[com_id_lb + com_id] = hash_table_key[0];
        }
        block.sync();
    }
}

__global__ void cal_exact_degree_of_communities_gl_bk_hash(
    vertex_t* com_to_vertex,
    vertex_t* reordered_vertices,
    vertex_t com_id_lb,
    vertex_t* shared_device_community_ids_global,
    vertex_t* device_offset,
    edge_t* device_edge,
    vertex_t* device_part_vertex_offset,
    int my_pe,
    int n_pes,
    vertex_t bin_size,
    vertex_t bin_offset,
    vertex_t* bin_permutation,
    vertex_t* hash_table_key,
    vertex_t hash_len,
    vertex_t* com_degree
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    int tile_id_block = block.thread_rank() / 32;
    int tile_num_block = block.num_threads() / 32;
    vertex_t hash_table_lb = block.group_index().x * hash_len;
    vertex_t hash_table_rb = (block.group_index().x + 1) * hash_len;

    int block_id;
    int i;
    vertex_t com_id;
    long long unsigned int tmp;
    vertex_t old_tmp;
    vertex_t hash;
    vertex_t degree;
    vertex_t lb;
    vertex_t rb;
    vertex_t vertex_id;
    edge_t e;
    int pe_dst;

    __shared__ vertex_t block_wide_degree_sum[1];

    for (block_id = block.group_index().x; block_id < bin_size; block_id += grid.num_blocks()) {

        for (i = block.thread_rank(); i < hash_len; i += block.num_threads()) {
            hash_table_key[hash_table_lb + i] = UINT32_MAX;
        }
        block.sync();
        com_id = bin_permutation[bin_offset + block_id];
        degree = 0;

        for (i = com_to_vertex[com_id] + tile_id_block; i < com_to_vertex[com_id + 1]; i += tile_num_block) {

            vertex_id = reordered_vertices[i];
            locating_vertex(pe_dst, vertex_id, device_part_vertex_offset, n_pes);

            if (tile.thread_rank() == 0) {
                lb = nvshmem_uint32_g(device_offset + vertex_id, pe_dst);
                rb = nvshmem_uint32_g(device_offset + vertex_id + 1, pe_dst);
            }
            tile.sync();
            lb = tile.shfl(lb, 0);
            rb = tile.shfl(rb, 0);
            tile.sync();

            for (e = lb + tile.thread_rank(); e < rb; e += 32) {
                vertex_t nei_id = nvshmem_uint32_g(device_edge + e, pe_dst);         // access the global id of neighbor
                vertex_t nei_com_id = shared_device_community_ids_global[nei_id];   // access the community id of neighbor
                tmp = nei_com_id * 107;
                hash = tmp % hash_len;

                while (true) {
                    old_tmp = atomicCAS(hash_table_key + hash_table_lb + hash, UINT32_MAX, nei_com_id);
                    if (old_tmp == UINT32_MAX || old_tmp == nei_com_id) {
                        degree = old_tmp == UINT32_MAX ? degree + 1 : degree;
                        break;
                    }
                    hash = (hash + 1) % hash_len;
                }
            }
        }

        block.sync();

        // count the number of neighboring community
        // tile-wide reduce
        degree = cg::reduce(tile, degree, cg::plus<vertex_t>());
        block.sync();
        // block-wide reduce
        if (block.thread_rank() == 0) {
            block_wide_degree_sum[0] = 0;
        }
        block.sync();
        if (tile.thread_rank() == 0) {
            atomicAdd(block_wide_degree_sum, degree);
        }
        block.sync();
        if (block.thread_rank() == 0) {
            com_degree[com_id_lb + com_id] = block_wide_degree_sum[0];
        }
        block.sync();
    }
}

template <int HASH_LEN>
__global__ void cal_exact_degree_of_communities_sh_bk_hash_trial(
    vertex_t* com_to_vertex,
    vertex_t* reordered_vertices,
    vertex_t com_id_lb,
    vertex_t* shared_device_community_ids_global,
    vertex_t* device_offset,
    edge_t* device_edge,
    vertex_t* device_part_vertex_offset,
    int my_pe,
    int n_pes,
    vertex_t bin_size,
    vertex_t bin_offset,
    vertex_t* bin_permutation,
    vertex_t* device_fail_count,
    vertex_t* device_fail_permutation,
    vertex_t* com_degree
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    int tile_id_block = block.thread_rank() / 32;
    int tile_num_block = block.num_threads() / 32;

    long long unsigned int tmp;
    int pe_dst;
    int block_id;
    vertex_t i;
    vertex_t com_id;
    vertex_t old_tmp;
    vertex_t hash;
    vertex_t lb;
    vertex_t rb;
    vertex_t vertex_id;
    vertex_t nei_id, nei_com_id;
    edge_t e;
    const double collision_rate = 0.8;
    vertex_t collision_bound = collision_rate * HASH_LEN;

    __shared__ vertex_t com_degree_sh[1];
    __shared__ vertex_t hash_table_key[HASH_LEN];

    for (block_id = block.group_index().x; block_id < bin_size; block_id += grid.num_blocks()) {

        for (i = block.thread_rank(); i < HASH_LEN; i += block.num_threads()) {
            hash_table_key[i] = UINT32_MAX;
        }
        if (block.thread_rank() == 0) {
            com_degree_sh[0] = 0;
        }
        block.sync();
        com_id = bin_permutation[bin_offset + block_id];

        for (i = com_to_vertex[com_id] + tile_id_block; i < com_to_vertex[com_id + 1]; i += tile_num_block) {
            vertex_id = reordered_vertices[i];
            locating_vertex(pe_dst, vertex_id, device_part_vertex_offset, n_pes);

            if (tile.thread_rank() == 0) {
                lb = nvshmem_uint32_g(device_offset + vertex_id, pe_dst);
                rb = nvshmem_uint32_g(device_offset + vertex_id + 1, pe_dst);
            }
            tile.sync();
            lb = tile.shfl(lb, 0);
            rb = tile.shfl(rb, 0);
            tile.sync();

            for (e = lb + tile.thread_rank(); e < rb; e += 32) {
                nei_id = nvshmem_uint32_g(device_edge + e, pe_dst);         // access the global id of neighbor
                nei_com_id = shared_device_community_ids_global[nei_id];   // access the community id of neighbor
                tmp = nei_com_id * 107;
                hash = tmp % HASH_LEN;

                while (com_degree_sh[0] < collision_bound) {
                    old_tmp = atomicCAS(hash_table_key + hash, UINT32_MAX, nei_com_id);
                    if (old_tmp == UINT32_MAX || old_tmp == nei_com_id) {
                        if (old_tmp == UINT32_MAX) {
                            atomicAdd(com_degree_sh, 1);
                        }
                        break;
                    }
                    hash = (hash + 1) % HASH_LEN;
                }
            }
        }

        block.sync();

        if (com_degree_sh[0] >= collision_bound) {
            if (block.thread_rank() == 0) {
                vertex_t loc = atomicAdd(device_fail_count, 1);
                device_fail_permutation[loc] = com_id;
            }
        } else {
            if (block.thread_rank() == 0) {
                com_degree[com_id_lb + com_id] = com_degree_sh[0];
            }
        }

        block.sync();
    }
}

__global__ void __launch_bounds__(1024, 1)
cal_exact_degree_of_communities_gl_grid_hash_trial(
    vertex_t* com_to_vertex,
    vertex_t* reordered_vertices,
    vertex_t com_id_lb,
    vertex_t* shared_device_community_ids_global,
    vertex_t* device_offset,
    edge_t* device_edge,
    vertex_t* device_part_vertex_offset,
    int my_pe,
    int n_pes,
    vertex_t* hash_table_key,
    vertex_t hash_table_len,
    vertex_t fail_count,
    vertex_t* device_fail_permutation,
    vertex_t* com_degree
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    vertex_t i, j;
    vertex_t com_id;
    vertex_t vertex_id;
    vertex_t lb, rb;
    vertex_t e;
    vertex_t nei_id, nei_com_id;
    vertex_t old_tmp;
    vertex_t hash;
    vertex_t degree;
    long long unsigned int tmp;
    int pe_dst;

    __shared__ vertex_t cache[2];

    for (i = 0; i < fail_count; i++) {          // all threads are assigned to one community

        degree = 0;
        com_id = device_fail_permutation[i];
        if (grid.thread_rank() == 0) {
            com_degree[com_id_lb + com_id] = 0;
        }
        for (j = grid.thread_rank(); j < hash_table_len; j += grid.num_threads()) {
            hash_table_key[j] = UINT32_MAX;
        }
        grid.sync();

        for (j = com_to_vertex[com_id] + block.group_index().x; j < com_to_vertex[com_id + 1]; j += grid.num_blocks()) {    // one block is assigned to one vertex
            vertex_id = reordered_vertices[j];
            locating_vertex(pe_dst, vertex_id, device_part_vertex_offset, n_pes);

            if(block.thread_rank() == 0) {
                lb = nvshmem_uint32_g(device_offset + vertex_id, pe_dst);
                rb = nvshmem_uint32_g(device_offset + vertex_id + 1, pe_dst);
                cache[0] = lb;
                cache[1] = rb;
            }
            block.sync();
            lb = cache[0];
            rb = cache[1];

            for (e = lb + block.thread_rank(); e < rb; e += block.num_threads()) {
                nei_id = nvshmem_uint32_g(device_edge + e, pe_dst);         // access the global id of neighbor
                nei_com_id = shared_device_community_ids_global[nei_id];   // access the community id of neighbor
                tmp = nei_com_id * 107;
                hash = tmp % hash_table_len;

                while (true) {
                    old_tmp = atomicCAS(hash_table_key + hash, UINT32_MAX, nei_com_id);
                    if (old_tmp == UINT32_MAX || old_tmp == nei_com_id) {
                        degree = old_tmp == UINT32_MAX ? degree + 1 : degree;
                        break;
                    }
                    hash = (hash + 1) % hash_table_len;
                }
            }
        }
        grid.sync();

        // count the number of neighboring community
        degree = cg::reduce(tile, degree, cg::plus<vertex_t>());        // tile-wide reduce
        tile.sync();
        if (block.thread_rank() == 0) {                                 // block-wide reduce
            cache[0] = 0;
        }
        block.sync();
        if (tile.thread_rank() == 0) {
            atomicAdd(cache, degree);
        }
        block.sync();
        if (block.thread_rank() == 0) {
            atomicAdd(com_degree + com_id_lb + com_id, cache[0]);
        }
    }
}

__global__ void sub_first(vertex_t *offset, vertex_t first_ele, vertex_t len)
{
    for (vertex_t id = (blockIdx.x * blockDim.x) + threadIdx.x; id < len; id += blockDim.x * gridDim.x) {
        offset[id] -= first_ele;
    }
}

template <int HASH_LEN>
__global__ void aggregate_edges_and_weights_sh_tl(
    vertex_t* com_to_vertex,
    vertex_t* reordered_vertices,
    vertex_t* shared_device_community_ids_global,
    vertex_t* device_part_vertex_offset,
    vertex_t bin_size,
    vertex_t bin_offset,
    vertex_t* bin_permutation,
    int my_pe,
    int n_pes,
    vertex_t* device_offset,
    edge_t* device_edge,
    weight_t* device_edge_weight,
    vertex_t* device_offset_tmp,
    edge_t* device_edge_tmp,
    weight_t* device_edge_weight_tmp
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    int tile_id_block = block.thread_rank() / 32;

    vertex_t hash_table_lb = (block.thread_rank() / 32) * 32;
    vertex_t hash_table_rb = (block.thread_rank() / 32 + 1) * 32;

    vertex_t i;
    vertex_t tile_id;
    vertex_t com_id;
    vertex_t vertex_id;
    int pe_dst;
    vertex_t lb;
    vertex_t rb;
    long long unsigned int tmp;
    vertex_t old_tmp;
    vertex_t hash;
    edge_t e;
    vertex_t nei_id;
    weight_t nei_weight;
    vertex_t nei_com_id;
    vertex_t offset;

    __shared__ vertex_t hash_table_key[HASH_LEN];
    __shared__ weight_t hash_table_value[HASH_LEN];
    __shared__ vertex_t non_entry_num[HASH_LEN / 32];

    for (tile_id = grid.thread_rank() / 32; tile_id < bin_size; tile_id += grid.num_threads() / 32) {

        for (i = hash_table_lb + tile.thread_rank(); i < hash_table_rb; i += 32) {
            hash_table_key[i] = UINT32_MAX;
            hash_table_value[i] = 0.;
        }
        if (tile.thread_rank() == 0) {
            non_entry_num[tile_id_block] = 0;
        }
        tile.sync();
        com_id = bin_permutation[bin_offset + tile_id];
        offset = device_offset_tmp[com_id];

        for (i = com_to_vertex[com_id]; i < com_to_vertex[com_id + 1]; i++) {
            vertex_id = reordered_vertices[i];
            locating_vertex(pe_dst, vertex_id, device_part_vertex_offset, n_pes);

            if (tile.thread_rank() == 0) {
                lb = nvshmem_uint32_g(device_offset + vertex_id, pe_dst);
                rb = nvshmem_uint32_g(device_offset + vertex_id + 1, pe_dst);
            }
            tile.sync();
            lb = tile.shfl(lb, 0);
            rb = tile.shfl(rb, 0);
            tile.sync();

            for (e = lb + tile.thread_rank(); e < rb; e += 32) {
                nei_id = nvshmem_uint32_g(device_edge + e, pe_dst);         // access the global id of neighbor
                nei_weight = nvshmem_double_g(device_edge_weight + e, pe_dst);
                nei_com_id = shared_device_community_ids_global[nei_id];   // access the community id of neighbor
                tmp = nei_com_id * 107;
                hash = tmp % 32;

                while (true) {
                    old_tmp = atomicCAS(hash_table_key + hash_table_lb + hash, UINT32_MAX, nei_com_id);
                    if (old_tmp == UINT32_MAX || old_tmp == nei_com_id) {
                        atomicAdd(hash_table_value + hash_table_lb + hash, nei_weight);
                        break;
                    }
                    hash = (hash + 1) % 32;
                }
            }
        }
        tile.sync();

        // write the entries in hash table into global memory
        for (i = hash_table_lb + tile.thread_rank(); i < hash_table_rb; i += 32) {
            if (hash_table_key[i] != UINT32_MAX) {
                nei_id = atomicAdd(non_entry_num + tile_id_block, 1);   // reuse 'nei_id'
                device_edge_tmp[offset + nei_id] = hash_table_key[i];
                device_edge_weight_tmp[offset + nei_id] = hash_table_value[i];
            }
        }
    }
}

template <int HASH_LEN>
__global__ void aggregate_edges_and_weights_sh_bk(
    vertex_t* com_to_vertex,
    vertex_t* reordered_vertices,
    vertex_t* shared_device_community_ids_global,
    vertex_t* device_part_vertex_offset,
    vertex_t bin_size,
    vertex_t bin_offset,
    vertex_t* bin_permutation,
    int my_pe,
    int n_pes,
    vertex_t* device_offset,
    edge_t* device_edge,
    weight_t* device_edge_weight,
    vertex_t* device_offset_tmp,
    edge_t* device_edge_tmp,
    weight_t* device_edge_weight_tmp
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    int tile_id_block = block.thread_rank() / 32;
    int tile_num_block = block.num_threads() / 32;

    int block_id;
    vertex_t i;
    vertex_t com_id;
    long long unsigned int tmp;
    vertex_t old_tmp;
    vertex_t hash;
    vertex_t lb;
    vertex_t rb;
    vertex_t vertex_id;
    edge_t e;
    int pe_dst;
    vertex_t nei_id;
    weight_t nei_weight;
    vertex_t nei_com_id;
    vertex_t offset;

    __shared__ vertex_t hash_table_key[HASH_LEN];
    __shared__ weight_t hash_table_value[HASH_LEN];
    __shared__ vertex_t non_entry_num[1];

    for (block_id = block.group_index().x; block_id < bin_size; block_id += grid.num_blocks()) {

        for (i = block.thread_rank(); i < HASH_LEN; i += block.num_threads()) {
            hash_table_key[i] = UINT32_MAX;
            hash_table_value[i] = 0.;
        }
        if (block.thread_rank() == 0) {
            non_entry_num[0] = 0;
        }
        block.sync();
        com_id = bin_permutation[bin_offset + block_id];
        offset = device_offset_tmp[com_id];

        for (i = com_to_vertex[com_id] + tile_id_block; i < com_to_vertex[com_id + 1]; i += tile_num_block) {
            vertex_id = reordered_vertices[i];
            locating_vertex(pe_dst, vertex_id, device_part_vertex_offset, n_pes);

            if (tile.thread_rank() == 0) {
                lb = nvshmem_uint32_g(device_offset + vertex_id, pe_dst);
                rb = nvshmem_uint32_g(device_offset + vertex_id + 1, pe_dst);
            }
            tile.sync();
            lb = tile.shfl(lb, 0);
            rb = tile.shfl(rb, 0);
            tile.sync();

            for (e = lb + tile.thread_rank(); e < rb; e += 32) {
                nei_id = nvshmem_uint32_g(device_edge + e, pe_dst);         // access the global id of neighbor
                nei_weight = nvshmem_double_g(device_edge_weight + e, pe_dst);
                nei_com_id = shared_device_community_ids_global[nei_id];   // access the community id of neighbor
                tmp = nei_com_id * 107;
                hash = tmp % HASH_LEN;

                while (true) {
                    old_tmp = atomicCAS(hash_table_key + hash, UINT32_MAX, nei_com_id);
                    if (old_tmp == UINT32_MAX || old_tmp == nei_com_id) {
                        atomicAdd(hash_table_value + hash, nei_weight);
                        break;
                    }
                    hash = (hash + 1) % HASH_LEN;
                }
            }
        }
        block.sync();

        // write the entries in hash table into global memory
        for (i = block.thread_rank(); i < HASH_LEN; i += block.num_threads()) {
            if (hash_table_key[i] != UINT32_MAX) {
                nei_id = atomicAdd(non_entry_num, 1);   // reuse 'nei_id'
                device_edge_tmp[offset + nei_id] = hash_table_key[i];
                device_edge_weight_tmp[offset + nei_id] = hash_table_value[i];
            }
        }

        block.sync();
    }
}

__global__ void __launch_bounds__(1024, 1)
aggregate_edges_and_weights_gl_bk(
    vertex_t* com_to_vertex,
    vertex_t* reordered_vertices,
    vertex_t* shared_device_community_ids_global,
    vertex_t* device_part_vertex_offset,
    vertex_t bin_size,
    vertex_t bin_offset,
    vertex_t* bin_permutation,
    int my_pe,
    int n_pes,
    vertex_t* device_offset,
    edge_t* device_edge,
    weight_t* device_edge_weight,
    vertex_t* device_offset_tmp,
    edge_t* device_edge_tmp,
    weight_t* device_edge_weight_tmp,
    vertex_t* hash_table_key,
    weight_t* hash_table_value,
    vertex_t hash_len
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    int tile_id_block = block.thread_rank() / 32;
    int tile_num_block = block.num_threads() / 32;
    vertex_t hash_table_lb = block.group_index().x * hash_len;
    vertex_t hash_table_rb = (block.group_index().x + 1) * hash_len;

    int block_id;
    vertex_t i;
    vertex_t com_id;
    long long unsigned int tmp;
    vertex_t old_tmp;
    vertex_t hash;
    vertex_t lb, rb;
    vertex_t vertex_id;
    edge_t e;
    int pe_dst;
    vertex_t nei_id;
    weight_t nei_weight;
    vertex_t nei_com_id;
    vertex_t offset;

    __shared__ vertex_t non_entry_num[1];

    for (block_id = block.group_index().x; block_id < bin_size; block_id += grid.num_blocks()) {

        for (i = hash_table_lb + block.thread_rank(); i < hash_table_rb; i += block.num_threads()) {
            hash_table_key[i] = UINT32_MAX;
            hash_table_value[i] = 0.;
        }
        if (block.thread_rank() == 0) {
            non_entry_num[0] = 0;
        }
        block.sync();
        com_id = bin_permutation[bin_offset + block_id];
        offset = device_offset_tmp[com_id];

        for (i = com_to_vertex[com_id] + tile_id_block; i < com_to_vertex[com_id + 1]; i += tile_num_block) {
            vertex_id = reordered_vertices[i];
            locating_vertex(pe_dst, vertex_id, device_part_vertex_offset, n_pes);

            if (tile.thread_rank() == 0) {
                lb = nvshmem_uint32_g(device_offset + vertex_id, pe_dst);
                rb = nvshmem_uint32_g(device_offset + vertex_id + 1, pe_dst);
            }
            tile.sync();
            lb = tile.shfl(lb, 0);
            rb = tile.shfl(rb, 0);
            tile.sync();

            for (e = lb + tile.thread_rank(); e < rb; e += 32) {
                nei_id = nvshmem_uint32_g(device_edge + e, pe_dst);         // access the global id of neighbor
                nei_weight = nvshmem_double_g(device_edge_weight + e, pe_dst);
                nei_com_id = shared_device_community_ids_global[nei_id];   // access the community id of neighbor
                tmp = nei_com_id * 107;
                hash = tmp % hash_len;

                while (true) {
                    old_tmp = atomicCAS(hash_table_key + hash_table_lb + hash, UINT32_MAX, nei_com_id);
                    if (old_tmp == UINT32_MAX || old_tmp == nei_com_id) {
                        atomicAdd(hash_table_value + hash_table_lb + hash, nei_weight);
                        break;
                    }
                    hash = (hash + 1) % hash_len;
                }
            }
        }
        block.sync();

        // write the entries in hash table into global memory
        for (i = hash_table_lb + block.thread_rank(); i < hash_table_rb; i += block.num_threads()) {
            if (hash_table_key[i] != UINT32_MAX) {
                nei_id = atomicAdd(non_entry_num, 1);   // reuse 'nei_id'
                device_edge_tmp[offset + nei_id] = hash_table_key[i];
                device_edge_weight_tmp[offset + nei_id] = hash_table_value[i];
            }
        }

        block.sync();
    }
}

///////// cuda code end ////////////

void bitmap_range(
    vertex_t& range_size,
    vertex_t total_com_num,
    size_t max_com_num_sh_mem,
    int& iter_num)
{
    iter_num = 1;
    range_size = iDivUp(total_com_num, iter_num);
    while (range_size > max_com_num_sh_mem) {
        iter_num++;
        range_size = iDivUp(total_com_num, iter_num);
    }
}

void gather_community_id(
    vertex_t* shared_device_community_ids_local,
    vertex_t* shared_device_community_ids_global,
    vertex_t local_vertices,
    vertex_t total_vertices,
    vertex_t* device_part_vertex_offset,
    int my_pe,
    int n_pes,
    cudaStream_t default_stream)
{
    int block_dims = 512;
    int grid_size;
    size_t d_shared_mem = 0;
    void *kernel_args[] = {
            (void *) &shared_device_community_ids_local,
            (void *) &shared_device_community_ids_global,
            (void *) &local_vertices,
            (void *) &total_vertices,
            (void *) &device_part_vertex_offset,
            (void *) &my_pe,
            (void *) &n_pes,
    };
    NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)gather_community_id_cuda, block_dims, kernel_args, d_shared_mem, &grid_size));
    nvshmem_barrier_all();

    NVSHMEM_CHECK(nvshmemx_collective_launch((void *)gather_community_id_cuda, grid_size, block_dims, kernel_args, d_shared_mem, default_stream));
    nvshmemx_barrier_all_on_stream(default_stream);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream))
}

void cal_degree_of_communities(
    vertex_t* shared_device_community_ids_global,
    vertex_t total_vertices,
    vertex_t* device_part_vertex_offset,
    vertex_t* com_degree,
    vertex_t* device_offset,
    int my_pe,
    int n_pes,
    vertex_t community_num,
    cudaStream_t default_stream
    )
{

    // 1. vertex partition
    vertex_t* new_community_offset = new vertex_t[n_pes + 1];
    new_community_offset[0] = 0;
#pragma unroll
    for (int i = 0; i < n_pes; i++) {
        new_community_offset[i + 1] = community_num / n_pes + (i < (community_num % n_pes) ? 1 : 0);
    }
#pragma unroll
    for (int i = 0; i < n_pes; i++) {
        new_community_offset[i + 1] += new_community_offset[i];
    }

    // 2. To count the number of vertices in each community and calculate the sum
    // of vertex degrees within the community as the upper bound of the new vertex degree.
    int block_num = 512;
    int grid_num = iDivUp(total_vertices, block_num);
    cal_degree_of_communities_cuda<<<grid_num, block_num, 0, default_stream>>>(shared_device_community_ids_global,
                                                                                total_vertices,
                                                                                device_part_vertex_offset,
                                                                                com_degree + new_community_offset[my_pe],
                                                                                device_offset,
                                                                                my_pe,
                                                                                n_pes,
                                                                                new_community_offset[my_pe],
                                                                                new_community_offset[my_pe + 1]);
    nvshmemx_barrier_all_on_stream(default_stream);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

    // 3. gather com_degree
    vertex_t local_com_num = new_community_offset[my_pe + 1] - new_community_offset[my_pe];
    grid_num = iDivUp(local_com_num, block_num);
    gather_cuda<<<grid_num, block_num, 0, default_stream>>>(com_degree, new_community_offset[my_pe], new_community_offset[my_pe + 1], my_pe, n_pes);
    nvshmemx_barrier_all_on_stream(default_stream);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
}

void reorder_vertices(
    vertex_t* com_size,
    vertex_t* com_degree,
    vertex_t total_vertices,
    vertex_t total_edges,
    vertex_t* shared_device_community_ids_global,
    edge_t*& part_edge_offset,
    edge_t*& device_part_edge_offset,
    vertex_t*& part_community_offset,
    vertex_t*& device_part_community_offset,
    vertex_t*& reordered_vertices,
    int my_pe,
    int n_pes,
    vertex_t global_com_num,
    cudaStream_t default_stream
    )
{
    int block_num;
    int grid_num;
    vertex_t local_com_num;

    part_edge_offset = new edge_t[n_pes + 1];
    part_community_offset = new vertex_t[n_pes + 1];
    CUDA_RT_CALL(cudaMalloc((void **) &device_part_community_offset, sizeof(vertex_t) * (n_pes + 1)));
    CUDA_RT_CALL(cudaMalloc((void **) &device_part_edge_offset, sizeof(edge_t) * (n_pes + 1)));

    // 1. partition edge based on com_degree
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, com_degree + 1, com_degree + 1, global_com_num);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, com_degree + 1, com_degree + 1, global_com_num);
    CUDA_RT_CALL(cudaFree(d_temp_storage));

    block_num = 512;
    grid_num = iDivUp(global_com_num, block_num);

    // vertex partition
    part_community_offset[0] = 0;
    for (int i = 0; i < n_pes; i++) {
        part_community_offset[i + 1] = (part_community_offset[i] + (global_com_num / n_pes + (((global_com_num % n_pes) > i) ? 1 : 0)));
    }
    local_com_num = part_community_offset[my_pe + 1] - part_community_offset[my_pe];
    cal_vertices_num_of_communities_cuda<<<grid_num, block_num, 0, default_stream>>>(shared_device_community_ids_global,
                                                                                     total_vertices,
                                                                                     com_size + 2,
                                                                                     part_community_offset[my_pe],
                                                                                     part_community_offset[my_pe + 1]);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, com_size + 2, com_size + 2, local_com_num);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, com_size + 2, com_size + 2, local_com_num);
    CUDA_RT_CALL(cudaFree(d_temp_storage));

    // 4. reorder vertices
    reorder_vertices_cuda<<<grid_num, block_num, 0, default_stream>>>(shared_device_community_ids_global,
                                                                      total_vertices,
                                                                      com_size + 1,
                                                                      part_community_offset[my_pe],
                                                                      part_community_offset[my_pe + 1],
                                                                      reordered_vertices);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
}

void reorder_vertices(
    vertex_t* com_size,
    vertex_t total_vertices,
    vertex_t* shared_device_community_ids_global,
    vertex_t* part_community_offset,
    vertex_t* reordered_vertices,
    int my_pe,
    cudaStream_t default_stream
    )
{
    int block_num;
    int grid_num;
    vertex_t local_com_num = part_community_offset[my_pe + 1] - part_community_offset[my_pe];
    block_num = 1024;
    grid_num = iDivUp(local_com_num, block_num);
    set_array<<<grid_num, block_num, 0, default_stream>>>(com_size, 0, (local_com_num + 2));
    grid_num = iDivUp(total_vertices, block_num);
    cal_vertices_num_of_communities_cuda<<<grid_num, block_num, 0, default_stream>>>(shared_device_community_ids_global,
                                                                                     total_vertices,
                                                                                     com_size + 2,
                                                                                     part_community_offset[my_pe],
                                                                                     part_community_offset[my_pe + 1]);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, com_size + 2, com_size + 2, local_com_num);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, com_size + 2, com_size + 2, local_com_num);
    CUDA_RT_CALL(cudaFree(d_temp_storage));

    grid_num = iDivUp(total_vertices, block_num);
    reorder_vertices_cuda<<<grid_num, block_num, 0, default_stream>>>(shared_device_community_ids_global,
                                                                      total_vertices,
                                                                      com_size + 1,
                                                                      part_community_offset[my_pe],
                                                                      part_community_offset[my_pe + 1],
                                                                      reordered_vertices);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
}

void renumber_community_id(
    vertex_t* shared_device_community_ids_global,
    vertex_t* com_size,
    vertex_t total_vertices,
    int my_pe,
    cudaStream_t default_stream,
    vertex_t& community_num
    )
{
    int block_num = 512;
    int grid_num = iDivUp(total_vertices, block_num);
    com_size_is_empty_all_range_cuda<<<grid_num, block_num, 0, default_stream>>>(shared_device_community_ids_global,
                                                                                 com_size,
                                                                                 total_vertices);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream))

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, com_size, com_size, total_vertices);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, com_size, com_size, total_vertices);
    CUDA_RT_CALL(cudaFree(d_temp_storage));

    CUDA_RT_CALL(cudaMemcpy(&community_num, com_size + total_vertices - 1, sizeof(vertex_t), cudaMemcpyDeviceToHost));

    renew_community_id_all_range_cuda<<<grid_num, block_num, 0, default_stream>>>(shared_device_community_ids_global,
                                                                                  com_size,
                                                                                  total_vertices);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream))


    set_array<<<iDivUp((total_vertices + 2), 512), 512, 0, default_stream>>>(com_size, 0, (total_vertices + 2));
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream))
}

void renumber_community_id_multi_gpu(
    vertex_t* shared_device_community_ids_global,
    vertex_t* com_size,
    vertex_t total_vertices,
    vertex_t local_vertices,
    vertex_t* device_part_vertex_offset,
    int n_pes,
    int my_pe,
    cudaStream_t default_stream
    )
{
    auto *shared_part_vertex_offset_new = (vertex_t *) nvshmem_malloc(sizeof(vertex_t) * (n_pes + 1));
    vertex_t *part_vertex_offset_new = new vertex_t [n_pes + 1];
    for (int i = 0; i < n_pes + 1; i++) {
        part_vertex_offset_new[i] = 0;
    }
    CUDA_RT_CALL(cudaMemcpy(shared_part_vertex_offset_new, part_vertex_offset_new, sizeof(vertex_t) * (n_pes + 1), cudaMemcpyHostToDevice));

    // 2. local community is whether empty or not
    int block_num = 512;
    int grid_num = iDivUp(total_vertices, block_num);
    com_size_is_empty_cuda<<<grid_num, block_num, 0, default_stream>>>(shared_device_community_ids_global,
                                                                        com_size,
                                                                        total_vertices,
                                                                        device_part_vertex_offset,
                                                                        my_pe);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream))

    // 3. prefix sum to get new id of community
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, com_size, com_size, local_vertices);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, com_size, com_size, local_vertices);
    CUDA_RT_CALL(cudaFree(d_temp_storage));

    vertex_t new_com_num;
    CUDA_RT_CALL(cudaMemcpy(&new_com_num, com_size + local_vertices - 1, sizeof(vertex_t), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_pes; i++) {
        nvshmem_uint32_p(shared_part_vertex_offset_new + my_pe + 1, new_com_num, (my_pe + i) % n_pes);
    }
    nvshmem_barrier_all();
    CUDA_RT_CALL(cudaMemcpy(part_vertex_offset_new, shared_part_vertex_offset_new, sizeof(vertex_t) * (n_pes + 1), cudaMemcpyDeviceToHost));

    vertex_t sum = 0;
    for (int i = 1; i < (n_pes + 1); i++) {
        sum += part_vertex_offset_new[i - 1];
        part_vertex_offset_new[i] += sum;
    }
    CUDA_RT_CALL(cudaMemcpy(shared_part_vertex_offset_new, part_vertex_offset_new, sizeof(vertex_t) * (n_pes + 1), cudaMemcpyHostToDevice));

    grid_num = iDivUp(local_vertices, block_num);
    new_global_community_id_mapping_cuda<<<grid_num, block_num, 0, default_stream>>>(com_size, part_vertex_offset_new[my_pe], local_vertices);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream))
    // barrier wait all pes finishing mapping
    nvshmem_barrier_all();

    grid_num = iDivUp(total_vertices, block_num);
    renew_community_id_cuda<<<grid_num, block_num, 0, default_stream>>>(shared_device_community_ids_global,
                                                                        com_size,
                                                                        device_part_vertex_offset,
                                                                        total_vertices,
                                                                        n_pes);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream))
}

void bin_create(
    BIN*& bins,
    vertex_t* com_degree,
    vertex_t com_id_lb,
    vertex_t com_id_rb,
    cudaStream_t default_stream,
    int my_pe
    )
{
    bins = new BIN(BIN_NUM_COARSEN_GRAPH, com_id_rb - com_id_lb);
    bins->bin_create(com_degree, com_id_lb, com_id_rb, default_stream, my_pe);
}

void bin_create(
    BIN*& bins,
    vertex_t* device_offset_tmp,
    vertex_t local_com_num,
    cudaStream_t default_stream,
    int my_pe
    )
{
    bins = new BIN(BIN_NUM_COARSEN_GRAPH_NUMERIC, local_com_num);
    bins->bin_create(device_offset_tmp, local_com_num, default_stream, my_pe);
}

void cal_exact_degree_of_communities(
    BIN*& bins,
    vertex_t* com_size,
    vertex_t* reordered_vertices,
    vertex_t* part_community_offset,
    vertex_t* shared_device_community_ids_global,
    vertex_t* device_offset,
    edge_t* device_edge,
    vertex_t* device_part_vertex_offset,
    int my_pe,
    int n_pes,
    vertex_t* com_degree,
    cudaStream_t default_stream,
    cudaStream_t *streams
    )
{
    vertex_t total_com_num = part_community_offset[n_pes];
    int grid_num;
    int block_num;
//    size_t d_shared_mem;

    vertex_t fail_count = 0;
    vertex_t *device_fail_count;
    vertex_t *device_fail_permutation;
    CUDA_RT_CALL(cudaMalloc((void **) &device_fail_count, sizeof(vertex_t)));
    CUDA_RT_CALL(cudaMalloc((void **) &device_fail_permutation, sizeof(vertex_t) * bins->bin_size[10]));
    CUDA_RT_CALL(cudaMemcpy(device_fail_count, &fail_count, sizeof(vertex_t), cudaMemcpyHostToDevice));

    for (int i = BIN_NUM_COARSEN_GRAPH - 1; i >= 0; i--) {
        if (bins->bin_size[i] > 0) {
            switch (i) {
                case 0:
                    block_num = 512;
                    grid_num = iDivUp(bins->bin_size[i], block_num / 32);
                    cal_exact_degree_of_communities_sh_tl_hash<512><<<grid_num, block_num, 0, streams[6]>>>(com_size,
                                                                                                             reordered_vertices,
                                                                                                             part_community_offset[my_pe],
                                                                                                             shared_device_community_ids_global,
                                                                                                             device_offset,
                                                                                                             device_edge,
                                                                                                             device_part_vertex_offset,
                                                                                                             my_pe,
                                                                                                             n_pes,
                                                                                                             bins->bin_size[i],
                                                                                                             bins->bin_offset[i],
                                                                                                             bins->device_bin_permutation,
                                                                                                             com_degree);
                    break;
                case 1:
                    block_num = 32;
                    grid_num = bins->bin_size[i];
                    cal_exact_degree_of_communities_sh_bk_hash<64><<<grid_num, block_num, 0, streams[7]>>>(com_size,
                                                                                                          reordered_vertices,
                                                                                                          part_community_offset[my_pe],
                                                                                                          shared_device_community_ids_global,
                                                                                                          device_offset,
                                                                                                          device_edge,
                                                                                                          device_part_vertex_offset,
                                                                                                          my_pe,
                                                                                                          n_pes,
                                                                                                          bins->bin_size[i],
                                                                                                          bins->bin_offset[i],
                                                                                                          bins->device_bin_permutation,
                                                                                                          com_degree);
                    break;
                case 2:
                    block_num = 64;
                    grid_num = bins->bin_size[i];
                    cal_exact_degree_of_communities_sh_bk_hash<128><<<grid_num, block_num, 0, streams[0]>>>(com_size,
                                                                                                           reordered_vertices,
                                                                                                           part_community_offset[my_pe],
                                                                                                           shared_device_community_ids_global,
                                                                                                           device_offset,
                                                                                                           device_edge,
                                                                                                           device_part_vertex_offset,
                                                                                                           my_pe,
                                                                                                           n_pes,
                                                                                                           bins->bin_size[i],
                                                                                                           bins->bin_offset[i],
                                                                                                           bins->device_bin_permutation,
                                                                                                           com_degree);
                    break;
                case 3:
                    block_num = 128;
                    grid_num = bins->bin_size[i];
                    cal_exact_degree_of_communities_sh_bk_hash<256><<<grid_num, block_num, 0, streams[1]>>>(com_size,
                                                                                                           reordered_vertices,
                                                                                                           part_community_offset[my_pe],
                                                                                                           shared_device_community_ids_global,
                                                                                                           device_offset,
                                                                                                           device_edge,
                                                                                                           device_part_vertex_offset,
                                                                                                           my_pe,
                                                                                                           n_pes,
                                                                                                           bins->bin_size[i],
                                                                                                           bins->bin_offset[i],
                                                                                                           bins->device_bin_permutation,
                                                                                                           com_degree);
                    break;
                case 4:
                    block_num = 256;
                    grid_num = bins->bin_size[i];
                    cal_exact_degree_of_communities_sh_bk_hash<512><<<grid_num, block_num, 0, streams[2]>>>(com_size,
                                                                                                           reordered_vertices,
                                                                                                           part_community_offset[my_pe],
                                                                                                           shared_device_community_ids_global,
                                                                                                           device_offset,
                                                                                                           device_edge,
                                                                                                           device_part_vertex_offset,
                                                                                                           my_pe,
                                                                                                           n_pes,
                                                                                                           bins->bin_size[i],
                                                                                                           bins->bin_offset[i],
                                                                                                           bins->device_bin_permutation,
                                                                                                           com_degree);
                    break;
                case 5:
                    block_num = 512;
                    grid_num = bins->bin_size[i];
                    cal_exact_degree_of_communities_sh_bk_hash<1024><<<grid_num, block_num, 0, streams[3]>>>(com_size,
                                                                                                            reordered_vertices,
                                                                                                            part_community_offset[my_pe],
                                                                                                            shared_device_community_ids_global,
                                                                                                            device_offset,
                                                                                                            device_edge,
                                                                                                            device_part_vertex_offset,
                                                                                                            my_pe,
                                                                                                            n_pes,
                                                                                                            bins->bin_size[i],
                                                                                                            bins->bin_offset[i],
                                                                                                            bins->device_bin_permutation,
                                                                                                            com_degree);
                    break;
                case 6:
                    block_num = 1024;
                    grid_num = bins->bin_size[i];
                    cal_exact_degree_of_communities_sh_bk_hash<2048><<<grid_num, block_num, 0, streams[4]>>>(com_size,
                                                                                                            reordered_vertices,
                                                                                                            part_community_offset[my_pe],
                                                                                                            shared_device_community_ids_global,
                                                                                                            device_offset,
                                                                                                            device_edge,
                                                                                                            device_part_vertex_offset,
                                                                                                            my_pe,
                                                                                                            n_pes,
                                                                                                            bins->bin_size[i],
                                                                                                            bins->bin_offset[i],
                                                                                                            bins->device_bin_permutation,
                                                                                                            com_degree);
                    break;
                case 7:
                    block_num = 1024;
                    grid_num = bins->bin_size[i];
                    cal_exact_degree_of_communities_sh_bk_hash<4096><<<grid_num, block_num, 0, streams[5]>>>(com_size,
                                                                                                            reordered_vertices,
                                                                                                            part_community_offset[my_pe],
                                                                                                            shared_device_community_ids_global,
                                                                                                            device_offset,
                                                                                                            device_edge,
                                                                                                            device_part_vertex_offset,
                                                                                                            my_pe,
                                                                                                            n_pes,
                                                                                                            bins->bin_size[i],
                                                                                                            bins->bin_offset[i],
                                                                                                            bins->device_bin_permutation,
                                                                                                            com_degree);
                    break;
                case 8:
                    block_num = 1024;
                    grid_num = bins->bin_size[i];
                    cal_exact_degree_of_communities_sh_bk_hash<8192><<<grid_num, block_num, 0, streams[6]>>>(com_size,
                                                                                                            reordered_vertices,
                                                                                                            part_community_offset[my_pe],
                                                                                                            shared_device_community_ids_global,
                                                                                                            device_offset,
                                                                                                            device_edge,
                                                                                                            device_part_vertex_offset,
                                                                                                            my_pe,
                                                                                                            n_pes,
                                                                                                            bins->bin_size[i],
                                                                                                            bins->bin_offset[i],
                                                                                                            bins->device_bin_permutation,
                                                                                                            com_degree);
                    break;
                case 9:
                    block_num = 1024;
                    grid_num = bins->bin_size[i];
                    cal_exact_degree_of_communities_sh_bk_hash<12288><<<grid_num, block_num, 0, streams[7]>>>(com_size,
                                                                                                                  reordered_vertices,
                                                                                                                  part_community_offset[my_pe],
                                                                                                                  shared_device_community_ids_global,
                                                                                                                  device_offset,
                                                                                                                  device_edge,
                                                                                                                  device_part_vertex_offset,
                                                                                                                  my_pe,
                                                                                                                  n_pes,
                                                                                                                  bins->bin_size[i],
                                                                                                                  bins->bin_offset[i],
                                                                                                                  bins->device_bin_permutation,
                                                                                                                  com_degree);
                    break;
                case 10:
//                    //  version 1:
//                    int SMs = 80;
//                    int hash_table_num = min(SMs, bins->bin_size[i]);
//                    int hash_table_len_total;
//                    vertex_t * hash_table_key;
//                    vertex_t hash_table_len = 8192;
//                    while (hash_table_len <= bins->max_degree) {
//                        hash_table_len *= 2;
//                    }
//                    hash_table_len = hash_table_len > total_com_num ? total_com_num : hash_table_len;
//                    hash_table_len_total = hash_table_num * hash_table_len;
//                    printf("pe %d, max_degree: %d, hash_table_len: %d, com_num: %d\n", my_pe, bins->max_degree, hash_table_len, total_com_num);
//                    CUDA_RT_CALL(cudaMalloc((void **) &hash_table_key, sizeof(vertex_t) * hash_table_len_total));
//                    block_num = 1024;
//                    grid_num = hash_table_num;
//                    cal_exact_degree_of_communities_gl_bk_hash<<<grid_num, block_num, 0, default_stream>>>(com_size,
//                                                                                                      reordered_vertices,
//                                                                                                      part_community_offset[my_pe],
//                                                                                                      shared_device_community_ids_global,
//                                                                                                      device_offset,
//                                                                                                      device_edge,
//                                                                                                      device_part_vertex_offset,
//                                                                                                      my_pe,
//                                                                                                      n_pes,
//                                                                                                      bins->bin_size[i],
//                                                                                                      bins->bin_offset[i],
//                                                                                                      bins->device_bin_permutation,
//                                                                                                      hash_table_key,
//                                                                                                      hash_table_len,
//                                                                                                      com_degree);
//
//                    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
//                    CUDA_RT_CALL(cudaFree(hash_table_key));

                    //  version 2: try insert into shared memory
                    block_num = 1024;
                    grid_num = bins->bin_size[i];
                    cal_exact_degree_of_communities_sh_bk_hash_trial<12286><<<grid_num, block_num, 0, streams[8]>>>(com_size,
                                                                                                                       reordered_vertices,
                                                                                                                       part_community_offset[my_pe],
                                                                                                                       shared_device_community_ids_global,
                                                                                                                       device_offset,
                                                                                                                       device_edge,
                                                                                                                       device_part_vertex_offset,
                                                                                                                       my_pe,
                                                                                                                       n_pes,
                                                                                                                       bins->bin_size[i],
                                                                                                                       bins->bin_offset[i],
                                                                                                                       bins->device_bin_permutation,
                                                                                                                       device_fail_count,
                                                                                                                       device_fail_permutation,
                                                                                                                       com_degree);
                    break;
            }
        }
    }

    for (int i = 0; i < STREAM_NUM; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    block_num = 1024;
    CUDA_RT_CALL(cudaMemcpy(&fail_count, device_fail_count, sizeof(vertex_t), cudaMemcpyDeviceToHost));
    vertex_t hash_table_len = bins->max_degree * 2;
    hash_table_len = hash_table_len > total_com_num ? total_com_num : hash_table_len;
    vertex_t * hash_table_key = nullptr;
    CUDA_RT_CALL(cudaMalloc((void **) &hash_table_key, sizeof(vertex_t) * hash_table_len));

    void *kernel_args[] = {
            (void *) &com_size,
            (void *) &reordered_vertices,
            (void *) &part_community_offset[my_pe],
            (void *) &shared_device_community_ids_global,
            (void *) &device_offset,
            (void *) &device_edge,
            (void *) &device_part_vertex_offset,
            (void *) &my_pe,
            (void *) &n_pes,
            (void *) &hash_table_key,
            (void *) &hash_table_len,
            (void *) &fail_count,
            (void *) &device_fail_permutation,
            (void *) &com_degree
    };
    NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)cal_exact_degree_of_communities_gl_grid_hash_trial, block_num, kernel_args, 0, &grid_num));
    nvshmem_barrier_all();
    NVSHMEM_CHECK(nvshmemx_collective_launch((void *)cal_exact_degree_of_communities_gl_grid_hash_trial, grid_num, block_num, kernel_args, 0, default_stream));
    nvshmemx_barrier_all_on_stream(default_stream);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
    CUDA_RT_CALL(cudaFree(hash_table_key));
    delete bins;
}

void partition_communities(
    HostGraph* hostGraph,
    GpuGraph* gpuGraph,
    vertex_t* com_degree,
    vertex_t* part_community_offset,
    vertex_t* device_part_community_offset,
    edge_t* part_edge_offset,
    edge_t* device_part_edge_offset,
    int my_pe,
    int n_pes,
    cudaStream_t default_stream
   )
{
    int grid_num;
    int block_num;
    vertex_t new_total_edges;
    vertex_t local_com_num = part_community_offset[my_pe + 1] - part_community_offset[my_pe];
    vertex_t global_com_num = part_community_offset[n_pes];
    block_num = 1024;
    grid_num = iDivUp(local_com_num, block_num);
    gather_cuda<<<grid_num, block_num, 0, default_stream>>>(com_degree + 1,
                                                           part_community_offset[my_pe],
                                                           part_community_offset[my_pe + 1],
                                                           my_pe,
                                                           n_pes);
    nvshmemx_barrier_all_on_stream(default_stream);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, com_degree + 1, com_degree + 1, global_com_num);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, com_degree + 1, com_degree + 1, global_com_num);
    CUDA_RT_CALL(cudaFree(d_temp_storage));
    CUDA_RT_CALL(cudaMemcpy(&new_total_edges, com_degree + global_com_num, sizeof(vertex_t), cudaMemcpyDeviceToHost));

    part_edge_offset[0] = 0;
    for (int i = 0; i < n_pes; i++) {
        part_edge_offset[i + 1] = (part_edge_offset[i] + (new_total_edges / n_pes + (((new_total_edges % n_pes) > i) ? 1 : 0)));
    }
    CUDA_RT_CALL(cudaMemcpy(device_part_edge_offset, part_edge_offset, sizeof(edge_t) * (n_pes + 1), cudaMemcpyHostToDevice));
    grid_num = iDivUp(global_com_num, block_num);
    edge_partition<<<grid_num, block_num, 0, default_stream>>>(com_degree,
                                                               device_part_edge_offset,
                                                               device_part_community_offset,
                                                               n_pes,
                                                               global_com_num);
    CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
    CUDA_RT_CALL(cudaMemcpy(part_community_offset, device_part_community_offset, sizeof(vertex_t) * (n_pes + 1), cudaMemcpyDeviceToHost));
    local_com_num = part_community_offset[my_pe + 1] - part_community_offset[my_pe];


    hostGraph->set_total_edge_(new_total_edges);
    gpuGraph->set_total_edges_(new_total_edges);
    hostGraph->set_total_vertices_(part_community_offset[n_pes]);
    gpuGraph->set_total_vertices_(part_community_offset[n_pes]);
    gpuGraph->set_local_vertices_(local_com_num);

    vertex_t first_ele;
    CUDA_RT_CALL(cudaMemcpy(&first_ele, com_degree + part_community_offset[my_pe], sizeof(vertex_t), cudaMemcpyDeviceToHost));
    sub_first<<<iDivUp(local_com_num + 1, 512), 512>>>(com_degree + part_community_offset[my_pe], first_ele, local_com_num + 1);
}

void allocate_new_graph(
    vertex_t*& device_offset_tmp,
    edge_t*& device_edges_tmp,
    weight_t*& device_edges_weight_tmp,
    vertex_t* com_degree,
    vertex_t* part_community_offset,
    int my_pe,
    cudaStream_t default_stream
    )
{
    device_offset_tmp = com_degree + part_community_offset[my_pe];
    vertex_t local_com_num = part_community_offset[my_pe + 1] - part_community_offset[my_pe];
    vertex_t new_local_edges;
    CUDA_RT_CALL(cudaMemcpy(&new_local_edges, device_offset_tmp + local_com_num, sizeof(vertex_t), cudaMemcpyDeviceToHost));

    CUDA_RT_CALL(cudaMalloc((void **) &device_edges_tmp, sizeof(edge_t) * new_local_edges));
    CUDA_RT_CALL(cudaMalloc((void **) &device_edges_weight_tmp, sizeof(weight_t) * new_local_edges));
}

void aggregate_edges_and_weights(
    BIN*& bins,
    vertex_t* com_size,
    vertex_t* reordered_vertices,
    vertex_t* part_community_offset,
    vertex_t* shared_device_community_ids_global,
    vertex_t* device_offset,
    edge_t* device_edge,
    weight_t* device_edge_weight,
    vertex_t* device_part_vertex_offset,
    int my_pe,
    int n_pes,
    vertex_t* device_offset_tmp,
    edge_t* device_edges_tmp,
    weight_t* device_edges_weight_tmp,
    cudaStream_t default_stream,
    cudaStream_t *streams
    )
{
    int grid_num;
    int block_num;
    vertex_t * hash_table_key = nullptr;
    weight_t * hash_table_value = nullptr;

    for (int i = BIN_NUM_COARSEN_GRAPH_NUMERIC - 1; i >= 0; i--) {
        if (bins->bin_size[i] > 0) {
            switch (i) {
                case 0:
                    block_num = 512;
                    grid_num = iDivUp(bins->bin_size[i], block_num / 32);
                    aggregate_edges_and_weights_sh_tl<512><<<grid_num, block_num, 0, streams[0]>>>(com_size,
                                                                                                       reordered_vertices,
                                                                                                       shared_device_community_ids_global,
                                                                                                       device_part_vertex_offset,
                                                                                                       bins->bin_size[i],
                                                                                                       bins->bin_offset[i],
                                                                                                       bins->device_bin_permutation,
                                                                                                       my_pe,
                                                                                                       n_pes,
                                                                                                       device_offset,
                                                                                                       device_edge,
                                                                                                       device_edge_weight,
                                                                                                       device_offset_tmp,
                                                                                                       device_edges_tmp,
                                                                                                       device_edges_weight_tmp);
                    break;
                case 1:
                    block_num = 256;
                    grid_num = bins->bin_size[i];
                    aggregate_edges_and_weights_sh_bk<128><<<grid_num, block_num, 0, streams[1]>>>(com_size,
                                                                                                      reordered_vertices,
                                                                                                      shared_device_community_ids_global,
                                                                                                      device_part_vertex_offset,
                                                                                                      bins->bin_size[i],
                                                                                                      bins->bin_offset[i],
                                                                                                      bins->device_bin_permutation,
                                                                                                      my_pe,
                                                                                                      n_pes,
                                                                                                      device_offset,
                                                                                                      device_edge,
                                                                                                      device_edge_weight,
                                                                                                      device_offset_tmp,
                                                                                                      device_edges_tmp,
                                                                                                      device_edges_weight_tmp);
                    break;
                case 2:
                    block_num = 256;
                    grid_num = bins->bin_size[i];
                    aggregate_edges_and_weights_sh_bk<256><<<grid_num, block_num, 0, streams[2]>>>(com_size,
                                                                                                       reordered_vertices,
                                                                                                       shared_device_community_ids_global,
                                                                                                       device_part_vertex_offset,
                                                                                                       bins->bin_size[i],
                                                                                                       bins->bin_offset[i],
                                                                                                       bins->device_bin_permutation,
                                                                                                       my_pe,
                                                                                                       n_pes,
                                                                                                       device_offset,
                                                                                                       device_edge,
                                                                                                       device_edge_weight,
                                                                                                       device_offset_tmp,
                                                                                                       device_edges_tmp,
                                                                                                       device_edges_weight_tmp);
                    break;
                case 3:
                    block_num = 256;
                    grid_num = bins->bin_size[i];
                    aggregate_edges_and_weights_sh_bk<512><<<grid_num, block_num, 0, streams[3]>>>(com_size,
                                                                                                       reordered_vertices,
                                                                                                       shared_device_community_ids_global,
                                                                                                       device_part_vertex_offset,
                                                                                                       bins->bin_size[i],
                                                                                                       bins->bin_offset[i],
                                                                                                       bins->device_bin_permutation,
                                                                                                       my_pe,
                                                                                                       n_pes,
                                                                                                       device_offset,
                                                                                                       device_edge,
                                                                                                       device_edge_weight,
                                                                                                       device_offset_tmp,
                                                                                                       device_edges_tmp,
                                                                                                       device_edges_weight_tmp);
                    break;
                case 4:
                    block_num = 512;
                    grid_num = bins->bin_size[i];
                    aggregate_edges_and_weights_sh_bk<1024><<<grid_num, block_num, 0, streams[4]>>>(com_size,
                                                                                                       reordered_vertices,
                                                                                                       shared_device_community_ids_global,
                                                                                                       device_part_vertex_offset,
                                                                                                       bins->bin_size[i],
                                                                                                       bins->bin_offset[i],
                                                                                                       bins->device_bin_permutation,
                                                                                                       my_pe,
                                                                                                       n_pes,
                                                                                                       device_offset,
                                                                                                       device_edge,
                                                                                                       device_edge_weight,
                                                                                                       device_offset_tmp,
                                                                                                       device_edges_tmp,
                                                                                                       device_edges_weight_tmp);
                    break;
                case 5:
                    block_num = 512;
                    grid_num = bins->bin_size[i];
                    aggregate_edges_and_weights_sh_bk<2048><<<grid_num, block_num, 0, streams[5]>>>(com_size,
                                                                                                        reordered_vertices,
                                                                                                        shared_device_community_ids_global,
                                                                                                        device_part_vertex_offset,
                                                                                                        bins->bin_size[i],
                                                                                                        bins->bin_offset[i],
                                                                                                        bins->device_bin_permutation,
                                                                                                        my_pe,
                                                                                                        n_pes,
                                                                                                        device_offset,
                                                                                                        device_edge,
                                                                                                        device_edge_weight,
                                                                                                        device_offset_tmp,
                                                                                                        device_edges_tmp,
                                                                                                        device_edges_weight_tmp);
                    break;
                case 6:
                    block_num = 512;
                    grid_num = bins->bin_size[i];
                    aggregate_edges_and_weights_sh_bk<4095><<<grid_num, block_num, 0, streams[6]>>>(com_size,
                                                                                                        reordered_vertices,
                                                                                                        shared_device_community_ids_global,
                                                                                                        device_part_vertex_offset,
                                                                                                        bins->bin_size[i],
                                                                                                        bins->bin_offset[i],
                                                                                                        bins->device_bin_permutation,
                                                                                                        my_pe,
                                                                                                        n_pes,
                                                                                                        device_offset,
                                                                                                        device_edge,
                                                                                                        device_edge_weight,
                                                                                                        device_offset_tmp,
                                                                                                        device_edges_tmp,
                                                                                                        device_edges_weight_tmp);
                    break;
                case 7:
                    block_num = 512;
                    grid_num = bins->bin_size[i];
                    aggregate_edges_and_weights_sh_bk<4095><<<grid_num, block_num, 0, streams[7]>>>(com_size,
                                                                                                        reordered_vertices,
                                                                                                        shared_device_community_ids_global,
                                                                                                        device_part_vertex_offset,
                                                                                                        bins->bin_size[i],
                                                                                                        bins->bin_offset[i],
                                                                                                        bins->device_bin_permutation,
                                                                                                        my_pe,
                                                                                                        n_pes,
                                                                                                        device_offset,
                                                                                                        device_edge,
                                                                                                        device_edge_weight,
                                                                                                        device_offset_tmp,
                                                                                                        device_edges_tmp,
                                                                                                        device_edges_weight_tmp);
                    break;
                case 8:
                    int SMs = 80;
                    int hash_table_num = min(SMs, bins->bin_size[i]);
                    vertex_t max_degree = bins->max_degree * 2;
                    vertex_t len_hash_table = hash_table_num * max_degree;
                    CUDA_RT_CALL(cudaMalloc((void **) &hash_table_key, sizeof(vertex_t) * len_hash_table));
                    CUDA_RT_CALL(cudaMalloc((void **) &hash_table_value, sizeof(weight_t) * len_hash_table));
                    block_num = 1024;
                    grid_num = hash_table_num;
                    aggregate_edges_and_weights_gl_bk<<<grid_num, block_num, 0, streams[8]>>>(com_size,
                                                                                                  reordered_vertices,
                                                                                                  shared_device_community_ids_global,
                                                                                                  device_part_vertex_offset,
                                                                                                  bins->bin_size[i],
                                                                                                  bins->bin_offset[i],
                                                                                                  bins->device_bin_permutation,
                                                                                                  my_pe,
                                                                                                  n_pes,
                                                                                                  device_offset,
                                                                                                  device_edge,
                                                                                                  device_edge_weight,
                                                                                                  device_offset_tmp,
                                                                                                  device_edges_tmp,
                                                                                                  device_edges_weight_tmp,
                                                                                                  hash_table_key,
                                                                                                  hash_table_value,
                                                                                                  max_degree);
                    break;
            }
        }
    }

    for (int i = 0; i < STREAM_NUM; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    CUDA_RT_CALL(cudaFree(hash_table_key));
    CUDA_RT_CALL(cudaFree(hash_table_value));
    nvshmem_barrier_all();
}

void coarsen_graph_mg::coarsen_graph(HostGraph *hostGraph, GpuGraph *gpuGraph, int my_pe, int n_pes, cudaStream_t default_stream, cudaStream_t *streams, double& symbolic_time, double& numeric_time) {
    auto* shared_device_community_ids_local = gpuGraph->get_shared_device_community_ids_();
    auto* shared_device_community_ids_global = gpuGraph->get_shared_device_community_ids_new_();
    auto* part_vertex_offset = gpuGraph->get_part_vertex_offset_();
    auto* device_part_vertex_offset = gpuGraph->get_private_device_part_vertex_offset_();
    auto* device_offset = gpuGraph->get_private_device_offset_();
    auto* device_edge = gpuGraph->get_private_device_edge_();
    auto* device_edge_weight = gpuGraph->get_private_device_edge_weight_();

    auto local_vertices = gpuGraph->get_local_vertices_();
    auto total_vertices = gpuGraph->get_total_vertices_();
    auto total_edges = gpuGraph->get_total_edges_();

    edge_t *part_edge_offset = nullptr;
    edge_t *device_part_edge_offset = nullptr;
    vertex_t* part_community_offset = nullptr;
    vertex_t* device_part_community_offset = nullptr;
    vertex_t* reordered_vertices = nullptr;
    BIN* bins = nullptr;

    vertex_t* device_offset_tmp = nullptr;
    edge_t* device_edges_tmp = nullptr;
    weight_t* device_edges_weight_tmp = nullptr;

    double start = MPI_Wtime();
    double stop;

    vertex_t* com_size;
    vertex_t* com_degree;
    CUDA_RT_CALL(cudaMalloc((void **) &reordered_vertices, sizeof(vertex_t) * total_vertices));
    CUDA_RT_CALL(cudaMalloc((void **) &com_size, sizeof(vertex_t) * (total_vertices + 2)));
    com_degree = (vertex_t *) nvshmem_malloc(sizeof(vertex_t) * total_vertices);

    set_array<<<iDivUp((total_vertices + 2), 512), 512, 0, default_stream>>>(com_size, 0, (total_vertices + 2));
    set_array<<<iDivUp(total_vertices, 512), 512, 0, default_stream>>>(com_degree, 0, total_vertices);

    // Symbolic phase
    // 1. gather community id info from all GPUs
    gather_community_id(shared_device_community_ids_local,
                        shared_device_community_ids_global,
                        local_vertices,
                        total_vertices,
                        device_part_vertex_offset,
                        my_pe,
                        n_pes,
                        default_stream);

    // 2. renumber community id
    vertex_t community_num;
    renumber_community_id(shared_device_community_ids_global,
                          com_size,
                          total_vertices,
                          my_pe,
                          default_stream,
                          community_num);

    // 3. calculate com_degree
    cal_degree_of_communities(shared_device_community_ids_global,
                               total_vertices,
                               device_part_vertex_offset,
                               com_degree + 1,
                               device_offset,
                               my_pe,
                               n_pes,
                               community_num,
                               default_stream);

    // 4. reorder vertices
    reorder_vertices(com_size,
                     com_degree,
                     total_vertices,
                     total_edges,
                     shared_device_community_ids_global,
                     part_edge_offset,
                     device_part_edge_offset,
                     part_community_offset,
                     device_part_community_offset,
                     reordered_vertices,
                     my_pe,
                     n_pes,
                     community_num,
                     default_stream);

    // 5. create bins
    bin_create(bins,
                com_degree,
                part_community_offset[my_pe],
                part_community_offset[my_pe + 1],
                default_stream,
                my_pe);

    // 6. Calculate the exact degree of each community
    cal_exact_degree_of_communities(bins,
                                    com_size,
                                    reordered_vertices,
                                    part_community_offset,
                                    shared_device_community_ids_global,
                                    device_offset,
                                    device_edge,
                                    device_part_vertex_offset,
                                    my_pe,
                                    n_pes,
                                    com_degree + 1,
                                    default_stream,
                                    streams);


//     7. gather com_degree and partition communities again
    partition_communities(hostGraph,
                          gpuGraph,
                          com_degree,
                          part_community_offset,
                          device_part_community_offset,
                          part_edge_offset,
                          device_part_edge_offset,
                          my_pe,
                          n_pes,
                          default_stream);

    // 8. allocate new graph
    device_offset_tmp = com_degree + part_community_offset[my_pe];
    vertex_t local_com_num = part_community_offset[my_pe + 1] - part_community_offset[my_pe];
    vertex_t new_local_edges;
    CUDA_RT_CALL(cudaMemcpy(&new_local_edges, device_offset_tmp + local_com_num, sizeof(vertex_t), cudaMemcpyDeviceToHost));
    gpuGraph->set_local_edges_(new_local_edges);

    CUDA_RT_CALL(cudaMalloc((void **) &device_edges_tmp, sizeof(edge_t) * new_local_edges));
    CUDA_RT_CALL(cudaMalloc((void **) &device_edges_weight_tmp, sizeof(weight_t) * new_local_edges));

    stop = MPI_Wtime();
    symbolic_time += (stop - start);
    start = MPI_Wtime();

    // Numeric phase
    // 9. create bins
    bin_create(bins,
               device_offset_tmp,
               part_community_offset[my_pe + 1] - part_community_offset[my_pe],
               default_stream,
               my_pe);

    // 10. reorder
    reorder_vertices(com_size,
                     total_vertices,
                     shared_device_community_ids_global,
                     part_community_offset,
                     reordered_vertices,
                     my_pe,
                     default_stream);

    // 11. aggregate edges and edge weights
    aggregate_edges_and_weights(bins,
                                com_size,
                                reordered_vertices,
                                part_community_offset,
                                shared_device_community_ids_global,
                                device_offset,
                                device_edge,
                                device_edge_weight,
                                device_part_vertex_offset,
                                my_pe,
                                n_pes,
                                device_offset_tmp,
                                device_edges_tmp,
                                device_edges_weight_tmp,
                                default_stream,
                                streams);


    // copy tmp to device_offset, device_edge, device_edge_weight
    copy<vertex_t><<<80, 1024, 0, default_stream>>>(device_offset_tmp, device_offset, part_community_offset[my_pe + 1] - part_community_offset[my_pe] + 1);
    copy<edge_t><<<80, 1024, 0, default_stream>>>(device_edges_tmp, device_edge, new_local_edges);
    copy<weight_t><<<80, 1024, 0, default_stream>>>(device_edges_weight_tmp, device_edge_weight, new_local_edges);
    std::copy(part_community_offset, part_community_offset + n_pes + 1, part_vertex_offset);
    CUDA_RT_CALL(cudaMemcpy(device_part_vertex_offset, part_vertex_offset, sizeof(vertex_t) * (n_pes + 1), cudaMemcpyHostToDevice));

    nvshmem_free(com_degree);
    CUDA_RT_CALL(cudaFree(com_size));
    CUDA_RT_CALL(cudaFree(device_part_edge_offset));
    CUDA_RT_CALL(cudaFree(device_part_community_offset));
    CUDA_RT_CALL(cudaFree(device_edges_tmp));
    CUDA_RT_CALL(cudaFree(device_edges_weight_tmp));
    CUDA_RT_CALL(cudaFree(reordered_vertices));

    stop = MPI_Wtime();
    numeric_time += (stop - start);
}

