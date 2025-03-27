#include "../../include/louvain/louvain_gl.cuh"
#include "../../include/bin/BIN.cuh"

#define _CG_ABI_EXPERIMENTAL

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace louvain_gl {

__global__ void
init_community_id(
    vertex_t* __restrict__ shared_device_community_ids,
    vertex_t begin_vertex_id,
    vertex_t end_vertex_id,
    vertex_t local_vertices
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    for (vertex_t vertex_id = begin_vertex_id + grid.thread_rank(); vertex_id < end_vertex_id;
         vertex_id += grid.num_threads()) {
        shared_device_community_ids[vertex_id - begin_vertex_id] = vertex_id;
    }
}

__global__ void reduce_vertices_weights(
    vertex_t local_vertex,
    vertex_t* private_device_offset,
    weight_t* private_device_edge_weight,
    weight_t* private_device_vertex_weight)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    __shared__ vertex_t range[(1024 / 32) * 2];
    vertex_t* t_ranges = &range[(block.thread_index().x / 32) * 2];
    auto tile32 = cg::tiled_partition<32>(block);
    int tile32_num_grid = grid.num_threads() / 32;                                                         /* the total number of tile32 in the whole grid */
    int tile32_id_grid = block.num_threads() * block.group_index().x / 32 + block.thread_index().x / 32;       /* the global id of the tile32 in the grid */
    weight_t w = 0;
    edge_t e;
    vertex_t vertex_id;
    edge_t begin_edge;
    edge_t end_edge;

    // a vertex is assigned to a tile32
    for (vertex_id = tile32_id_grid; vertex_id < local_vertex; vertex_id += tile32_num_grid) {
        w = 0;
        if (tile32.thread_rank() < 2) {
            t_ranges[tile32.thread_rank()] = private_device_offset[vertex_id + tile32.thread_rank()];
        }

        tile32.sync();

        begin_edge = t_ranges[0];
        end_edge = t_ranges[1];

        for(e = begin_edge + tile32.thread_rank(); e < end_edge; e += 32){
            w += private_device_edge_weight[e];
        }

        w = cg::reduce(tile32, w, cg::plus<weight_t>());

        if(tile32.thread_rank() == 0) {
            private_device_vertex_weight[vertex_id] = w;
        }

        tile32.sync();
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

__global__ void __launch_bounds__(1024, 1)
compute_modularity(
    weight_t mass,
    vertex_t local_vertices,
    vertex_t* private_device_offset,
    edge_t* private_device_edge,
    weight_t* private_device_edge_weight,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* shared_device_community_ids,
    weight_t* shared_device_community_weight,
    weight_t* shared_sum_community_weight,
    weight_t* shared_sum_internal_edge_weight,
    int my_pe,
    int n_pes,
    weight_t* Q
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    auto tile32 = cg::tiled_partition<32>(block);
    int tile32_num_grid = grid.num_threads() / 32;                                                         /* the total number of tile32 in the whole grid */
    int tile32_id_grid = block.num_threads() * block.group_index().x / 32 + block.thread_index().x / 32;       /* the global id of the tile32 in the grid */

    // Calculate the sum of community weighted degree

    __shared__ weight_t local_sum_degree_squared_tmp[1024 / 32];

    block.sync();

    weight_t sum_degree_squared = 0.0;
    weight_t degree_thread;

    if (grid.thread_rank() == 0)
    {
        shared_sum_community_weight[0] = 0.0;
        shared_sum_internal_edge_weight[0] = 0.0;
    }

    grid.sync();

    for (vertex_t i = grid.thread_rank(); i < local_vertices; i += grid.num_threads()) {
        degree_thread = shared_device_community_weight[i];
        sum_degree_squared += (degree_thread * degree_thread);
    }

    sum_degree_squared = cg::reduce(tile32, sum_degree_squared, cg::plus<weight_t>());

    if (tile32.thread_rank() == 0) {
        local_sum_degree_squared_tmp[tile32.meta_group_rank()] = sum_degree_squared;
    }

    block.sync();

    if (tile32.meta_group_rank() == 0) {
        sum_degree_squared = tile32.thread_rank() < tile32.meta_group_size() ? local_sum_degree_squared_tmp[tile32.thread_rank()] : 0.0;
        sum_degree_squared = cg::reduce(tile32, sum_degree_squared, cg::plus<weight_t>()); // sum over block
        if(tile32.thread_rank() == 0){
            atomicAdd(shared_sum_community_weight, sum_degree_squared);
        }
    }

    grid.sync();

    if(block.group_index().x == 0 && tile32.meta_group_rank() == 0) {
        nvshmemx_double_sum_reduce_warp(NVSHMEM_TEAM_WORLD, shared_sum_community_weight, shared_sum_community_weight, 1);
    }

    grid.sync();


    // Calculate the sum of the weights of inter-community edges

    vertex_t vertex_id;
    edge_t e;
    vertex_t src_community_id;
    vertex_t dst_community_id;
    int pe_dst;
    vertex_t neighbor_id;
    sum_degree_squared = 0.0; // reuse this register

    // a vertex is assigned to a tile32
    for (vertex_id = tile32_id_grid; vertex_id < local_vertices; vertex_id += tile32_num_grid) {
        src_community_id = shared_device_community_ids[vertex_id];

        for (e = private_device_offset[vertex_id] + tile32.thread_rank(); e < private_device_offset[vertex_id + 1]; e += 32) {
            neighbor_id = private_device_edge[e];

            // which GPU this neighbor belongs to, neighbor_id is transformed to neighbor_id_local
            locating_vertex(pe_dst, neighbor_id, private_device_part_vertex_offset, n_pes);

            if (pe_dst == my_pe) {
                dst_community_id = shared_device_community_ids[neighbor_id];
            } else {
                dst_community_id = nvshmem_uint32_g(shared_device_community_ids + neighbor_id, pe_dst);
            }

            if (src_community_id == dst_community_id) {
                sum_degree_squared += private_device_edge_weight[e];
            }
        }
    }

    sum_degree_squared = cg::reduce(tile32, sum_degree_squared, cg::plus<weight_t>());

    if (tile32.thread_rank() == 0) {
        local_sum_degree_squared_tmp[tile32.meta_group_rank()] = sum_degree_squared;
    }

    block.sync();

    if (tile32.meta_group_rank() == 0) {
        sum_degree_squared = tile32.thread_rank() < tile32.meta_group_size() ? local_sum_degree_squared_tmp[tile32.thread_rank()] : 0.0;
        sum_degree_squared = cg::reduce(tile32, sum_degree_squared, cg::plus<weight_t>()); // sum over block
        if(tile32.thread_rank() == 0){
            atomicAdd(shared_sum_internal_edge_weight, sum_degree_squared);
        }
    }

    grid.sync();

    if(block.group_index().x == 0 && tile32.meta_group_rank() == 0) {
        nvshmemx_double_sum_reduce_warp(NVSHMEM_TEAM_WORLD, shared_sum_internal_edge_weight, shared_sum_internal_edge_weight, 1);
    }

    grid.sync();

    // compute Q (modularity)
    if (grid.thread_rank() == 0) {
        Q[0] = shared_sum_internal_edge_weight[0] / mass - shared_sum_community_weight[0] / (mass * mass);
    }

    grid.sync();
}

__global__ void
calculate_eicj_and_move_vertex_hash_gl(
    vertex_t local_vertices,
    vertex_t begin_vertex_id,
    vertex_t* private_device_offset,
    edge_t* private_device_edge,
    weight_t* private_device_edge_weight,
    weight_t* private_device_vertex_weight,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* shared_device_community_ids_new_,
    vertex_t* shared_device_community_ids,
    weight_t* shared_device_community_weight,
    vertex_t* hash_table_key,
    weight_t* hash_table_value,
    weight_t mass,
    int my_pe,
    int n_pes,
    bool up_down
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    int tile32_num_grid = grid.num_threads() / 32;                                                         /* the total number of tile32 in the whole grid */
    int tile32_id_grid = block.num_threads() * block.group_index().x / 32 + block.thread_index().x / 32;       /* the global id of the tile32 in the grid */

    vertex_t vertex_id;
    edge_t e;
    vertex_t src_community_id;
    vertex_t dst_community_id;
    weight_t eici;
    vertex_t hash;
    vertex_t old_tmp;
    weight_t wij;
    weight_t best_modularity;
    weight_t aci;
    weight_t acj;
    int pe_dst;
    vertex_t neighbor_id;

    vertex_t hash_table_lb;
    vertex_t hash_table_rb;
    vertex_t edge_lb;
    vertex_t edge_rb;
    long long unsigned int tmp;

    mass = mass / 2.0;

    __shared__ vertex_t neighbor_community_num[1024 / 32];

    for (vertex_id = tile32_id_grid; vertex_id < local_vertices; vertex_id += tile32_num_grid) {

        eici = 0.;
        best_modularity = 0.0;
        src_community_id = shared_device_community_ids[vertex_id];
        weight_t ki =  private_device_vertex_weight[vertex_id];
        edge_lb = private_device_offset[vertex_id];
        edge_rb = private_device_offset[vertex_id + 1];
        hash_table_lb = edge_lb * 2;   // The space allocated to each vertex in the hash table is equal to twice the number of its adjacent edges.
        hash_table_rb = edge_rb * 2;

        for (e = hash_table_lb + tile32.thread_rank(); e < hash_table_rb; e += 32) {
            hash_table_key[e] = UINT32_MAX;
            hash_table_value[e] = 0.;
        }

        if (tile32.thread_rank() == 0) {
            neighbor_community_num[tile32.meta_group_rank()] = 0;

            dst_community_id = src_community_id; // In order to reuse the register dst_community_id
            locating_vertex(pe_dst, dst_community_id, private_device_part_vertex_offset, n_pes);
            if (pe_dst == my_pe) {
                aci = shared_device_community_weight[dst_community_id];
            } else {
                aci = nvshmem_double_g(shared_device_community_weight + dst_community_id, pe_dst);
            }
        }

        aci = tile32.shfl(aci, 0);  // broadcast aci
        aci -= ki;

        for (e = edge_lb + tile32.thread_rank(); e < edge_rb; e += 32) {
            neighbor_id = private_device_edge[e];
            locating_vertex(pe_dst, neighbor_id, private_device_part_vertex_offset, n_pes);
            dst_community_id = nvshmem_uint32_g(shared_device_community_ids + neighbor_id, pe_dst);

            wij = private_device_edge_weight[e];
            tmp = dst_community_id * 107;
            hash = tmp % (hash_table_rb - hash_table_lb);

            if ((up_down && src_community_id > dst_community_id) || (!up_down && src_community_id < dst_community_id)) {
                // hash table insert
                while (true) {
                    old_tmp = atomicCAS(hash_table_key + hash_table_lb + hash, UINT32_MAX, dst_community_id);
                    if (old_tmp == UINT32_MAX || old_tmp == dst_community_id) {
                        atomicAdd(hash_table_value + hash_table_lb + hash, wij);
                        break;
                    }
                    hash = (hash + 1) % (hash_table_rb - hash_table_lb);
                }
            } else if (src_community_id == dst_community_id && private_device_edge[e] != (vertex_id + begin_vertex_id)) {
                eici += wij;
            }
        }

        tile32.sync();

        // reduce eici
        eici = cg::reduce(tile32, eici, cg::plus<weight_t>());

        // shuffle hash table
        for (e = hash_table_lb + tile32.thread_rank(); e < hash_table_rb; e += 32) {
            if (hash_table_key[e] != UINT32_MAX) {
                old_tmp = atomicAdd(neighbor_community_num + tile32.meta_group_rank(), 1);
                hash_table_key[hash_table_lb + old_tmp] = hash_table_key[e];
                hash_table_value[hash_table_lb + old_tmp] = hash_table_value[e];
            }
        }

//        tile32.sync();

        dst_community_id = src_community_id;

        // move vertex based on best modularity gain
        // iterate all neighbor community
        for (e = hash_table_lb + tile32.thread_rank(); e < hash_table_lb + neighbor_community_num[tile32.meta_group_rank()]; e += 32) {

            src_community_id = hash_table_key[e];   // neighbor community id
            old_tmp = src_community_id;
            locating_vertex(pe_dst, old_tmp, private_device_part_vertex_offset, n_pes);
            if (pe_dst == my_pe) {
                acj = shared_device_community_weight[old_tmp];
            } else {
                acj = nvshmem_double_g(shared_device_community_weight + old_tmp, pe_dst);
            }

            wij = hash_table_value[e];
            wij -= (eici - ki * (aci - acj) / (2. * mass));
            wij /= mass;
            if (wij > best_modularity) {
                dst_community_id = src_community_id;
                best_modularity = wij;
            }
        }

        // Aggregate the dst_community_id and best_modularity results for tile32, and store the results in the thread with lane_id = 0
        for (e = 32 / 2; e > 0; e /= 2) {
            weight_t best_modularity_tmp = tile32.shfl_down(best_modularity, e);    // available only for sizes lower or equal to 32
            vertex_t dst_community_id_tmp = tile32.shfl_down(dst_community_id, e);
            if (best_modularity_tmp > best_modularity) {
                best_modularity = best_modularity_tmp;
                dst_community_id = dst_community_id_tmp;
            }
        }

        if (tile32.thread_rank() == 0) {
            shared_device_community_ids_new_[vertex_id] = dst_community_id;
        }

        tile32.sync();
    }

}

template <int TILE_THREADS>
__global__ void calculate_eicj_and_move_vertex_sh_tile(
    vertex_t begin_vertex_id,
    vertex_t* private_device_offset,
    edge_t* private_device_edge,
    weight_t* private_device_edge_weight,
    weight_t* private_device_vertex_weight,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* shared_device_community_ids_new_,
    vertex_t* shared_device_community_ids,
    weight_t* shared_device_community_weight,
    weight_t mass,
    int my_pe,
    int n_pes,
    bool up_down,
    vertex_t bin_size,
    vertex_t bin_offset,
    vertex_t* bin_permutation
)
{
    int hash_len_tile = TILE_THREADS;   // the size of hash table that each tile has

    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile = cg::tiled_partition<TILE_THREADS>(block);
    int tile_num_grid = grid.num_threads() / TILE_THREADS;      /* the total number of tile in the whole grid */
    int tile_id_grid = grid.thread_rank() / TILE_THREADS;      /* the global id of the tile in the grid */

    vertex_t vertex_id;
    vertex_t tile_id;
    edge_t e;
    vertex_t src_community_id;
    vertex_t dst_community_id;
    weight_t eici;
    vertex_t hash;
    vertex_t old_tmp;
    weight_t wij;
    weight_t best_modularity;
    weight_t aci;
    weight_t acj;
    int pe_dst;
    vertex_t neighbor_id;

    vertex_t hash_table_lb = (block.thread_rank() / TILE_THREADS) * hash_len_tile;
    vertex_t hash_table_rb = (block.thread_rank() / TILE_THREADS + 1) * hash_len_tile;
    vertex_t edge_lb;
    vertex_t edge_rb;
    long long unsigned int tmp;

    mass = mass / 2.0;

    // dynamic shared memory, |hash| = (512 / TILE_THREADS) * hash_len_tile = 512
    extern  __shared__ unsigned char shared_memory[];

    vertex_t* hash_table_key = (vertex_t*) shared_memory;
    weight_t* hash_table_value = (weight_t*) &hash_table_key[(block.num_threads() / TILE_THREADS) * hash_len_tile];


    for (tile_id = tile_id_grid; tile_id < bin_size; tile_id += tile_num_grid) {

        vertex_id = bin_permutation[bin_offset + tile_id];
        eici = 0.;
        best_modularity = 0.0;
        src_community_id = shared_device_community_ids[vertex_id];
        weight_t ki =  private_device_vertex_weight[vertex_id];
        edge_lb = private_device_offset[vertex_id];
        edge_rb = private_device_offset[vertex_id + 1];

        for (e = hash_table_lb + tile.thread_rank(); e < hash_table_rb; e += TILE_THREADS) {
            hash_table_key[e] = UINT32_MAX;
            hash_table_value[e] = 0.;
        }

        tile.sync();

        if (tile.thread_rank() == 0) {

            dst_community_id = src_community_id; // In order to reuse the register dst_community_id
            locating_vertex(pe_dst, dst_community_id, private_device_part_vertex_offset, n_pes);
            if (pe_dst == my_pe) {
                aci = shared_device_community_weight[dst_community_id];
            } else {
                aci = nvshmem_double_g(shared_device_community_weight + dst_community_id, pe_dst);
            }
        }

        tile.sync();

        aci = tile.shfl(aci, 0);  // broadcast aci
        aci -= ki;

        for (e = edge_lb + tile.thread_rank(); e < edge_rb; e += TILE_THREADS) {
            neighbor_id = private_device_edge[e];
            locating_vertex(pe_dst, neighbor_id, private_device_part_vertex_offset, n_pes);
            dst_community_id = nvshmem_uint32_g(shared_device_community_ids + neighbor_id, pe_dst);

            wij = private_device_edge_weight[e];
            tmp = dst_community_id * 107;
            hash = tmp % (hash_table_rb - hash_table_lb);

            if ((up_down && src_community_id > dst_community_id) || (!up_down && src_community_id < dst_community_id)) {
                // hash table insert
                while (true) {
                    old_tmp = atomicCAS(hash_table_key + hash_table_lb + hash, UINT32_MAX, dst_community_id);
                    if (old_tmp == UINT32_MAX || old_tmp == dst_community_id) {
                        atomicAdd(hash_table_value + hash_table_lb + hash, wij);
                        break;
                    }
                    hash = (hash + 1) % (hash_table_rb - hash_table_lb);
                }
            } else if (src_community_id == dst_community_id && private_device_edge[e] != (vertex_id + begin_vertex_id)) {
                eici += wij;
            }
        }

        tile.sync();

        // reduce eici
        eici = cg::reduce(tile, eici, cg::plus<weight_t>());

        dst_community_id = src_community_id;

        // move vertex based on best modularity gain
        // iterate all neighbor community
        for (e = hash_table_lb + tile.thread_rank(); e < hash_table_rb; e += TILE_THREADS) {
            if (hash_table_key[e] != UINT32_MAX) {
                src_community_id = hash_table_key[e];   // neighbor community id
                old_tmp = src_community_id;
                locating_vertex(pe_dst, old_tmp, private_device_part_vertex_offset, n_pes);
                if (pe_dst == my_pe) {
                    acj = shared_device_community_weight[old_tmp];
                } else {
                    acj = nvshmem_double_g(shared_device_community_weight + old_tmp, pe_dst);
                }

                wij = hash_table_value[e];
                wij -= (eici - ki * (aci - acj) / (2. * mass));
                wij /= mass;
                if (up_down) {
                    if ( (wij > best_modularity) || (wij == best_modularity && src_community_id < dst_community_id) ) {
                        dst_community_id = src_community_id;
                        best_modularity = wij;
                    }
                } else {
                    if ( (wij > best_modularity) || (wij == best_modularity && src_community_id > dst_community_id) ) {
                        dst_community_id = src_community_id;
                        best_modularity = wij;
                    }
                }
            }
        }

        tile.sync();

        // Aggregate the dst_community_id and best_modularity results for tile, and store the results in the thread with lane_id = 0
        for (e = TILE_THREADS / 2; e > 0; e /= 2) {
            weight_t best_modularity_tmp = tile.shfl_down(best_modularity, e);    // available only for sizes lower or equal to 32
            vertex_t dst_community_id_tmp = tile.shfl_down(dst_community_id, e);
            if (up_down) {
                if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp < dst_community_id)) {
                    best_modularity = best_modularity_tmp;
                    dst_community_id = dst_community_id_tmp;
                }
            } else {
                if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp > dst_community_id)) {
                    best_modularity = best_modularity_tmp;
                    dst_community_id = dst_community_id_tmp;
                }
            }
        }

        tile.sync();

        if (tile.thread_rank() == 0) {
            shared_device_community_ids_new_[vertex_id] = dst_community_id;
        }

        tile.sync();
    }
}

template <int HASH_LEN>
__global__ void calculate_eicj_and_move_vertex_sh_bk(
    vertex_t begin_vertex_id,
    vertex_t* private_device_offset,
    edge_t* private_device_edge,
    weight_t* private_device_edge_weight,
    weight_t* private_device_vertex_weight,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* shared_device_community_ids_new_,
    vertex_t* shared_device_community_ids,
    weight_t* shared_device_community_weight,
    weight_t mass,
    int my_pe,
    int n_pes,
    bool up_down,
    vertex_t bin_size,
    vertex_t bin_offset,
    vertex_t* bin_permutation,
    vertex_t total_vertices
    )
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    int tile32_num = tile32.meta_group_size();

    vertex_t vertex_id = block.group_index().x;
    vertex_t neighbor_id;

    edge_t e;
    vertex_t src_community_id;
    vertex_t dst_community_id;
    weight_t eici = 0.;
    vertex_t hash;
    vertex_t old_tmp;
    weight_t wij;
    weight_t best_modularity = 0.0;
    weight_t aci;
    weight_t acj;
    int pe_dst;
    long long unsigned int tmp;
    mass = mass / 2.0;

    __shared__ vertex_t hash_table_key[HASH_LEN];
    __shared__ weight_t hash_table_value[HASH_LEN];
    __shared__ weight_t reduce_buffer[2];

    for (e = block.thread_rank(); e < HASH_LEN; e += block.num_threads()) {
        hash_table_key[e] = UINT32_MAX;
        hash_table_value[e] = 0.;
    }

    if(vertex_id >= bin_size) return;

    vertex_id = bin_permutation[bin_offset + vertex_id];

    src_community_id = shared_device_community_ids[vertex_id];
    weight_t ki =  private_device_vertex_weight[vertex_id];

    if (block.thread_rank() == 0) {
        dst_community_id = src_community_id;
        locating_vertex(pe_dst, dst_community_id, private_device_part_vertex_offset, n_pes);
        if (pe_dst == my_pe) {
            aci = shared_device_community_weight[dst_community_id];
        } else {
            aci = nvshmem_double_g(shared_device_community_weight + dst_community_id, pe_dst);
        }
        reduce_buffer[0] = aci;     // write result into shared memory which is visible to all threads within a block
        reduce_buffer[1] = 0.;
    }

    block.sync();

    // broadcast aci among block
    aci = reduce_buffer[0];
    aci -= ki;
    block.sync();

    for (e = private_device_offset[vertex_id] + block.thread_rank(); e < private_device_offset[vertex_id + 1]; e += block.num_threads()) {
        neighbor_id = private_device_edge[e];
        locating_vertex(pe_dst, neighbor_id, private_device_part_vertex_offset, n_pes);
        dst_community_id = nvshmem_uint32_g(shared_device_community_ids + neighbor_id, pe_dst);

        wij = private_device_edge_weight[e];
        tmp = dst_community_id * 107;
        hash = tmp % HASH_LEN;

        if ((up_down && src_community_id > dst_community_id) || (!up_down && src_community_id < dst_community_id)) {
            // hash table insert
            while (true) {
                old_tmp = atomicCAS(hash_table_key + hash, UINT32_MAX, dst_community_id);
                if (old_tmp == UINT32_MAX || old_tmp == dst_community_id) {
                    atomicAdd(hash_table_value + hash, wij);
                    break;
                }
                hash = (hash + 1) % HASH_LEN;
            }
        } else if (src_community_id == dst_community_id && private_device_edge[e] != (vertex_id + begin_vertex_id)) {
            eici += wij;
        }
    }

    block.sync();

    // reduce eici among block
    eici = cg::reduce(tile32, eici, cg::plus<weight_t>());      //reduce eici among tile32
    if (tile32.thread_rank() == 0) {
        atomicAdd(reduce_buffer + 1, eici);
    }
    block.sync();

    eici = reduce_buffer[1];

    dst_community_id = src_community_id;

    // move vertex based on best modularity gain
    // iterate all neighbor community
    for (e = block.thread_rank(); e < HASH_LEN; e += block.num_threads()) {
        if (hash_table_key[e] != UINT32_MAX) {
            src_community_id = hash_table_key[e];   // neighbor community id
            old_tmp = src_community_id;
            locating_vertex(pe_dst, old_tmp, private_device_part_vertex_offset, n_pes);
            if (pe_dst == my_pe) {
                acj = shared_device_community_weight[old_tmp];
            } else {
                acj = nvshmem_double_g(shared_device_community_weight + old_tmp, pe_dst);
            }

            wij = hash_table_value[e];
            wij -= (eici - ki * (aci - acj) / (2. * mass));
            wij /= mass;
            // Add a constraint: when the best modularity is the same, choose the one with the smallest ID.
            if (up_down) {
                if ( (wij > best_modularity) || (wij == best_modularity && src_community_id < dst_community_id) ) {
                    dst_community_id = src_community_id;
                    best_modularity = wij;
                }
            } else {
                if ( (wij > best_modularity) || (wij == best_modularity && src_community_id > dst_community_id) ) {
                    dst_community_id = src_community_id;
                    best_modularity = wij;
                }
            }
        }
    }

    block.sync();

    //  Aggregate the dst_community_id and best_modularity results for block, and store the results in the thread with lane_id = 0

    // tile32-wide reduce result
    for (e = 32 / 2; e > 0; e /= 2) {
        weight_t best_modularity_tmp = tile32.shfl_down(best_modularity, e);    // available only for sizes lower or equal to 32
        vertex_t dst_community_id_tmp = tile32.shfl_down(dst_community_id, e);
        // Add a constraint: when the best modularity is the same, choose the one with the smallest ID.
        if (up_down) {
            if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp < dst_community_id)) {
                best_modularity = best_modularity_tmp;
                dst_community_id = dst_community_id_tmp;
            }
        } else {
            if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp > dst_community_id)) {
                best_modularity = best_modularity_tmp;
                dst_community_id = dst_community_id_tmp;
            }
        }
    }

    tile32.sync();

    // write tile32-wide reduce result into shared memory
    if (tile32.thread_rank() == 0) {
        hash_table_key[tile32.meta_group_rank()] = dst_community_id;
        hash_table_value[tile32.meta_group_rank()] = best_modularity;
    }

    block.sync();

    if (block.thread_rank() < tile32_num) {
        best_modularity = hash_table_value[block.thread_rank()];
        dst_community_id = hash_table_key[block.thread_rank()];
    }

    block.sync();

    if (block.thread_rank() < 32) {
        for (e = 32 / 2; e > 0; e /= 2) {
            weight_t best_modularity_tmp = tile32.shfl_down(best_modularity, e);    // available only for sizes lower or equal to 32
            vertex_t dst_community_id_tmp = tile32.shfl_down(dst_community_id, e);
            // Add a constraint: when the best modularity is the same, choose the one with the smallest ID.
            if (up_down) {
                if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp < dst_community_id)) {
                    best_modularity = best_modularity_tmp;
                    dst_community_id = dst_community_id_tmp;
                }
            } else {
                if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp > dst_community_id)) {
                    best_modularity = best_modularity_tmp;
                    dst_community_id = dst_community_id_tmp;
                }
            }
        }
    }

    block.sync();

    if (block.thread_rank() == 0) {
        shared_device_community_ids_new_[vertex_id] = dst_community_id;
    }
    block.sync();
}


__global__ void __launch_bounds__(1024, 1)
calculate_eicj_and_move_vertex_gl_bk(
    vertex_t begin_vertex_id,
    vertex_t* private_device_offset,
    edge_t* private_device_edge,
    weight_t* private_device_edge_weight,
    weight_t* private_device_vertex_weight,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* shared_device_community_ids_new_,
    vertex_t* shared_device_community_ids,
    weight_t* shared_device_community_weight,
    vertex_t* hash_table_key,
    weight_t* hash_table_value,
    weight_t mass,
    int my_pe,
    int n_pes,
    bool up_down,
    vertex_t bin_size,
    vertex_t bin_offset,
    vertex_t* bin_permutation,
    edge_t HASH_LEN
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    int tile32_num = tile32.meta_group_size();

    vertex_t bk_id = block.group_index().x;
    vertex_t vertex_id;
    vertex_t neighbor_id;

    edge_t e;
    vertex_t src_community_id;
    vertex_t dst_community_id;
    weight_t eici = 0.;
    vertex_t hash;
    vertex_t old_tmp;
    weight_t wij;
    weight_t best_modularity = 0.0;
    weight_t aci;
    weight_t acj;
    int pe_dst;
    long long unsigned int tmp;
    mass = mass / 2.0;

    __shared__ vertex_t buffer_int[1024/32];
    __shared__ weight_t buffer_double[1024/32];

    edge_t hash_table_lb = block.group_index().x * HASH_LEN;

    if(bk_id >= bin_size) return;

    for (; bk_id < bin_size; bk_id += grid.num_blocks()) {
        vertex_id = bin_permutation[bin_offset + bk_id];

        eici = 0.;
        best_modularity = 0.0;
        src_community_id = shared_device_community_ids[vertex_id];
        weight_t ki =  private_device_vertex_weight[vertex_id];

        for (e = block.thread_rank(); e < HASH_LEN; e += block.num_threads()) {
            hash_table_key[hash_table_lb + e] = UINT32_MAX;
            hash_table_value[hash_table_lb + e] = 0.;
        }
        block.sync();

        if (block.thread_rank() == 0) {
            dst_community_id = src_community_id;
            locating_vertex(pe_dst, dst_community_id, private_device_part_vertex_offset, n_pes);
            if (pe_dst == my_pe) {
                aci = shared_device_community_weight[dst_community_id];
            } else {
                aci = nvshmem_double_g(shared_device_community_weight + dst_community_id, pe_dst);
            }
            buffer_double[0] = aci;     // write result into shared memory which is visible to all threads within a block
            buffer_double[1] = 0.;
        }
        block.sync();

        aci = buffer_double[0];
        aci -= ki;
        block.sync();

        for (e = private_device_offset[vertex_id] + block.thread_rank(); e < private_device_offset[vertex_id + 1]; e += block.num_threads()) {
            neighbor_id = private_device_edge[e];
            locating_vertex(pe_dst, neighbor_id, private_device_part_vertex_offset, n_pes);
            dst_community_id = nvshmem_uint32_g(shared_device_community_ids + neighbor_id, pe_dst);

            wij = private_device_edge_weight[e];
            tmp = dst_community_id * 107;
            hash = tmp % HASH_LEN;

            if ((up_down && src_community_id > dst_community_id) || (!up_down && src_community_id < dst_community_id)) {
                // hash table insert
                while (true) {
                    old_tmp = atomicCAS(hash_table_key + hash_table_lb + hash, UINT32_MAX, dst_community_id);
                    if (old_tmp == UINT32_MAX || old_tmp == dst_community_id) {
                        atomicAdd(hash_table_value + hash_table_lb + hash, wij);
                        break;
                    }
                    hash = (hash + 1) % HASH_LEN;
                }
            } else if (src_community_id == dst_community_id && private_device_edge[e] != (vertex_id + begin_vertex_id)) {
                eici += wij;
            }
        }

        block.sync();

        // reduce eici among block
        eici = cg::reduce(tile32, eici, cg::plus<weight_t>());      //reduce eici among tile32
        if (tile32.thread_rank() == 0) {
            atomicAdd(buffer_double + 1, eici);
        }
        block.sync();

        eici = buffer_double[1];

        dst_community_id = src_community_id;

//         move vertex based on best modularity gain
//         iterate all neighbor community
        for (e = hash_table_lb + block.thread_rank(); e < hash_table_lb + HASH_LEN; e += block.num_threads()) {
            if (hash_table_key[e] != UINT32_MAX) {
                src_community_id = hash_table_key[e];   // neighbor community id
                old_tmp = src_community_id;
                locating_vertex(pe_dst, old_tmp, private_device_part_vertex_offset, n_pes);
                if (pe_dst == my_pe) {
                    acj = shared_device_community_weight[old_tmp];
                } else {
                    acj = nvshmem_double_g(shared_device_community_weight + old_tmp, pe_dst);
                }

                wij = hash_table_value[e];
                wij -= (eici - ki * (aci - acj) / (2. * mass));
                wij /= mass;
                // Add a constraint: when the best modularity is the same, choose the one with the smallest ID.
                if (up_down) {
                    if ( (wij > best_modularity) || (wij == best_modularity && src_community_id < dst_community_id) ) {
                        dst_community_id = src_community_id;
                        best_modularity = wij;
                    }
                } else {
                    if ( (wij > best_modularity) || (wij == best_modularity && src_community_id > dst_community_id) ) {
                        dst_community_id = src_community_id;
                        best_modularity = wij;
                    }
                }
//                if (wij > best_modularity) {
//                    dst_community_id = src_community_id;
//                    best_modularity = wij;
//                }
            }
        }

        block.sync();

        // tile32-wide reduce result
        for (e = 32 / 2; e > 0; e /= 2) {
            weight_t best_modularity_tmp = tile32.shfl_down(best_modularity, e);    // available only for sizes lower or equal to 32
            vertex_t dst_community_id_tmp = tile32.shfl_down(dst_community_id, e);
            // Add a constraint: when the best modularity is the same, choose the one with the smallest ID.
            if (up_down) {
                if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp < dst_community_id)) {
                    best_modularity = best_modularity_tmp;
                    dst_community_id = dst_community_id_tmp;
                }
            } else {
                if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp > dst_community_id)) {
                    best_modularity = best_modularity_tmp;
                    dst_community_id = dst_community_id_tmp;
                }
            }
        }

        // write tile32-wide reduce result into shared memory
        if (tile32.thread_rank() == 0) {
            buffer_int[tile32.meta_group_rank()] = dst_community_id;
            buffer_double[tile32.meta_group_rank()] = best_modularity;
        }

        block.sync();

        if (block.thread_rank() < tile32_num) {
            best_modularity = buffer_double[block.thread_rank()];
            dst_community_id = buffer_int[block.thread_rank()];
        }

        block.sync();

        if (block.thread_rank() < 32) {
            for (e = 32 / 2; e > 0; e /= 2) {
                weight_t best_modularity_tmp = tile32.shfl_down(best_modularity, e);    // available only for sizes lower or equal to 32
                vertex_t dst_community_id_tmp = tile32.shfl_down(dst_community_id, e);
                // Add a constraint: when the best modularity is the same, choose the one with the smallest ID.
                if (up_down) {
                    if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp < dst_community_id)) {
                        best_modularity = best_modularity_tmp;
                        dst_community_id = dst_community_id_tmp;
                    }
                } else {
                    if (best_modularity_tmp > best_modularity || (best_modularity_tmp == best_modularity && dst_community_id_tmp > dst_community_id)) {
                        best_modularity = best_modularity_tmp;
                        dst_community_id = dst_community_id_tmp;
                    }
                }
            }
        }

        block.sync();

        if (block.thread_rank() == 0) {
            shared_device_community_ids_new_[vertex_id] = dst_community_id;
        }

        block.sync();
    }
}

void calculate_eicj_and_move_vertex_bin(
    vertex_t local_vertices,
    vertex_t begin_vertex_id,
    vertex_t* private_device_offset,
    edge_t* private_device_edge,
    weight_t* private_device_edge_weight,
    weight_t* private_device_vertex_weight,
    vertex_t* private_device_part_vertex_offset,
    vertex_t* shared_device_community_ids_new,
    vertex_t* shared_device_community_ids,
    weight_t* shared_device_community_weight,
    weight_t mass,
    int my_pe,
    int n_pes,
    bool up_down,
    BIN *bins,
    cudaStream_t *streams,
    vertex_t total_vertices,
    cudaStream_t default_stream
){
    int grid_num;
    int block_num;
    size_t d_shared_mem;

    vertex_t * hash_table_key = nullptr;
    weight_t * hash_table_value = nullptr;

    for (int i = BIN_NUM - 1; i >= 0; i--) {
        if (bins->bin_size[i] > 0) {
            switch (i) {
                case 0:
                    // tile2 for a vertex whose #edges is less than 2
                    block_num = 512;
                    grid_num = iDivUp(bins->bin_size[i], block_num / 2);
                    d_shared_mem = sizeof(vertex_t) * block_num + sizeof(weight_t) * block_num;
                    calculate_eicj_and_move_vertex_sh_tile<2><<<grid_num, block_num, d_shared_mem, streams[0]>>>(begin_vertex_id,
                                                                                                                    private_device_offset,
                                                                                                                    private_device_edge,
                                                                                                                    private_device_edge_weight,
                                                                                                                    private_device_vertex_weight,
                                                                                                                    private_device_part_vertex_offset,
                                                                                                                    shared_device_community_ids_new,
                                                                                                                    shared_device_community_ids,
                                                                                                                    shared_device_community_weight,
                                                                                                                    mass,
                                                                                                                    my_pe,
                                                                                                                    n_pes,
                                                                                                                    up_down,
                                                                                                                    bins->bin_size[i],
                                                                                                                    bins->bin_offset[i],
                                                                                                                    bins->device_bin_permutation);
                    break;
                case 1:
                    // tile4 for a vertex whose #edges is less than 4
                    block_num = 512;
                    grid_num = iDivUp(bins->bin_size[i], block_num / 4);
                    d_shared_mem = sizeof(vertex_t) * block_num + sizeof(weight_t) * block_num;
                    calculate_eicj_and_move_vertex_sh_tile<4><<<grid_num, block_num, d_shared_mem, streams[1]>>>(begin_vertex_id,
                                                                                                                     private_device_offset,
                                                                                                                     private_device_edge,
                                                                                                                     private_device_edge_weight,
                                                                                                                     private_device_vertex_weight,
                                                                                                                     private_device_part_vertex_offset,
                                                                                                                     shared_device_community_ids_new,
                                                                                                                     shared_device_community_ids,
                                                                                                                     shared_device_community_weight,
                                                                                                                     mass,
                                                                                                                     my_pe,
                                                                                                                     n_pes,
                                                                                                                     up_down,
                                                                                                                     bins->bin_size[i],
                                                                                                                     bins->bin_offset[i],
                                                                                                                     bins->device_bin_permutation);
                    break;
                case 2:
                    // tile8 for a vertex whose #edges is less than 8
                    block_num = 512;
                    grid_num = iDivUp(bins->bin_size[i], block_num / 8);
                    d_shared_mem = sizeof(vertex_t) * block_num + sizeof(weight_t) * block_num;
                    calculate_eicj_and_move_vertex_sh_tile<8><<<grid_num, block_num, d_shared_mem, streams[2]>>>(begin_vertex_id,
                                                                                                                     private_device_offset,
                                                                                                                     private_device_edge,
                                                                                                                     private_device_edge_weight,
                                                                                                                     private_device_vertex_weight,
                                                                                                                     private_device_part_vertex_offset,
                                                                                                                     shared_device_community_ids_new,
                                                                                                                     shared_device_community_ids,
                                                                                                                     shared_device_community_weight,
                                                                                                                     mass,
                                                                                                                     my_pe,
                                                                                                                     n_pes,
                                                                                                                     up_down,
                                                                                                                     bins->bin_size[i],
                                                                                                                     bins->bin_offset[i],
                                                                                                                     bins->device_bin_permutation);
                    break;
                case 3:
                    // tile16 for a vertex whose #edges is less than 16
                    block_num = 512;
                    grid_num = iDivUp(bins->bin_size[i], block_num / 16);
                    d_shared_mem = sizeof(vertex_t) * block_num + sizeof(weight_t) * block_num;
                    calculate_eicj_and_move_vertex_sh_tile<16><<<grid_num, block_num, d_shared_mem, streams[3]>>>(begin_vertex_id,
                                                                                                                     private_device_offset,
                                                                                                                     private_device_edge,
                                                                                                                     private_device_edge_weight,
                                                                                                                     private_device_vertex_weight,
                                                                                                                     private_device_part_vertex_offset,
                                                                                                                     shared_device_community_ids_new,
                                                                                                                     shared_device_community_ids,
                                                                                                                     shared_device_community_weight,
                                                                                                                     mass,
                                                                                                                     my_pe,
                                                                                                                     n_pes,
                                                                                                                     up_down,
                                                                                                                     bins->bin_size[i],
                                                                                                                     bins->bin_offset[i],
                                                                                                                     bins->device_bin_permutation);
                    break;

                case 4:
                    // tile32 for a vertex whose #edges is less than 32
                    block_num = 512;
                    grid_num = iDivUp(bins->bin_size[i], block_num / 32);
                    d_shared_mem = sizeof(vertex_t) * block_num + sizeof(weight_t) * block_num;
                    calculate_eicj_and_move_vertex_sh_tile<32><<<grid_num, block_num, d_shared_mem, streams[4]>>>(begin_vertex_id,
                                                                                                                      private_device_offset,
                                                                                                                      private_device_edge,
                                                                                                                      private_device_edge_weight,
                                                                                                                      private_device_vertex_weight,
                                                                                                                      private_device_part_vertex_offset,
                                                                                                                      shared_device_community_ids_new,
                                                                                                                      shared_device_community_ids,
                                                                                                                      shared_device_community_weight,
                                                                                                                      mass,
                                                                                                                      my_pe,
                                                                                                                      n_pes,
                                                                                                                      up_down,
                                                                                                                      bins->bin_size[i],
                                                                                                                      bins->bin_offset[i],
                                                                                                                      bins->device_bin_permutation);
                    break;
                case 5:
                    block_num = 128;
                    grid_num = bins->bin_size[i];
                    calculate_eicj_and_move_vertex_sh_bk<128><<<grid_num, block_num, 0, streams[5]>>>(begin_vertex_id,
                                                                                                          private_device_offset,
                                                                                                          private_device_edge,
                                                                                                          private_device_edge_weight,
                                                                                                          private_device_vertex_weight,
                                                                                                          private_device_part_vertex_offset,
                                                                                                          shared_device_community_ids_new,
                                                                                                          shared_device_community_ids,
                                                                                                          shared_device_community_weight,
                                                                                                          mass,
                                                                                                          my_pe,
                                                                                                          n_pes,
                                                                                                          up_down,
                                                                                                          bins->bin_size[i],
                                                                                                          bins->bin_offset[i],
                                                                                                          bins->device_bin_permutation,
                                                                                                          total_vertices);
                    break;
                case 6:
                    block_num = 512;
                    grid_num = bins->bin_size[i];
                    calculate_eicj_and_move_vertex_sh_bk<512><<<grid_num, block_num, 0, streams[6]>>>(begin_vertex_id,
                                                                                                          private_device_offset,
                                                                                                          private_device_edge,
                                                                                                          private_device_edge_weight,
                                                                                                          private_device_vertex_weight,
                                                                                                          private_device_part_vertex_offset,
                                                                                                          shared_device_community_ids_new,
                                                                                                          shared_device_community_ids,
                                                                                                          shared_device_community_weight,
                                                                                                          mass,
                                                                                                          my_pe,
                                                                                                          n_pes,
                                                                                                          up_down,
                                                                                                          bins->bin_size[i],
                                                                                                          bins->bin_offset[i],
                                                                                                          bins->device_bin_permutation,
                                                                                                          total_vertices);
                    break;
                case 7:
                    block_num = 1024;
                    grid_num = bins->bin_size[i];
                    calculate_eicj_and_move_vertex_sh_bk<2048><<<grid_num, block_num, 0, streams[7]>>>(begin_vertex_id,
                                                                                                           private_device_offset,
                                                                                                           private_device_edge,
                                                                                                           private_device_edge_weight,
                                                                                                           private_device_vertex_weight,
                                                                                                           private_device_part_vertex_offset,
                                                                                                           shared_device_community_ids_new,
                                                                                                           shared_device_community_ids,
                                                                                                           shared_device_community_weight,
                                                                                                           mass,
                                                                                                           my_pe,
                                                                                                           n_pes,
                                                                                                           up_down,
                                                                                                           bins->bin_size[i],
                                                                                                           bins->bin_offset[i],
                                                                                                           bins->device_bin_permutation,
                                                                                                           total_vertices);
                    break;
                case 8:
                    block_num = 1024;
                    grid_num = bins->bin_size[i];
                    calculate_eicj_and_move_vertex_sh_bk<4094><<<grid_num, block_num, 0, streams[8]>>>(begin_vertex_id,
                                                                                                           private_device_offset,
                                                                                                           private_device_edge,
                                                                                                           private_device_edge_weight,
                                                                                                           private_device_vertex_weight,
                                                                                                           private_device_part_vertex_offset,
                                                                                                           shared_device_community_ids_new,
                                                                                                           shared_device_community_ids,
                                                                                                           shared_device_community_weight,
                                                                                                           mass,
                                                                                                           my_pe,
                                                                                                           n_pes,
                                                                                                           up_down,
                                                                                                           bins->bin_size[i],
                                                                                                           bins->bin_offset[i],
                                                                                                           bins->device_bin_permutation,
                                                                                                           total_vertices);
                    break;
                case 9:
                    int SMs = 80;
                    int hash_table_num = min(SMs, bins->bin_size[i]);
                    vertex_t max_degree = bins->max_degree * 2;
                    vertex_t len_hash_table = hash_table_num * max_degree;
                    CUDA_RT_CALL(cudaMalloc((void **) &hash_table_key, sizeof(vertex_t) * len_hash_table));
                    CUDA_RT_CALL(cudaMalloc((void **) &hash_table_value, sizeof(weight_t) * len_hash_table));
                    grid_num = hash_table_num;
                    block_num = 1024;
                    calculate_eicj_and_move_vertex_gl_bk<<<grid_num, block_num, 0, streams[9]>>>(begin_vertex_id,
                                                                                                     private_device_offset,
                                                                                                     private_device_edge,
                                                                                                     private_device_edge_weight,
                                                                                                     private_device_vertex_weight,
                                                                                                     private_device_part_vertex_offset,
                                                                                                     shared_device_community_ids_new,
                                                                                                     shared_device_community_ids,
                                                                                                     shared_device_community_weight,
                                                                                                     hash_table_key,
                                                                                                     hash_table_value,
                                                                                                     mass,
                                                                                                     my_pe,
                                                                                                     n_pes,
                                                                                                     up_down,
                                                                                                     bins->bin_size[i],
                                                                                                     bins->bin_offset[i],
                                                                                                     bins->device_bin_permutation,
                                                                                                     max_degree);
                    break;
            }
        }
    }

    for (int i = 0; i < BIN_NUM; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    CUDA_RT_CALL(cudaFree(hash_table_key));
    CUDA_RT_CALL(cudaFree(hash_table_value));
}


__global__ void __launch_bounds__(1024, 1)
compute_community_weight_local_atomic(
    vertex_t local_vertices,
    vertex_t total_vertices,
    weight_t* private_device_vertex_weight,
    vertex_t* shared_device_community_ids,
    vertex_t* shared_device_community_ids_new_,
    weight_t* shared_device_community_weight,
    weight_t* shared_device_community_delta_weight,
    vertex_t* private_device_part_vertex_offset,
    int my_pe,
    int n_pes
)
{
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    vertex_t vertex_id;
    vertex_t src_community_id;
    vertex_t dst_community_id;
    weight_t vertex_weight;

    // init shared_device_community_delta_weight
    for (vertex_id = grid.thread_rank(); vertex_id < total_vertices; vertex_id += grid.num_threads()) {
        shared_device_community_delta_weight[vertex_id] = 0.;
    }

    grid.sync();

    // update local shared_device_community_delta_weight
    for (vertex_id = grid.thread_rank(); vertex_id < local_vertices; vertex_id += grid.num_threads()) {
        src_community_id = shared_device_community_ids[vertex_id];
        dst_community_id = shared_device_community_ids_new_[vertex_id];
        if (src_community_id != dst_community_id) {
            vertex_weight = private_device_vertex_weight[vertex_id];
            atomicAdd(shared_device_community_delta_weight + src_community_id, -vertex_weight);
            atomicAdd(shared_device_community_delta_weight + dst_community_id, vertex_weight);
        }
    }
}


__global__ void __launch_bounds__(1024, 1)
compute_community_weight(
    vertex_t local_vertices,
    vertex_t total_vertices,
    weight_t* private_device_vertex_weight,
    vertex_t* shared_device_community_ids,
    vertex_t* shared_device_community_ids_new_,
    weight_t* shared_device_community_weight,
    weight_t* shared_device_community_delta_weight,
    vertex_t* private_device_part_vertex_offset,
    int my_pe,
    int n_pes
)
{
    int i;
    int j;
    int nelems;
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);
    int tile_num_grid = grid.num_threads() / 32;      /* the total number of tile in the whole grid */
    int tile_id_grid = grid.thread_rank() / 32;      /* the global id of the tile in the grid */
    int shared_offset = (block.thread_rank() / 32) * 32;
    vertex_t start_vertex_id = private_device_part_vertex_offset[my_pe];

    __shared__ double buff[1024];

    int local_tile;
    int remote_tile;
    if (n_pes == 1) {
        local_tile = tile_num_grid;
        remote_tile = 0;
    } else {
        local_tile = tile_num_grid / n_pes;
        remote_tile = tile_num_grid - local_tile;
    }

    // local workload
    if (tile_id_grid < local_tile) {
        for (j = tile_id_grid * tile32.num_threads() + tile32.thread_rank(); j < local_vertices; j += (local_tile * tile32.num_threads())) {
            atomicAdd(shared_device_community_weight + j, shared_device_community_delta_weight[start_vertex_id + j]);
        }
    }

    // remote workload
    else {
        for (vertex_t tile_offset = (tile_id_grid - local_tile) * tile32.num_threads(); tile_offset < local_vertices; tile_offset += (remote_tile * tile32.num_threads())) {
            nelems = min(tile32.num_threads(), local_vertices - tile_offset);
            for (i = 1; i < n_pes; i++) {
                nvshmemx_double_get_warp(buff + shared_offset, shared_device_community_delta_weight + start_vertex_id + tile_offset, nelems, (my_pe + i) % n_pes);
                for (j = tile32.thread_rank(); j < nelems; j += tile32.num_threads()) {
                    atomicAdd(shared_device_community_weight + tile_offset + j, buff[shared_offset + j]);
                }
            }
        }
    }
}

}


void louvain_gl::run(HostGraph *hostGraph, GpuGraph *gpuGraph, const double threshold, const int max_iter,
                           const int max_phases) {
    int n_pes = nvshmem_n_pes();
    int my_pe = nvshmem_my_pe();

    // stream create
    cudaStream_t default_stream;
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&default_stream, cudaStreamDefault));

    cudaStream_t streams[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    nvshmem_barrier_all();

    auto mass = gpuGraph->get_mass_();
    auto local_vertices = gpuGraph->get_local_vertices_();
    auto local_edges = gpuGraph->get_local_edges_();
    auto total_vertices = gpuGraph->get_total_vertices_();
    auto *part_vertex_offset = gpuGraph->get_part_vertex_offset_();
    weight_t Q_host = 0;
    weight_t Q_old_host = -1;

    // private memory
    auto *private_device_offset = gpuGraph->get_private_device_offset_();
    auto *private_device_edge = gpuGraph->get_private_device_edge_();
    auto *private_device_edge_weight = gpuGraph->get_private_device_edge_weight_();
    auto *private_device_part_vertex_offset = gpuGraph->get_private_device_part_vertex_offset_();
    auto *private_device_vertex_weight = gpuGraph->get_private_device_vertex_weight_();

    // init bins
    BIN* bins = new BIN(BIN_NUM, local_vertices);

    // nvshmem shared memory
    auto *shared_device_community_weight = gpuGraph->get_shared_device_community_weight_();
    auto *shared_device_community_delta_weight = gpuGraph->get_shared_device_community_delta_weight_();
    auto *shared_device_community_ids = gpuGraph->get_shared_device_community_ids_();
    auto *shared_device_community_ids_new = gpuGraph->get_shared_device_community_ids_new_();
    auto *Q = (weight_t *) nvshmem_malloc(sizeof(weight_t));
    CUDA_RT_CALL(cudaMemset(Q, 0, sizeof(weight_t)));
    auto *shared_sum_community_weight = (weight_t *)nvshmem_malloc(sizeof(weight_t));
    CUDA_RT_CALL(cudaMemset(shared_sum_community_weight, 0, sizeof(weight_t)));
    auto *shared_sum_internal_edge_weight = (weight_t *)nvshmem_malloc(sizeof(weight_t));

    int block_dims = 1024;
    int grid_size = 0;
    size_t d_shared_mem = 0;

    CUDA_RT_CALL(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    double start;
    double stop;
    double start_total;
    double stop_total;
    double start_phase;
    double stop_phase;
    double start_in_loop;
    double stop_in_loop;

    double coarsen_graph_total_time = 0.;
    double main_loop_total_time = 0.;
    double symbolic_time = 0.;
    double numeric_time = 0.;
    double update_community_time = 0.;
    double update_weight_time = 0.;
    double compute_modularity_time = 0.;

    int phase_num = 0;
    int loop_total = 0;
    start_total = MPI_Wtime();
    while (phase_num < max_phases && (Q_host - Q_old_host) > threshold) {

        Q_old_host = Q_host;
        weight_t new_Q;
        weight_t cur_Q;
        vertex_t begin_vertex_id = part_vertex_offset[my_pe];
        vertex_t end_vertex_id = part_vertex_offset[my_pe + 1];

        int loop_num = 0;
        double loop_time = 0.;

        start_phase = MPI_Wtime();
        if (my_pe == 0) {
            printf("----------------------------------------\n");
            printf("Phase %d\n", phase_num);
        }

        // 1. Init community id for new graph

        init_community_id<<<80, 1024, 0, default_stream>>>(shared_device_community_ids, begin_vertex_id, end_vertex_id, local_vertices);
        CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

        // 2. Compute the vertices and communities weights

        reduce_vertices_weights<<<80, 1024, 0, default_stream>>>(local_vertices,
                                                                 private_device_offset,
                                                                 private_device_edge_weight,
                                                                 private_device_vertex_weight);
        CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

        copy<weight_t><<<80, 1024, 0, default_stream>>>(private_device_vertex_weight, shared_device_community_weight, local_vertices);
        CUDA_RT_CALL(cudaStreamSynchronize(default_stream));


        // 3. Compute the modularity of this phase
        // use NVSHMEM collective operation, need 'nvshmemx_collective_launch' to launch kernel
        void *kernel_args[] = {
                (void *) &mass,
                (void *) &local_vertices,
                (void *) &private_device_offset,
                (void *) &private_device_edge,
                (void *) &private_device_edge_weight,
                (void *) &private_device_part_vertex_offset,
                (void *) &shared_device_community_ids,
                (void *) &shared_device_community_weight,
                (void *) &shared_sum_community_weight,
                (void *) &shared_sum_internal_edge_weight,
                (void *) &my_pe,
                (void *) &n_pes,
                (void *) &Q
        };
        NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_modularity, block_dims, kernel_args, d_shared_mem, &grid_size));
        nvshmem_barrier_all();

        NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_modularity, grid_size, block_dims, kernel_args, d_shared_mem, default_stream));
        nvshmemx_barrier_all_on_stream(default_stream);
        CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

        CUDA_RT_CALL(cudaMemcpy(&new_Q, Q, sizeof(weight_t) , cudaMemcpyDeviceToHost));

        cur_Q = new_Q - 1;

        if(my_pe == 0) {
            printf("| %-10s | %-10s | %-10s | %-10s | %-10s |\n", "Loop", "Q", "dQ", "time(s)", "time(ms)");
            printf("|------------|------------|------------|------------|------------|\n");
            printf("| %-10d | %-10f | %-10f | %-10f | %-10f |\n", 0, new_Q, 0., 0., 0.);
        }

        bool up_down = true;

        bins->bin_create(private_device_offset, local_vertices);


        // 4. Update the community id of each vertex (main loop)
        // The presence of negative modularity leads to a direct termination of the cycle
        while ((new_Q - cur_Q) > threshold) {

            cur_Q = new_Q;

            start = MPI_Wtime();
            start_in_loop = MPI_Wtime();

            // a) update community id
            calculate_eicj_and_move_vertex_bin(local_vertices,
                                               begin_vertex_id,
                                               private_device_offset,
                                               private_device_edge,
                                               private_device_edge_weight,
                                               private_device_vertex_weight,
                                               private_device_part_vertex_offset,
                                               shared_device_community_ids_new,
                                               shared_device_community_ids,
                                               shared_device_community_weight,
                                               mass,
                                               my_pe,
                                               n_pes,
                                               up_down,
                                               bins,
                                               streams,
                                               total_vertices,
                                               default_stream);



            up_down = !up_down;

            nvshmem_barrier_all();
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

            stop_in_loop = MPI_Wtime();
            update_community_time += (stop_in_loop - start_in_loop);
            start_in_loop = MPI_Wtime();

            // b) update community weight
            void *kernel_args_ccw[] = {
                    (void *) &local_vertices,
                    (void *) &total_vertices,
                    (void *) &private_device_vertex_weight,
                    (void *) &shared_device_community_ids,
                    (void *) &shared_device_community_ids_new,
                    (void *) &shared_device_community_weight,
                    (void *) &shared_device_community_delta_weight,
                    (void *) &private_device_part_vertex_offset,
                    (void *) &my_pe,
                    (void *) &n_pes
            };
            NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_community_weight_local_atomic, block_dims, kernel_args_ccw, d_shared_mem, &grid_size));
            nvshmem_barrier_all();
            NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_community_weight_local_atomic, grid_size, block_dims, kernel_args_ccw, d_shared_mem, default_stream));
            nvshmemx_barrier_all_on_stream(default_stream);
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

            NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_community_weight, block_dims, kernel_args_ccw, d_shared_mem, &grid_size));
            nvshmem_barrier_all();
            NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_community_weight, grid_size, block_dims, kernel_args_ccw, d_shared_mem, default_stream));
            nvshmemx_barrier_all_on_stream(default_stream);
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

            stop_in_loop = MPI_Wtime();
            update_weight_time += (stop_in_loop - start_in_loop);
            start_in_loop = MPI_Wtime();

            // c) compute modularity
            void *kernel_args_cm[] = {
                    (void *) &mass,
                    (void *) &local_vertices,
                    (void *) &private_device_offset,
                    (void *) &private_device_edge,
                    (void *) &private_device_edge_weight,
                    (void *) &private_device_part_vertex_offset,
                    (void *) &shared_device_community_ids_new,
                    (void *) &shared_device_community_weight,
                    (void *) &shared_sum_community_weight,
                    (void *) &shared_sum_internal_edge_weight,
                    (void *) &my_pe,
                    (void *) &n_pes,
                    (void *) &Q
            };
            NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((void *)compute_modularity, block_dims, kernel_args_cm, d_shared_mem, &grid_size));
            nvshmem_barrier_all();
            NVSHMEM_CHECK(nvshmemx_collective_launch((void *)compute_modularity, grid_size, block_dims, kernel_args_cm, d_shared_mem, default_stream));
            nvshmemx_barrier_all_on_stream(default_stream);
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

            stop_in_loop = MPI_Wtime();
            compute_modularity_time += (stop_in_loop - start_in_loop);

            CUDA_RT_CALL(cudaMemcpy(&new_Q, Q, sizeof(weight_t) , cudaMemcpyDeviceToHost));

            if ((new_Q - cur_Q) > threshold) {
                copy<vertex_t><<<128, 1024, 0, default_stream>>>(shared_device_community_ids_new,
                                                                 shared_device_community_ids,
                                                                 local_vertices);
                CUDA_RT_CALL(cudaStreamSynchronize(default_stream));
            } else {
                new_Q = cur_Q;
            }


            loop_num++;
            stop = MPI_Wtime();
            if(my_pe == 0){
                printf("| %-10d | %-10f | %-10f | %-10f | %-10f |\n", loop_num, new_Q, (new_Q - cur_Q), (stop - start), (stop - start) * 1000);
            }

            loop_time += (stop - start) * 1000;

            nvshmem_barrier_all();
            CUDA_RT_CALL(cudaStreamSynchronize(default_stream));

        }

        nvshmem_barrier_all();
        CUDA_RT_CALL(cudaDeviceSynchronize());

        loop_time /= loop_num;
        loop_total += loop_num;
        stop_phase = MPI_Wtime();
        main_loop_total_time += (stop_phase - start_phase);
        if (my_pe == 0) {
            printf("Main loop total execution time: %f s, %f ms\n", (stop_phase - start_phase), double((stop_phase - start_phase) * 1000));
            printf("The average execution time per loop: %f s, %f ms\n", loop_time / 1000, loop_time);
        }

        Q_host = new_Q;
        if ((Q_host - Q_old_host) <= threshold) break;

        start_phase = MPI_Wtime();

        coarsen_graph_mg::coarsen_graph(hostGraph, gpuGraph, my_pe, n_pes, default_stream, streams, symbolic_time, numeric_time);

        stop_phase = MPI_Wtime();
        coarsen_graph_total_time += (stop_phase - start_phase);
        phase_num++;
        if (my_pe == 0) {
            printf("Coarsen graph execution time: %f s, %f ms\n", (stop_phase - start_phase), double((stop_phase - start_phase) * 1000));
        }

        local_vertices = gpuGraph->get_local_vertices_();
        total_vertices = gpuGraph->get_total_vertices_();

        nvshmem_barrier_all();
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }

    nvshmem_barrier_all();
    CUDA_RT_CALL(cudaDeviceSynchronize());

    stop_total = MPI_Wtime();
    if (my_pe == 0) {
        printf("----------------------------------------\n");
        printf("Total time for clustering   : %f s\n", main_loop_total_time);
        printf("Total time for coarsen graph: %f s\n", coarsen_graph_total_time);
        printf("TOTAL TIME                  : %f s\n", (stop_total - start_total));
//        printf("Total time for updating community: %f s\n", update_community_time);
//        printf("Total time for updating weight: %f s\n", update_weight_time);
//        printf("Total time for computing modularity: %f s\n", compute_modularity_time);
//        printf("Total time for symb. phase: %f s\n", symbolic_time);
//        printf("Total time for num. phase: %f s\n", numeric_time);
//        printf("Total phases: %d\n", phase_num++);
//        printf("Total loops: %d\n", loop_total);
    }

    nvshmem_free(Q);
    nvshmem_free(shared_sum_community_weight);
    nvshmem_free(shared_sum_internal_edge_weight);
}

