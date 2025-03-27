#ifndef PRIVATE_GPU_GRAPH_CUH
#define PRIVATE_GPU_GRAPH_CUH
#include "../common.h"
#include "./host_graph.h"
class GpuGraph{
private:
    vertex_t total_vertices_;
    edge_t total_edges_;
    vertex_t local_vertices_;
    edge_t local_edges_;
    vertex_t* part_vertex_offset_; /* Vertex partitioning result among GPUs */
    edge_t len_edges_array_; /* The actual length of the `private_device_edge_` array -- `avg_edges + total_vertices_` */
    weight_t mass_;

    // private
    vertex_t* private_device_offset_;
    edge_t* private_device_edge_;
    weight_t* private_device_edge_weight_;
    vertex_t* private_device_part_vertex_offset_;
    weight_t* private_device_vertex_weight_;

    // shared
    weight_t* shared_device_community_weight_;
    weight_t* shared_device_community_delta_weight_;
    vertex_t* shared_device_community_ids_;
    vertex_t* shared_device_community_ids_new_;


public:
    GpuGraph(int n_pes, HostGraph *hostGraph);
    ~GpuGraph(){
        nvshmem_free(private_device_offset_);
        CUDA_RT_CALL(cudaFree(private_device_edge_));
        CUDA_RT_CALL(cudaFree(private_device_edge_weight_));
        CUDA_RT_CALL(cudaFree(private_device_part_vertex_offset_));
        CUDA_RT_CALL(cudaFree(private_device_vertex_weight_));

        nvshmem_free(shared_device_community_weight_);
        nvshmem_free(shared_device_community_delta_weight_);
        nvshmem_free(shared_device_community_ids_);
        nvshmem_free(shared_device_community_ids_new_);
    }

    // get array point
    vertex_t* get_part_vertex_offset_();
    vertex_t* get_private_device_part_vertex_offset_();

    vertex_t* get_private_device_offset_();
    edge_t* get_private_device_edge_();
    weight_t* get_private_device_edge_weight_();

    edge_t get_len_edges_array_();
    weight_t* get_private_device_vertex_weight_();

    vertex_t* get_shared_device_community_ids_();
    vertex_t* get_shared_device_community_ids_new_();
    weight_t* get_shared_device_community_weight_();
    weight_t* get_shared_device_community_delta_weight_();

    vertex_t get_total_vertices_();
    edge_t get_total_edges_();
    vertex_t get_local_vertices_();
    edge_t get_local_edges_();
    weight_t get_mass_();

    void set_local_vertices_(vertex_t local_vertices);
    void set_local_edges_(edge_t local_edges);
    void set_total_vertices_(vertex_t total_vertices_);
    void set_total_edges_(vertex_t total_edges_);
    void set_len_edges_array_(edge_t len_edges_array);
    void set_private_device_edge_(edge_t* private_device_edge);
    void set_private_device_edge_weight_(weight_t* private_device_edge_weight);
};

#endif //PRIVATE_GPU_GRAPH_CUH
