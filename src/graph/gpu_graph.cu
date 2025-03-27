#include "../../include/graph/gpu_graph.cuh"

GpuGraph::GpuGraph(int n_pes, HostGraph *hostGraph):
total_vertices_(hostGraph->get_total_vertices_()), total_edges_(hostGraph->get_total_edge_()),
private_device_edge_(nullptr), private_device_edge_weight_(nullptr),
mass_(hostGraph->get_mass_())
{
    part_vertex_offset_ = new vertex_t[n_pes + 1];
    CUDA_RT_CALL(cudaMalloc((void **) &private_device_part_vertex_offset_, sizeof(vertex_t) * (n_pes + 1)));
    private_device_offset_ = (vertex_t *) nvshmem_malloc ((total_vertices_ + 1) * sizeof(vertex_t));
    CUDA_RT_CALL(cudaMalloc((void **) &private_device_vertex_weight_, sizeof(weight_t) * total_vertices_));
    shared_device_community_ids_new_ = (vertex_t *) nvshmem_malloc (total_vertices_ * sizeof(vertex_t));
    shared_device_community_ids_ = (vertex_t *) nvshmem_malloc (total_vertices_ * sizeof(vertex_t));
    shared_device_community_weight_ = (weight_t *) nvshmem_malloc (total_vertices_ * sizeof(weight_t));
    shared_device_community_delta_weight_ = (weight_t *) nvshmem_malloc (total_vertices_ * sizeof(weight_t));
    CUDA_RT_CALL(cudaMemset(shared_device_community_weight_, 0., total_vertices_ * sizeof(weight_t)));
    CUDA_RT_CALL(cudaMemset(shared_device_community_delta_weight_, 0., total_vertices_ * sizeof(weight_t)));
}

vertex_t *GpuGraph::get_private_device_offset_() {
    return private_device_offset_;
}

vertex_t *GpuGraph::get_private_device_edge_() {
    return private_device_edge_;
}

weight_t *GpuGraph::get_private_device_edge_weight_() {
    return private_device_edge_weight_;
}

vertex_t *GpuGraph::get_shared_device_community_ids_() {
    return shared_device_community_ids_;
}

vertex_t GpuGraph::get_total_vertices_() {
    return total_vertices_;
}

edge_t GpuGraph::get_total_edges_() {
    return total_edges_;
}

vertex_t GpuGraph::get_local_vertices_() {
    return local_vertices_;
}

edge_t GpuGraph::get_local_edges_() {
    return local_edges_;
}

vertex_t *GpuGraph::get_part_vertex_offset_() {
    return part_vertex_offset_;
}

vertex_t *GpuGraph::get_private_device_part_vertex_offset_() {
    return private_device_part_vertex_offset_;
}

weight_t *GpuGraph::get_shared_device_community_weight_() {
    return shared_device_community_weight_;
}

weight_t *GpuGraph::get_shared_device_community_delta_weight_() {
    return shared_device_community_delta_weight_;
}

weight_t *GpuGraph::get_private_device_vertex_weight_() {
    return private_device_vertex_weight_;
}

vertex_t *GpuGraph::get_shared_device_community_ids_new_() {
    return shared_device_community_ids_new_;
}

weight_t GpuGraph::get_mass_() {
    return mass_;
}

edge_t GpuGraph::get_len_edges_array_() {
    return len_edges_array_;
}

void GpuGraph::set_local_vertices_(vertex_t local_vertices) {
    local_vertices_ = local_vertices;
}

void GpuGraph::set_local_edges_(vertex_t local_edges) {
    assert(len_edges_array_ >= local_edges && "Length of edges array less than local_edges\n");
    local_edges_ = local_edges;
}

void GpuGraph::set_total_vertices_(vertex_t total_vertices) {
    total_vertices_ = total_vertices;
}

void GpuGraph::set_total_edges_(vertex_t total_edges) {
    total_edges_ = total_edges;
}

void GpuGraph::set_len_edges_array_(edge_t len_edges_array) {
    len_edges_array_ = len_edges_array;
}

void GpuGraph::set_private_device_edge_(edge_t *private_device_edge) {
    private_device_edge_ = private_device_edge;
}

void GpuGraph::set_private_device_edge_weight_(weight_t *private_device_edge_weight) {
    private_device_edge_weight_ = private_device_edge_weight;
}

