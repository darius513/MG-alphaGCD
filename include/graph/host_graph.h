#ifndef HOST_GRAPH_CUH
#define HOST_GRAPH_CUH

#include "../common.h"
#include "../mmio/mmio_wrapper.h"
#include <stdlib.h>

class HostGraph
{
private:
     vertex_t total_vertices_;
     edge_t total_edge_;
     weight_t mass_;

     vertex_t *host_offset_;
     edge_t *host_edge_;
     weight_t *host_edge_weight_;


public:
    HostGraph(char *graph_path, int my_pe);
    HostGraph(int random_vertex_num, double sparsity, int my_pe);
    ~HostGraph(){
        free(host_offset_);
        free(host_edge_);
        free(host_edge_weight_);
    }
    vertex_t get_total_vertices_();
    edge_t get_total_edge_();
    void set_total_vertices_(vertex_t total_vertices_);
    void set_total_edge_(vertex_t total_edge_);
    weight_t get_mass_();

    vertex_t* get_host_offset_();
    edge_t* get_host_edge_();
    weight_t* get_host_edge_weight_();

    void load_graph_mtx(char *graph_path);
    void randomly_generate_graph(int random_vertex_num, double sparsity);
    void compute_total_edge_weight();
};

#endif //HOST_GRAPH_CUH
