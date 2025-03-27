#include "../../include/graph/host_graph.h"

HostGraph::HostGraph(char *graph_path, int my_pe):
total_vertices_(0), total_edge_(0), host_offset_(nullptr),
host_edge_(nullptr), host_edge_weight_(nullptr), mass_(0)
{
    load_graph_mtx(graph_path);
    if (my_pe == 0) {
        std::cout << std::setfill('-') << std::setw(3 * 25) << "" << std::setfill(' ') << std::endl;
        std::cout << std::setw(25) << "Input graph" << std::setw(25) << "Num. vertices (n)" << std::setw(25) << "Num. edges (M)" << std::endl;
        std::cout << std::setfill('-') << std::setw(3 * 25) << "" << std::setfill(' ') << std::endl;
        std::cout << std::setw(25) << graph_path << std::setw(25) << total_vertices_ << std::setw(25) << total_edge_ << std::endl;
        std::cout << std::setfill('-') << std::setw(3 * 25) << "" << std::setfill(' ') << std::endl;
    }
}

HostGraph::HostGraph(int random_vertex_num, double sparsity, int my_pe):
total_vertices_(random_vertex_num), total_edge_(0), host_offset_(nullptr),
host_edge_(nullptr), host_edge_weight_(nullptr), mass_(0)
{
    randomly_generate_graph(random_vertex_num, sparsity);
    if (my_pe == 0) {
        std::cout << std::setfill('-') << std::setw(3 * 25) << "" << std::setfill(' ') << std::endl;
        std::cout << std::setw(25) << "Input graph" << std::setw(25) << "Num. vertices (n)" << std::setw(25) << "Num. edges (M)" << std::endl;
        std::cout << std::setfill('-') << std::setw(3 * 25) << "" << std::setfill(' ') << std::endl;
        std::cout << std::setw(25) << "random" << std::setw(25) << total_vertices_ << std::setw(25) << total_edge_ << std::endl;
        std::cout << std::setfill('-') << std::setw(3 * 25) << "" << std::setfill(' ') << std::endl;
    }
}

void HostGraph::load_graph_mtx(char *graph_path) {
    if(loadMMSparseMatrix(graph_path, 'd', true, &total_vertices_, &total_vertices_, &total_edge_,
                                 &host_edge_weight_, &host_offset_, &host_edge_, true)){
        exit(EXIT_FAILURE);
    }
}

void HostGraph::randomly_generate_graph(int random_vertex_num, double sparsity) {
    int k = 0;
    int l = 0;
    weight_t *mat = new weight_t[random_vertex_num * random_vertex_num];
    memset(mat, 0, random_vertex_num * random_vertex_num * sizeof(weight_t));
    for(int i = 0; i < random_vertex_num; i++)
    {
        for(int j=0; j < random_vertex_num; j++)
        {
            size_t x = rand() % 1000000;
            if( x < 1000000.0 * sparsity )
            {
                mat[i * random_vertex_num + j] = x / 1000000.0 + 1.0;
                total_edge_++;
            }
        }
    }

    host_offset_ = new vertex_t[total_vertices_ + 1];
    host_edge_ = new edge_t[total_edge_];
    host_edge_weight_ = new weight_t[total_edge_];

    for(int i = 0; i < random_vertex_num; i++)
    {
        for(int j = 0; j < random_vertex_num; j++)
        {
            if(j == 0)
            {
                host_offset_[l++] = k;
            }
            if(mat[i * random_vertex_num + j] != 0)
            {
                host_edge_[k] = j;
                host_edge_weight_[k] = mat[i * random_vertex_num + j];
                k++;
            }
        }
    }
    host_offset_[l] = total_edge_;
}


vertex_t HostGraph::get_total_vertices_() {
    return total_vertices_;
}

edge_t HostGraph::get_total_edge_() {
    return total_edge_;
}

void HostGraph::set_total_vertices_(vertex_t total_vertices) {
    total_vertices_ = total_vertices;
}

void HostGraph::set_total_edge_(vertex_t total_edge) {
    total_edge_ = total_edge;
}

vertex_t *HostGraph::get_host_offset_() {
    return host_offset_;
}

edge_t *HostGraph::get_host_edge_() {
    return host_edge_;
}

weight_t *HostGraph::get_host_edge_weight_() {
    return host_edge_weight_;
}

double HostGraph::get_mass_() {
    return mass_;
}

void HostGraph::compute_total_edge_weight() {
    mass_ = thrust::reduce(host_edge_weight_, host_edge_weight_ + total_edge_);
//    printf("mass: %f\n", mass_);
}