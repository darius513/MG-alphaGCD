cmake_minimum_required(VERSION 3.23)
project(MG_GCD)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)
include_directories(include/mmio)
include_directories(/root/local/include)

set(CMAKE_CUDA_ARCHITECTURES "70")
find_package(CUDAToolkit ${NVCC_VER_MIN} REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

enable_language(CUDA)

add_definitions(-DTIMING)

#set(CSR_SPGEMM_FLAGS "-Xcompiler -fopenmp")
#set_source_files_properties(source/csr_spgemm.cu PROPERTIES COMPILE_FLAGS ${CSR_SPGEMM_FLAGS})

add_executable(MG_GCD
        include/mmio/mmio.h
        include/mmio/mmio_wrapper.h
        include/common.h
        include/graph/host_graph.h
        include/graph/gpu_graph.cuh
        include/partition/edge_partition.cuh
        include/bin/BIN.cuh
        include/louvain/louvain_baseline.cuh
        include/louvain/louvain_gl.cuh
        include/louvain/louvain.cuh
        include/coarsen_graph/coarsen_graph_mg.cuh
        src/mmio/mmio.c
        src/mmio/mmio_wrapper.cu
        src/common.cu
        src/graph/host_graph.cpp
        src/main.cu
        src/graph/gpu_graph.cu
        src/partition/edge_partition.cu
        src/bin/BIN.cu
        src/louvain/louvain_baseline.cu
        src/louvain/louvain_gl.cu
        src/louvain/louvain.cu
        src/coarsen_graph/coarsen_graph_mg.cu)

find_package(OpenMP REQUIRED)
find_package(NVSHMEM REQUIRED)
find_package(MPI REQUIRED)

list(APPEND CMAKE_CUDA_FLAGS "-O3 -Xptxas -O3 -Xcompiler -O3 -Xcompiler -fopenmp")


target_link_libraries(MG_GCD
        CUDA::cudart
        OpenMP::OpenMP_CXX
        nvshmem::nvshmem
        MPI::MPI_CXX
        )

target_compile_options(MG_GCD PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -D_FORCE_INLINES
        >)