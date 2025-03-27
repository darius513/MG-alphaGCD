# MG-αGCD

**MG-αGCD** is an open source library that optimizing Louvain graph community detection method for efficiency and scalability on Multi-GPU platforms.

## Paper information 

Shuai Yang and Changyou Zhang. 2025. MG-𝛼GCD: Accelerating Graph Community Detection on Multi-GPU Platforms. In 2025 International Conference on Supercomputing (ICS ’25), June 8–11, 2025, Salt Lake City, UT, USA. ACM, New York, NY, USA, 14 pages. https://doi.org/10.1145/3721145.3725753

## Dependencies

1. Install CUDA 11.6 or newer

2. Install NVSHMEM 2.7.0 or newer (Spack install sample)

   - Install spack

     Please check this [branch](https://github.com/spack/spack/pull/36363) , which provides nvshmem 2.9.0 version.
     
   - Requirements for nvshmem installation

     - Load spack

       ```shell
       source ~/spack/share/spack/setup-env.sh
       ```

     - Add gcc and gfortran compilers to spack

       ```shell
       spack compiler find
       sudo apt-get install gfortran
       ```

       Modify /root/.spack/linux/compilers.yaml, Specifies the gfortran path

       ```yaml
       f77: /usr/bin/gfortran
       fc: /usr/bin/gfortran
       ```

       ```shell
       spack compilers
       ```

     - install libibverbs-dev

       ```
       sudo apt install libibverbs-dev
       ```

   - Spack install nvshmem

     ```shell
     spack install nvshmem +gpu_initiated_support +cuda +gdrcopy +ucx +mpi build_system=cmake ^openmpi +cuda fabrics=ucx ^ucx +cuda +gdrcopy +dm +thread_multiple ^pmix@4.2.2
     ```

   - NVSHMEM environment load

     ```shell
     spack load nvshmem ^pmix@4.2.2
     ```

3. Install OpenMP 5.0 or newer

4. Install CMake 3.15.5 or newer

## Getting Start

Our program only supports input graph files in matrix market format (.mtx). All the graph datasets used in our evaluation can be accessed from the SuiteSparse Matrix Collection.

1. Create the `build` file folder in the root path of the program, and generate `makefile` with:

   ```shell
   cmake ..
   ```

2. Generate the executable file `MG_GCD`:

   ```shell
   make
   ```

3. Run the program with `mpirun`. 

   ```shell
   mpirun -np 4 --allow-run-as-root ./MG_GCD -path_of_graph ../data/wiki-topcats.mtx 
   ```

   **Options:**

   - `-np`: The number of GPUs
   - `-path_of_graph`: The path of the input graph data (only .mtx format is acceptable)
   - `-version`: The different versions of MG-αGCD discussed in the paper.
     - `0`: The irregular sparse remote atomic implementation.
     - `1`: The global bi-probing and pipelined version.
     - `2`: The local bi-probing and pipelined version (default).
