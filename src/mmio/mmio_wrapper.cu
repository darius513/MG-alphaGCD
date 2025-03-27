
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_set>

#include "../../include/mmio/mmio_wrapper.h"
#include "../../include/common.h"

/* avoid Windows warnings (for example: strcpy, fscanf, etc.) */
#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#endif

/* various __inline__ __device__  function to initialize a T_ELEM */
template <typename T_ELEM>
__inline__ T_ELEM cuGet(int);
template <>
__inline__ float cuGet<float>(int x) {
    return float(x);
}

template <>
__inline__ double cuGet<double>(int x) {
    return double(x);
}

template <typename T_ELEM>
__inline__ T_ELEM cuGet(float);
template <>
__inline__ float cuGet<float>(float x) {
    return float(x);
}

template <>
__inline__ double cuGet<double>(float x) {
    return double(x);
}

template <typename T_ELEM>
__inline__ T_ELEM cuGet(float, float);
template <>
__inline__ float cuGet<float>(float x, float y) {
    return float(x);
}

template <>
__inline__ double cuGet<double>(float x, float y) {
    return double(x);
}

template <typename T_ELEM>
__inline__ T_ELEM cuGet(double);
template <>
__inline__ float cuGet<float>(double x) {
    return float(x);
}

template <>
__inline__ double cuGet<double>(double x) {
    return double(x);
}

template <typename T_ELEM>
__inline__ T_ELEM cuGet(double, double);
template <>
__inline__ float cuGet<float>(double x, double y) {
    return float(x);
}

template <>
__inline__ double cuGet<double>(double x, double y) {
    return double(x);
}

void init_data_from_mtx(char *file_path, vertex_t *&offset, edge_t *&colindex, weight_t *&value, vertex_t *m, vertex_t *n, edge_t *nz)
{
    edge_t i, j;
    edge_t num;
    bool mm_is_symmetric;
    char *line, *ch;
    FILE *fp;
    vertex_t *col_coo, *row_coo;
    vertex_t *row_coo_extend;
    weight_t *val_coo;
    int LINE_LENGTH_MAX = 1000;

    mm_is_symmetric = false;
    line = new char[LINE_LENGTH_MAX];

    fp = fopen(file_path, "r");
    if (fp == NULL) {
        printf("Cannot find file\n");
        exit(1);
    }

    fgets(line, LINE_LENGTH_MAX, fp);
    if (strstr(line, "general")) {
        mm_is_symmetric = false;
    } else if (strstr(line, "symmetric")) {
        mm_is_symmetric = true;
    }

    do {
        fgets(line, LINE_LENGTH_MAX, fp);
    } while (line[0] == '%');

    /* Get size info */
    edge_t nnz = 0;
    vertex_t nrow = 0;
    vertex_t ncol = 0;
    edge_t nnz_symmetric = 0;

    sscanf(line, "%u %u %u", &nrow, &ncol, &nnz);

    *m = nrow;
    *n = ncol;
    num = 0;
    row_coo = new vertex_t[nnz];
    col_coo = new vertex_t[nnz];
    val_coo = new weight_t[nnz];

    if (mm_is_symmetric) {
        while (fgets(line, LINE_LENGTH_MAX, fp)) {
            ch = line;
            /* Read first word (row id)*/
            row_coo[num] = (vertex_t)(atoi(ch) - 1);
            ch = strchr(ch, ' ');
            ch++;
            /* Read second word (column id)*/
            col_coo[num] = (vertex_t)(atoi(ch) - 1);
            ch = strchr(ch, ' ');
//            if (ch != NULL) {
//                ch++;
//                /* Read third word (value data)*/
//                val_coo[num] = (weight_t) atof(ch);
//                ch = strchr(ch, ' ');
//            } else {
//                val_coo[num] = 1.0;
//            }
            // We treat all weights as 1
            val_coo[num] = 1.0;
            num++;
        }
        assert(num == nnz);
        fclose(fp);
        delete[] line;
        for (i = 0; i < nnz; i++) {
            if (row_coo[i] != col_coo[i]) {
                nnz_symmetric++;
            }
        }
        nnz_symmetric += nnz;

        row_coo_extend = new vertex_t[nnz_symmetric];
        colindex = new edge_t[nnz_symmetric];
        value = new weight_t[nnz_symmetric];

        std::copy(row_coo, row_coo + nnz, row_coo_extend);
        std::copy(col_coo, col_coo + nnz, colindex);
        std::copy(val_coo, val_coo + nnz, value);

        for (i = 0, j = nnz; i < nnz; i++) {
            if (row_coo[i] != col_coo[i]) {
                row_coo_extend[j] = col_coo[i];
                colindex[j] = row_coo[i];
                value[j] = val_coo[i];
                j++;
            }
        }

        *nz = nnz_symmetric;
        assert(j == nnz_symmetric);
        delete[] row_coo;
        delete[] col_coo;
        delete[] val_coo;

//        Sort by row index, we treat all weights as 1
//        int *row_coo_extend_copy = new int [nnz_symmetric];
//        std::copy(row_coo_extend, row_coo_extend + nnz_symmetric, row_coo_extend_copy);
        thrust::stable_sort_by_key(thrust::host, row_coo_extend, row_coo_extend + nnz_symmetric, colindex);
//        thrust::stable_sort_by_key(thrust::host, row_coo_extend_copy, row_coo_extend_copy + nnz_symmetric, value);
//        delete[] row_coo_extend_copy;

//        int *d_row_coo_extend;
//        int *d_colindex;
//        CUDA_RT_CALL(cudaMalloc((void **) &d_row_coo_extend, sizeof(int) * nnz_symmetric));
//        CUDA_RT_CALL(cudaMalloc((void **) &d_colindex, sizeof(int) * nnz_symmetric));
//
//        CUDA_RT_CALL(cudaMemcpy(d_row_coo_extend, row_coo_extend, sizeof(int) * nnz_symmetric, cudaMemcpyHostToDevice));
//        CUDA_RT_CALL(cudaMemcpy(d_colindex, colindex, sizeof(int) * nnz_symmetric, cudaMemcpyHostToDevice));
//
//        void     *d_temp_storage = nullptr;
//        size_t   temp_storage_bytes = 0;
//        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
//                                        d_row_coo_extend, d_row_coo_extend, d_colindex, d_colindex, nnz_symmetric);
//        cudaMalloc(&d_temp_storage, temp_storage_bytes);
//        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
//                                        d_row_coo_extend, d_row_coo_extend, d_colindex, d_colindex, nnz_symmetric);
//        cudaFree(d_temp_storage);
//
//        CUDA_RT_CALL(cudaMemcpy(row_coo_extend, d_row_coo_extend, sizeof(int) * nnz_symmetric, cudaMemcpyDeviceToHost));
//        CUDA_RT_CALL(cudaMemcpy(colindex, d_colindex, sizeof(int) * nnz_symmetric, cudaMemcpyDeviceToHost));
//
//        CUDA_RT_CALL(cudaFree(d_row_coo_extend));
//        CUDA_RT_CALL(cudaFree(d_colindex));

        // convert coo to csr
        offset = new vertex_t[nrow + 1];
        for (i = 0; i < (nrow + 1); i++) {
            offset[i] = 0;
        }
        for (i = 0; i < nnz_symmetric; i++) {
            offset[row_coo_extend[i] + 1]++;
        }
        for (i = 0; i < nrow; i++) {
            offset[i + 1] = offset[i + 1] + offset[i];
        }
        assert(offset[nrow] == nnz_symmetric);
        delete[] row_coo_extend;
    } else {

        vertex_t row_id;
        vertex_t col_id;

        offset = new vertex_t[nrow + 1];
        for (i = 0; i < (nrow + 1); i++) {
            offset[i] = 0;
        }

        while (fgets(line, LINE_LENGTH_MAX, fp)) {
            ch = line;
            row_id = (vertex_t)(atoi(ch) - 1);
            ch = strchr(ch, ' ');
            ch++;
            col_id = (vertex_t)(atoi(ch) - 1);
            ch = strchr(ch, ' ');

            if (col_id >= row_id) continue;
            row_coo[num] = row_id;
            col_coo[num] = col_id;

            if (ch != NULL) {
                ch++;
                /* Read third word (value data)*/
                val_coo[num] = (weight_t) atof(ch);
                ch = strchr(ch, ' ');
            } else {
                val_coo[num] = 1.0;
            }

            offset[row_id + 1]++;
            offset[col_id + 1]++;
            num++;
        }

        fclose(fp);
        delete[] line;

        for (i = 0; i < nrow; i++) {
            offset[i + 1] = offset[i + 1] + offset[i];
        }
        assert(offset[nrow] == 2 * num);

        nnz_symmetric = offset[nrow];
        *nz = nnz_symmetric;

        colindex = new edge_t[nnz_symmetric];
        value = new weight_t[nnz_symmetric];

        omp_set_num_threads(80);
        edge_t* Counter = new edge_t [nrow];
#pragma omp parallel for
        for (vertex_t k = 0; k < nrow; k++)
            Counter[k] = 0;

#pragma omp parallel for
        for (vertex_t k = 0; k < num; k++) {
            vertex_t row_id_thread = row_coo[k];
            vertex_t col_id_thread = col_coo[k];
            weight_t weight_thread = val_coo[k];
            vertex_t index = offset[row_id_thread] + __sync_fetch_and_add(&Counter[row_id_thread], 1);
            colindex[index] = col_id_thread;
            value[index] = weight_thread;
            index = offset[col_id_thread] + __sync_fetch_and_add(&Counter[col_id_thread], 1);
            colindex[index] = row_id_thread;
            value[index] = weight_thread;
        }

        delete[] Counter;
        delete[] row_coo;
        delete[] col_coo;
        delete[] val_coo;
    }
}

int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, vertex_t *m, vertex_t *n, edge_t *nnz,
                       weight_t **aVal, vertex_t **aRowInd, edge_t **aColInd, int extendSymMatrix) {
    init_data_from_mtx(filename, *aRowInd, *aColInd, *aVal, m, n, nnz);
    return 0;
}
