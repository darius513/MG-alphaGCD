#ifndef MMIO_WRAPPER_H
#define MMIO_WRAPPER_H
#include "./mmio.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cstring>
#include "../common.h"

int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, vertex_t *m, vertex_t *n, edge_t *nnz,
                       weight_t **aVal, vertex_t **aRowInd, edge_t **aColInd, int extendSymMatrix);

#endif //MMIO_WRAPPER_H
