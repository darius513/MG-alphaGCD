#include "../include/common.h"

#include <iostream>

bool get_arg(char **begin, char **end, const std::string &arg) {
    char **itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

long long unsigned int parse_nvshmem_symmetric_size(char *value) {
    long long unsigned int units, size;

    assert(value != NULL);

    if (strchr(value, 'G') != NULL) {
        units = 1e9;
    } else if (strchr(value, 'M') != NULL) {
        units = 1e6;
    } else if (strchr(value, 'K') != NULL) {
        units = 1e3;
    } else {
        units = 1;
    }

    assert(atof(value) >= 0);
    size = (long long unsigned int)atof(value) * units;

    return size;
}

unsigned int iDivUp(unsigned int dividend, unsigned int divisor) {
    return ((dividend % divisor) == 0) ? (dividend / divisor)
                                       : (dividend / divisor + 1);
}
