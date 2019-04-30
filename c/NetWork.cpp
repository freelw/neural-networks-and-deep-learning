#include "network.h"

#include <stdio.h>

float randfloat() {
    return rand() / 32767.;
}

NetWork::NetWork(const std::vector<int> &_sizes)
    : num_layers(_sizes.size()),
    sizes(_sizes)   
{
    bias = new float* [num_layers-1];
    weights = new float** [num_layers-1];
    for (size_t i = 0; i < num_layers-1; ++ i) {
        size_t x = sizes[i];
        size_t y = sizes[i+1];
        bias[i] = new float[y];
        for (size_t j = 0; j < y; ++ j) {
            bias[i][j] = randfloat();
        }
        weights[i] = new float*[y];
        for (size_t j = 0; j < y; ++ j) {
            weights[i][j] = new float[x];
            for (size_t k = 0; k < x; ++ k) {
                weights[i][j][k] = randfloat();
            }
        }
    }
}

NetWork::~NetWork()
{
}
