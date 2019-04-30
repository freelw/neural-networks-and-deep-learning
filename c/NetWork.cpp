#include "network.h"
#include "mnist_loader.h"
#include <math.h>
#include <stdio.h>

float randfloat() {
    return rand() / 32767.;
}

float randinterval(int l, int r) {
    return rand() / 32767. * (r-l) + l;
}

void NetWork::initwb(float *** & _weights, float ** & _bias) {
    _bias = new float* [num_layers-1];
    _weights = new float** [num_layers-1];
    for (size_t i = 0; i < num_layers-1; ++ i) {
        size_t x = sizes[i];
        size_t y = sizes[i+1];
        _bias[i] = new float[y];
        for (size_t j = 0; j < y; ++ j) {
            _bias[i][j] = randfloat();
        }
        _weights[i] = new float*[y];
        for (size_t j = 0; j < y; ++ j) {
            _weights[i][j] = new float[x];
            for (size_t k = 0; k < x; ++ k) {
                _weights[i][j][k] = randfloat();
            }
        }
    }
}

void NetWork::fill0wb(float *** & _weights, float ** & _bias) {
    for (size_t i = 0; i < num_layers-1; ++ i) {
        size_t x = sizes[i];
        size_t y = sizes[i+1];
        for (size_t j = 0; j < y; ++ j) {
            _bias[i][j] = 0;
        }
        for (size_t j = 0; j < y; ++ j) {
            for (size_t k = 0; k < x; ++ k) {
                _weights[i][j][k] = 0;
            }
        }
    }
}

NetWork::NetWork(const std::vector<int> &_sizes, MnistLoader &_loader)
    : num_layers(_sizes.size()),
    sizes(_sizes),
    loader(_loader) {

    initwb(weights, bias);
    initwb(nabla_weights, nabla_bias);
    initwb(delta_weights, delta_bias);
    
    for (size_t i = 0; i < num_layers; ++ i) {
        if (max_size < sizes[i]) {
            max_size = sizes[i];
        }
    }
    tmpv = new float[max_size];
}

NetWork::~NetWork() {
}

std::vector<float> NetWork::feedforward(const std::vector<float> & a) {
    memset(tmpv, 0, sizeof(float)*max_size);
    for (size_t i = 0, len = a.size(); i < len; ++ i) {
        tmpv[i] = a[i];
    }
    for (size_t i = 0; i < num_layers-1; ++ i) {
        size_t x = sizes[i];
        size_t y = sizes[i+1];
        std::vector<float> t;
        for (size_t j = 0; j < x; ++ j) {
            t.push_back(tmpv[j]);
        }
        dot(weights[i], t, tmpv, x, y);
        for (size_t j = 0; j < y; ++ j) {
            tmpv[j] += bias[i][j];
        }
    }
}

void NetWork::sigmoid_array(float *arr, size_t size) {
    for (size_t i = 0; i < size; ++ i) {
        arr[i] = sigmoid(arr[i]);
    }
}

float NetWork::sigmoid(float z) {
    return 1./(1.+exp(-z));
}

void NetWork::dot(float **w, const std::vector<float> & t, float *tmpv, size_t x, size_t y) {
    for (size_t j = 0; j < y; ++ j) {
        float sum = 0;
        for (size_t i = 0; i < x; ++ i) {
            sum += w[j][i] * t[i];
        }
        tmpv[j] = sum;
    }
}

void NetWork::shuffle(float **p_training_data_x, float **p_training_data_y, size_t len) {
    for (int i = len-1; i >= 0; -- i) {
        size_t index = size_t(randinterval(0, len));
        float *tmp = p_training_data_x[index];
        p_training_data_x[index] = p_training_data_x[i];
        p_training_data_x[i] = tmp;
        tmp = p_training_data_y[index];
        p_training_data_y[index] = p_training_data_y[i];
        p_training_data_y[i] = tmp;
    }
}

void NetWork::SGD(
    int epochs,
    int mini_batch_size,
    float eta
) {
    float **p_training_data_x = new float*[loader.training_data_len];
    float **p_training_data_y = new float*[loader.training_data_len];
    for (size_t j = 0; j < loader.training_data_len; ++ j) {
        p_training_data_x[j] = loader.training_data_x[j];
        p_training_data_y[j] = loader.training_data_y[j];
    }
    for (size_t j = 0; j < epochs; ++ j) {
        shuffle(p_training_data_x, p_training_data_y, loader.training_data_len);
        for (size_t offset = 0; offset < loader.training_data_len; offset += mini_batch_size) {
            update_mini_batch(p_training_data_x, p_training_data_y, offset, mini_batch_size, loader.training_data_len);
        }
    }
}

void NetWork::update_mini_batch(float **p_training_data_x, float **p_training_data_y, size_t offset, int mini_batch_size, size_t len) {
    fill0wb(nabla_weights, nabla_bias);
    fill0wb(delta_weights, delta_bias);

    for (size_t i = offset; i < offset + mini_batch_size; ++ i) {

        backprop(p_training_data_x[i], p_training_data_y[i]);
    }
}

void NetWork::backprop(float *x, float *y) {
    
}
