#include "network.h"
#include "mnist_loader.h"
#include <math.h>
#include <stdio.h>
#include <iostream>

using namespace std;

float randfloat() {
    //return 0;
    return rand() * 1. / RAND_MAX - 0.5;
}

float randinterval(int l, int r) {
    return rand() * 1. / RAND_MAX * (r-l) + l;
}

void NetWork::initwb(float *** & _weights, float ** & _bias) {
    _bias = new float* [num_layers-1];
    _weights = new float** [num_layers-1];
    for (size_t i = 0; i < num_layers-1; ++ i) {
        size_t x = sizes[i];
        size_t y = sizes[i+1];
        //cout << "initwb  y : " << y << endl;
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

void NetWork::add_nabla_delta() {
    for (size_t i = 0; i < num_layers-1; ++ i) {
        size_t x = sizes[i];
        size_t y = sizes[i+1];
        //cout << "i x y : " << i << " " << x << " " << y << endl;
        for (size_t j = 0; j < y; ++ j) {
            //cout << "i j: " << i << " " << j << endl;
            //cout << "nabla_bias[i][j] delta_bias[i][j] " << nabla_bias[i][j] << " " << delta_bias[i][j] << endl;
            nabla_bias[i][j] += delta_bias[i][j];
        }
        for (size_t j = 0; j < y; ++ j) {
            for (size_t k = 0; k < x; ++ k) {
                //cout << "i x k : " << i << " " << x << " " << k << endl;
                nabla_weights[i][j][k] += delta_weights[i][j][k];
            }
        }
    }
}

void NetWork::minus_wb(int batch_cnt, float eta) {
    for (size_t i = 0; i < num_layers-1; ++ i) {
        size_t x = sizes[i];
        size_t y = sizes[i+1];
        for (size_t j = 0; j < y; ++ j) {
            for (size_t k = 0; k < x; ++ k) {
                weights[i][j][k] -= eta/batch_cnt*nabla_weights[i][j][k];
            }
            bias[i][j] -= eta/batch_cnt*nabla_bias[i][j];
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
    initwb(backprop_nabla_weights, backprop_nabla_bias);
    
    for (size_t i = 0; i < num_layers; ++ i) {
        if (max_size < sizes[i]) {
            max_size = sizes[i];
        }
    }
    tmpv = new float[max_size];
    delta = new float[max_size];
    cost_der = new float[max_size];
    sp = new float[max_size];
    z = new float[max_size];

    activations = new float*[num_layers];
    for (size_t i = 0; i < num_layers; ++ i) {
        size_t y = sizes[i];
        activations[i] = new float[y];
        for (size_t j = 0; j < y; ++ j) {
            activations[i][j] = 0;
        }
    }
    zs = new float*[num_layers-1];
    for (size_t i = 0; i < num_layers-1; ++ i) {
        size_t y = sizes[i+1];
        zs[i] = new float[y];
        for (size_t j = 0; j < y; ++ j) {
            zs[i][j] = 0;
        }
    }
}

NetWork::~NetWork() {
}

std::vector<float> NetWork::feedforward(const std::vector<float> & a) {
    //memset(tmpv, 0, sizeof(float)*max_size);
    //cout << "feedforward a.size()" << a.size() << endl;
    for (size_t i = 0, len = a.size(); i < len; ++ i) {
        tmpv[i] = a[i];
        //cout << tmpv[i] << " ";
    }
    //cout << endl;

    for (size_t i = 0; i < num_layers-1; ++ i) {
        size_t x = sizes[i];
        size_t y = sizes[i+1];
        //cout << "num_layers i x y " << num_layers << " " << i << " " << x << " " << y << endl;
        std::vector<float> t;
        t.reserve(1024);
        //cout << "tmpv ";
        for (size_t j = 0; j < x; ++ j) {
            t.push_back(tmpv[j]);
            //cout << tmpv[j] << " ";
        }
        //cout << endl;
        dot(weights[i], t, tmpv, x, y);
        //cout << "tmpv out  ";
        for (size_t j = 0; j < y; ++ j) {
            tmpv[j] += bias[i][j];
            //cout << tmpv[j] << " ";
        }
        //cout << endl;
        sigmoid_array(tmpv, y);
    }
    std::vector<float> ret;
    for (size_t i = 0, len = sizes[num_layers-1]; i < len; ++ i) {
        ret.push_back(tmpv[i]);
        //cout << tmpv[i] << " ";
    }
    //cout << endl;
    //cout << "ret.size() " << ret.size() << endl;
    return ret;
}

void NetWork::sigmoid_array(float *arr, size_t size) {
    for (size_t i = 0; i < size; ++ i) {
        arr[i] = sigmoid(arr[i]);
    }
}

float NetWork::sigmoid(float z) {
    return 1./(1.+exp(-z));
}

void NetWork::sigmoid_prime_array(float *arr, size_t size) {
    for (size_t i = 0; i < size; ++ i) {
        arr[i] = sigmoid_prime(arr[i]);
    }
}

float NetWork::sigmoid_prime(float z) {
    return sigmoid(z)*(1-sigmoid(z));
}

void NetWork::dot(float **w, const std::vector<float> & t, float *tmpv, size_t x, size_t y) {
    //cout << "dot ";
    for (size_t j = 0; j < y; ++ j) {
        float sum = 0;
        for (size_t i = 0; i < x; ++ i) {
            sum += w[j][i] * t[i];
        }
        tmpv[j] = sum;
        //cout << tmpv[j] << " ";
    }
    //cout << endl;
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
    //cout << "training_data_len : " << loader.training_data_len << endl;
    float **p_training_data_x = new float*[loader.training_data_len];
    float **p_training_data_y = new float*[loader.training_data_len];
    for (size_t j = 0; j < loader.training_data_len; ++ j) {
        p_training_data_x[j] = loader.training_data_x[j];
        p_training_data_y[j] = loader.training_data_y[j];
    }

    //cout << "dbg1" << endl;
    for (size_t j = 0; j < epochs; ++ j) {
        //cout << "before shuffle" << endl;
        shuffle(p_training_data_x, p_training_data_y, loader.training_data_len);
        //cout << "after shuffle" << endl;
        for (size_t offset = 0; offset < loader.training_data_len; offset += mini_batch_size) {
            //cout << "offset : " << offset << endl;
            update_mini_batch(p_training_data_x, p_training_data_y, offset, mini_batch_size, loader.training_data_len, eta);
        }
        //cout << "before evaluate" << endl;
        int evaluate_cnt = evaluate();
        std::cout << "Epoch " << j << ": " << evaluate_cnt << " / " << loader.test_data_len << std::endl;
    }
}

void NetWork::update_mini_batch(float **p_training_data_x, float **p_training_data_y, size_t offset, int mini_batch_size, size_t len, float eta) {
    fill0wb(nabla_weights, nabla_bias);
    int batch_size = mini_batch_size > len - offset ? len - offset : mini_batch_size;
    int batch_cnt = 0;
    //cout << "batch_size : " << batch_size << endl;
    for (size_t i = offset; i < offset + batch_size; ++ i) {
        //cout << "i : " << i << endl;
        ++ batch_cnt;
        fill0wb(delta_weights, delta_bias);
        //cout << "after fill0wb(delta_weights, delta_bias);" << endl;
        backprop(p_training_data_x[i], p_training_data_y[i]);
        //cout << "after bp" << endl;
        add_nabla_delta();
        //cout << "after bp1" << endl;
    }
    minus_wb(batch_cnt, eta);
}

void NetWork::backprop(float *px, float *py) {
    //fill0wb(backprop_nabla_weights, backprop_nabla_bias);
    for (size_t i = 0; i < sizes[0]; ++ i) {
        activations[0][i] = px[i];
    }
    for (size_t i = 0; i < num_layers-1; ++ i) {
        size_t x = sizes[i];
        size_t y = sizes[i+1];
        std::vector<float> t;
        for (size_t j = 0; j < x; ++ j) {
            t.push_back(activations[i][j]);
        }
        dot(weights[i], t, zs[i], x, y);
        for (size_t j = 0; j < y; ++ j) {
            zs[i][j]+=bias[i][j];
            activations[i+1][j] = sigmoid(zs[i][j]);
        }
    }
    cost_derivative(activations[num_layers-1], py, cost_der, loader.training_data_y_len);
    for (size_t i = 0; i < loader.training_data_y_len; ++ i) {
        cost_der[i] *= sigmoid_prime(zs[num_layers-2][i]);
        delta_bias[num_layers-2][i] = cost_der[i];
    }
    size_t x = sizes[num_layers-2];
    size_t y = sizes[num_layers-1];
    for (size_t i = 0; i < y; ++ i) {
        for (size_t j = 0; j < x; ++ j) {
            delta_weights[num_layers-2][i][j] = cost_der[i]*activations[num_layers-2][j];
        }
    }
    for (int l = num_layers-3; l >= 1; -- l) {
        size_t x = sizes[l];
        size_t y = sizes[l+1];
        for (size_t i = 0; i < x; ++ i) {
            sp[i] = zs[l][i];
        }
        sigmoid_prime_array(sp, x);
        for (size_t i = 0; i < y; ++ i) {
            for (size_t j = 0; j < x; ++ j) {
                delta_bias[l][j] += weights[l+1][i][j] * delta_bias[l+1][i] * sp[j];
            }
        }
        size_t z = sizes[l-1];
        for (size_t i = 0; i < x; ++ i) {
            for (size_t j = 0; j < z; ++ j) {
                delta_weights[l][i][j] = delta_bias[l][i] * activations[l-1][j];
            }
        }
    }
}

int NetWork::evaluate() {
    int sum = 0;
    cout << "loader.test_data_len : " << loader.test_data_len << endl;
    for (size_t i = 0; i < loader.test_data_len; ++ i) {
        cout << "i : " << i << " sum : " << sum <<endl;
        std::vector<float> input;
        for (size_t j = 0; j < loader.test_data_x_len; ++ j) {
            input.push_back(loader.test_data_x[i][j]);
        }
        //cout << "before feedforward" << endl;
        std::vector<float> res = feedforward(input);
        //cout << "after feedforward" << endl;
        int max_index = 0;
        for (size_t j = 0; j < res.size(); ++ j) {
            if (res[j] > res[max_index]) {
                max_index = j;
            }    
        }
        if (max_index == loader.test_data_y[i]) {
            ++ sum;
        }
    }
    return sum;
}

void NetWork::cost_derivative(float *pa, float *py, float *c_d, size_t size) {
    for (size_t i = 0; i < size; ++ i) {
        c_d[i] = pa[i] - py[i];
    }
}