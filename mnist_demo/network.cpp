#include "network.h"
#include "mnist_loader.h"
#include <math.h>
#include <stdio.h>
#include <iostream>

using namespace std;

double randdouble() {
    return rand() * 1. / RAND_MAX - 0.5;
}

double randinterval(int l, int r) {
    return rand() * 1. / RAND_MAX * (r-l) + l;
}

void NetWork::initwb(double *** & _weights, double ** & _bias) {
    _bias = new double* [num_layers-1];
    _weights = new double** [num_layers-1];
    for (size_t i = 0; i < num_layers-1; ++ i) {
        size_t x = sizes[i];
        size_t y = sizes[i+1];
        _bias[i] = new double[y];
        for (size_t j = 0; j < y; ++ j) {
            _bias[i][j] = randdouble();
        }
        _weights[i] = new double*[y];
        for (size_t j = 0; j < y; ++ j) {
            _weights[i][j] = new double[x];
            for (size_t k = 0; k < x; ++ k) {
                _weights[i][j][k] = randdouble();
            }
        }
    }
}

void NetWork::fill0wb(double *** & _weights, double ** & _bias) {
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
        for (size_t j = 0; j < y; ++ j) {
            nabla_bias[i][j] += delta_bias[i][j];
        }
        for (size_t j = 0; j < y; ++ j) {
            for (size_t k = 0; k < x; ++ k) {
                nabla_weights[i][j][k] += delta_weights[i][j][k];
            }
        }
    }
}

void NetWork::minus_wb(int batch_cnt, double eta) {
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
    tmpv = new double[max_size];
    delta = new double[max_size];
    cost_der = new double[max_size];
    sp = new double[max_size];
    z = new double[max_size];

    activations = new double*[num_layers];
    for (size_t i = 0; i < num_layers; ++ i) {
        size_t y = sizes[i];
        activations[i] = new double[y];
        for (size_t j = 0; j < y; ++ j) {
            activations[i][j] = 0;
        }
    }
    zs = new double*[num_layers-1];
    for (size_t i = 0; i < num_layers-1; ++ i) {
        size_t y = sizes[i+1];
        zs[i] = new double[y];
        for (size_t j = 0; j < y; ++ j) {
            zs[i][j] = 0;
        }
    }
}

NetWork::~NetWork() {
}

std::vector<double> NetWork::feedforward(const std::vector<double> & a) {
    for (size_t i = 0, len = a.size(); i < len; ++ i) {
        tmpv[i] = a[i];
    }

    for (size_t i = 0; i < num_layers-1; ++ i) {
        size_t x = sizes[i];
        size_t y = sizes[i+1];
        std::vector<double> t;
        t.reserve(1024);
        for (size_t j = 0; j < x; ++ j) {
            t.push_back(tmpv[j]);
        }
        dot(weights[i], t, tmpv, x, y);
        for (size_t j = 0; j < y; ++ j) {
            tmpv[j] += bias[i][j];
        }
        sigmoid_array(tmpv, y);
    }
    std::vector<double> ret;
    for (size_t i = 0, len = sizes[num_layers-1]; i < len; ++ i) {
        ret.push_back(tmpv[i]);
    }
    return ret;
}

void NetWork::sigmoid_array(double *arr, size_t size) {
    for (size_t i = 0; i < size; ++ i) {
        arr[i] = sigmoid(arr[i]);
    }
}

double NetWork::sigmoid(double z) {
    return 1./(1.+exp(-z));
}

void NetWork::sigmoid_prime_array(double *arr, size_t size) {
    for (size_t i = 0; i < size; ++ i) {
        arr[i] = sigmoid_prime(arr[i]);
    }
}

double NetWork::sigmoid_prime(double z) {
    return sigmoid(z)*(1-sigmoid(z));
}

void NetWork::dot(double **w, const std::vector<double> & t, double *tmpv, size_t x, size_t y) {
    for (size_t j = 0; j < y; ++ j) {
        double sum = 0;
        for (size_t i = 0; i < x; ++ i) {
            sum += w[j][i] * t[i];
        }
        tmpv[j] = sum;
    }
}

void NetWork::shuffle(double **p_training_data_x, double **p_training_data_y, size_t len) {
    for (int i = len-1; i >= 0; -- i) {
        size_t index = size_t(randinterval(0, len));
        double *tmp = p_training_data_x[index];
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
    double eta
) {
    double **p_training_data_x = new double*[loader.training_data_len];
    double **p_training_data_y = new double*[loader.training_data_len];
    for (size_t j = 0; j < loader.training_data_len; ++ j) {
        p_training_data_x[j] = loader.training_data_x[j];
        p_training_data_y[j] = loader.training_data_y[j];
    }
    for (size_t j = 0; j < epochs; ++ j) {
        shuffle(p_training_data_x, p_training_data_y, loader.training_data_len);
        for (size_t offset = 0; offset < loader.training_data_len; offset += mini_batch_size) {
            update_mini_batch(p_training_data_x, p_training_data_y, offset, mini_batch_size, loader.training_data_len, eta);
        }
        int evaluate_cnt = evaluate();
        std::cout << "Epoch " << j << ": " << evaluate_cnt << " / " << loader.test_data_len << std::endl;
    }
}

void NetWork::update_mini_batch(double **p_training_data_x, double **p_training_data_y, size_t offset, int mini_batch_size, size_t len, double eta) {
    fill0wb(nabla_weights, nabla_bias);
    int batch_size = mini_batch_size > len - offset ? len - offset : mini_batch_size;
    int batch_cnt = 0;
    for (size_t i = offset; i < offset + batch_size; ++ i) {
        ++ batch_cnt;
        fill0wb(delta_weights, delta_bias);
        backprop(p_training_data_x[i], p_training_data_y[i]);
        add_nabla_delta();
    }
    minus_wb(batch_cnt, eta);
}

void NetWork::backprop(double *px, double *py) {
    for (size_t i = 0; i < sizes[0]; ++ i) {
        activations[0][i] = px[i];
    }
    for (size_t i = 0; i < num_layers-1; ++ i) {
        size_t x = sizes[i];
        size_t y = sizes[i+1];
        std::vector<double> t;
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
    for (int l = num_layers-3; l >= 0; -- l) {
        size_t x = sizes[l];
        size_t y = sizes[l+1];
        size_t z = sizes[l+2];
        for (size_t i = 0; i < x; ++ i) {
            sp[i] = zs[l][i];
        }
        sigmoid_prime_array(sp, x);
        for (size_t i = 0; i < y; ++ i) {
            for (size_t j = 0; j < z; ++ j) {
                delta_bias[l][i] += weights[l+1][j][i] * delta_bias[l+1][j] * sp[i];
            }
        }
        
        for (size_t i = 0; i < y; ++ i) {
            for (size_t j = 0; j < x; ++ j) {
                delta_weights[l][i][j] = delta_bias[l][i] * activations[l][j];
            }
        }
    }
}

int NetWork::evaluate() {
    int sum = 0;
    for (size_t i = 0; i < loader.test_data_len; ++ i) {
        std::vector<double> input;
        for (size_t j = 0; j < loader.test_data_x_len; ++ j) {
            input.push_back(loader.test_data_x[i][j]);
        }
        std::vector<double> res = feedforward(input);
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

void NetWork::cost_derivative(double *pa, double *py, double *c_d, size_t size) {
    for (size_t i = 0; i < size; ++ i) {
        c_d[i] = pa[i] - py[i];
    }
}

void NetWork::printwb() {
    for (size_t i = 0; i < num_layers - 1; ++ i) {
        size_t x = sizes[i];
        size_t y = sizes[i+1];
        cout << "[" << i  << "] w " << weights[i][0][0] << endl;
        cout << "[" << i  << "] b ";
        for (size_t j = 0; j < y; ++ j) {
            cout << bias[i][j] << " ";
        }
        cout << endl;
    }
}

void NetWork::printnwb() {
    for (size_t i = 0; i < num_layers - 1; ++ i) {
        size_t x = sizes[i];
        size_t y = sizes[i+1];
        cout << "[" << i  << "] nw " << nabla_weights[i][0][0] << endl;
        cout << "[" << i  << "] nb ";
        for (size_t j = 0; j < y; ++ j) {
            cout << nabla_bias[i][j] << " ";
        }
        cout << endl;
    }
}
