#include "network_v2.h"
#include "matrix.h"
#include <math.h>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <random>

TrainingData::TrainingData(int input_layer_size, int _y)
    : x(Shape(input_layer_size, 1)), y(_y) {
    x.zero();
}

double sigmoid_double(double z) {
    double ret = 1./(1.+exp(-z));

    // cout << "z : " << z << endl;
    // cout << "ret : " << ret << endl;
    return ret;
}

double sigmoid_prime_double(double z) {
    return sigmoid_double(z)*(1-sigmoid_double(z));
}

Matrix sigmoid(Matrix m) {
    Shape shape = m.getShape();
    Matrix res(m);
    for (auto i = 0; i < shape.rowCnt; ++i) {
        for (auto j = 0; j < shape.colCnt; ++j) {
            res[i][j] = sigmoid_double(res[i][j]);
        }
    }
    return res;
}

Matrix sigmoid_prime(Matrix m) {
    // Shape shape = m.getShape();
    // Matrix res(m);
    // for (auto i = 0; i < shape.rowCnt; ++i) {
    //     for (auto j = 0; j < shape.colCnt; ++j) {
    //         res[i][j] = sigmoid_prime_double(res[i][j]);
    //     }
    // }
    // return res;
    return sigmoid(m) * (1 - sigmoid(m));
}

NetWork::NetWork(const std::vector<int> &_sizes)
        : sizes(_sizes), num_layers(_sizes.size()) {
    for (auto i = 1; i < sizes.size(); ++ i) {
        biases.emplace_back(Matrix(Shape(sizes[i], 1)).zero());
    }
    for (auto i = 1; i < sizes.size(); ++ i) {
        weights.emplace_back(Matrix(Shape(sizes[i], sizes[i-1])).zero());
    }
    assert(biases.size() == weights.size());
}

Matrix NetWork::feedforward(const Matrix &a) {
    Matrix res(a);
    // cout << "feedforward a : " << a << endl;
    for (auto i = 0; i < sizes.size()-1; ++ i) {
        // cout << "weights[" << i << "]" << weights[i] << endl;
        // cout << "biases[" << i << "]" << biases[i] << endl;
        res = sigmoid(weights[i].dot(res) + biases[i]);
        // cout << "feedforward res : " << res << endl;
    }
    // cout << "feedforward a : " << a << endl;
    // cout << "feedforward res : " << res << endl;
    return res;
}

void NetWork::SGD(
    std::vector<TrainingData*> &v_training_data,
    std::vector<TrainingData*> &v_test_data,
    int epochs, int mini_batch_size, double eta) {

    int n = v_training_data.size();
    for (auto e = 0; e < epochs; ++ e) {
        // std::cout << "Epoch " << e << " start . ------------------------" << endl;
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(v_training_data), std::end(v_training_data), rng);
        std::vector<std::vector<TrainingData*>> mini_batches;
        for (auto i = 0; i < n; i += mini_batch_size) {
            std::vector<TrainingData*> tmp;
            auto end = min(i+mini_batch_size, n);
            tmp.assign(v_training_data.begin()+i,v_training_data.begin()+end);
            mini_batches.emplace_back(tmp);
        }

        for (auto i = 0; i < mini_batches.size(); ++ i) {
            update_mini_batch(mini_batches[i], eta);
        }
        std::cout << "Epoch " << e << " : " << evaluate(v_test_data) << " / " << n << std::endl;
    }   
}

void NetWork::update_mini_batch(
    std::vector<TrainingData*> &mini_batch,
    double eta) {
        
    std::vector<Matrix> nabla_b;
    std::vector<Matrix> nabla_w;
    const auto L = sizes.size() - 1;
    for (auto i = 0; i < L; ++ i) {
        nabla_b.emplace_back(Matrix(biases[i].getShape()).zero());
    }

    for (auto i = 0; i < L; ++ i) {
        nabla_w.emplace_back(Matrix(weights[i].getShape()).zero());
    }

    for (auto i = 0; i < mini_batch.size(); ++ i) {
        std::vector<Matrix> delta_nabla_b;
        std::vector<Matrix> delta_nabla_w;
        Matrix y(Shape(sizes[L], 1));
        y.zero();
        y[mini_batch[i]->y][0] = 1;
        backprop(mini_batch[i]->x, y, delta_nabla_b, delta_nabla_w);
        for (auto j = 0; j < L; ++ j) {
            nabla_b[j] = nabla_b[j] + delta_nabla_b[j];
            nabla_w[j] = nabla_w[j] + delta_nabla_w[j];
        }
    }

    for (auto i = 0; i < L; ++ i) {
        weights[i] = weights[i] - nabla_w[i] * eta / mini_batch.size();
        biases[i] = biases[i] - nabla_b[i] * eta / mini_batch.size();
    }
}

void NetWork::backprop(
    Matrix &x, Matrix &y,
    std::vector<Matrix> &delta_nabla_b,
    std::vector<Matrix> &delta_nabla_w) {
    
    // cout << "y : " << y << endl;
    const auto L = sizes.size() - 1;
    for (auto i = 0; i < L; ++ i) {
        delta_nabla_b.emplace_back(Matrix(biases[i].getShape()));
    }
    for (auto i = 0; i < L; ++ i) {
        delta_nabla_w.emplace_back(Matrix(weights[i].getShape()));
    }

    Matrix activation(x);
    std::vector<Matrix> activations;
    activations.emplace_back(activation);
    std::vector<Matrix> zs;
    for (auto i = 0; i < L; ++ i) {
        Matrix z = weights[i].dot(activation) + biases[i];
        zs.emplace_back(z);
        activation = sigmoid(z);
        activations.emplace_back(activation);
    }
    assert(activations.size() == L + 1);
    Matrix delta = cost_derivative(activations[L], y) * sigmoid_prime(zs[L-1]);
    // cout << "detlta L : " << delta << endl;
    // cout << "activations[L] : " << activations[L] << endl;
    // cout << "zs[L-1] : " << zs[L-1] << endl;
    // cout << "y : " << y << endl;
    for (int l = L-1; l >= 0; -- l) {
        // cout << "l : " << l << endl;
        // cout << "delta : " << delta << endl;
        delta_nabla_b[l] = delta;
        auto activation_transpose = activations[l].transpose();
        // cout << "activation_transpose : " << activation_transpose << endl;
        delta_nabla_w[l] = delta.dot(activation_transpose);
        // cout << "delta_nabla_w[" << l << "] " << delta_nabla_w[l] << endl;
        if (l >= 1) {
            // cout << "zs [" << l-1 << "] : " << zs[l-1] << endl;
            //cout << "delta before : " << delta << endl;
            delta = weights[l].transpose().dot(delta) * sigmoid_prime(zs[l-1]);
            // cout << "delta next : " << delta << endl;
        }
    }
}

int NetWork::evaluate(std::vector<TrainingData*> &v_test_data) {
    int sum = 0;
    for (auto i = 0; i < v_test_data.size(); ++ i) {
        Matrix res = feedforward(v_test_data[i]->x);
        // cout << "res : " << res << endl;
        int index = 0;
        for (auto j = 1; j < sizes[sizes.size() - 1]; ++ j) {
            if (res[j][0] > res[index][0]) {
                index = j;
            }
        }
        // cout << "evaluate : " << index << " " << v_test_data[i]->y << endl;
        if (index == v_test_data[i]->y) {
            sum ++;
        }
    }
    return sum;
}

Matrix NetWork::cost_derivative(
    const Matrix &output_activations,
    const Matrix &y) {
    return output_activations - y;
}