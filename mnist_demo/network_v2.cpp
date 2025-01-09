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
    // std::cout << "z : " << z << std::endl;
    // std::cout << "ret : " << ret << std::endl;
    return ret;
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
    return sigmoid(m)*(1-sigmoid(m));
}

NetWork::NetWork(const std::vector<int> &_sizes)
        : sizes(_sizes), num_layers(_sizes.size()) {
    for (auto i = 1; i < sizes.size(); ++ i) {
        biases.emplace_back(Matrix(Shape(sizes[i], 1)).zero());
    }
    for (auto i = 1; i < sizes.size(); ++ i) {
        weights.emplace_back(Matrix(Shape(sizes[i], sizes[i-i])).zero());
    }
    assert(biases.size() == weights.size());
}

Matrix NetWork::feedforward(const Matrix &a) {
    Matrix res(a);
    for (auto i = 0; i < sizes.size()-1; ++ i) {
        res = sigmoid(weights[i].dot(res) + biases[i]);
    }
    return res;
}

void NetWork::SGD(
    std::vector<TrainingData*> &v_training_data,
    std::vector<TrainingData*> &v_test_data,
    int epochs, int mini_batch_size, double eta) {

    int n = v_training_data.size();
    for (auto e = 0; e < epochs; ++ e) {
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
        std::cout << "Epoch " <<  e << " complete." << std::endl;
    }   
}

void NetWork::update_mini_batch(
    std::vector<TrainingData*> &mini_batch,
    double eta) {
        
    std::vector<Matrix> nabla_b;
    std::vector<Matrix> nabla_w;    
    for (auto i = 0; i < sizes.size()-1; ++ i) {
        nabla_b.emplace_back(Matrix(biases[i].getShape()).zero());
    }

    for (auto i = 0; i < sizes.size()-1; ++ i) {
        nabla_w.emplace_back(Matrix(weights[i].getShape()).zero());
    }

    for (auto i = 0; i < mini_batch.size(); ++ i) {
        std::vector<Matrix> delta_nabla_b;
        std::vector<Matrix> delta_nabla_w;
        Matrix y(Shape(sizes[sizes.size()-1], 1));
        y.zero();
        y[mini_batch[i]->y][0] = 1;
        backprop(mini_batch[i]->x, y, delta_nabla_b, delta_nabla_w);
        for (auto j = 0; j < sizes.size()-1; ++ j) {
            nabla_b[j] = nabla_b[j] + delta_nabla_b[j];
            nabla_w[j] = nabla_w[j] + delta_nabla_w[j];
        }
    }

    for (auto i = 0; i < sizes.size()-1; ++ i) {
        weights[i] = weights[i] - nabla_w[i] * (eta / mini_batch.size());
        biases[i] = biases[i] - nabla_b[i] * (eta / mini_batch.size());
    }
}


void NetWork::backprop(
    Matrix &x, Matrix &y,
    std::vector<Matrix> &delta_nabla_b,
    std::vector<Matrix> &delta_nabla_w) {
    
    for (auto i = 0; i < biases.size(); ++ i) {
        delta_nabla_b.emplace_back(Matrix(biases[i].getShape()).zero());
    }

    for (auto i = 0; i < weights.size(); ++ i) {
        delta_nabla_w.emplace_back(Matrix(weights[i].getShape()).zero());
    }

    // Matrix activation(x);

    // std::vector<Matrix> activations;
    // activations.emplace_back(activation);
    // std::vector<Matrix> zs;
    // for (auto i = 0; i < biases.size(); ++ i) {
    //     Matrix z = weights[i].dot(activation) + biases[i];
    //     zs.emplace_back(z);
    //     activation = sigmoid(z);
    //     activations.emplace_back(activation);
    // }
    // Matrix delta = cost_derivative(activations[activations.size()-1], y) * sigmoid_prime(zs[zs.size()-1]);
}

int NetWork::evaluate(std::vector<TrainingData*> &v_test_data) {
    int sum = 0;
    for (auto i = 0; i < v_test_data.size(); ++ i) {
        Matrix res = feedforward(v_test_data[i]->x);
        int index = 0;
        for (auto j = 1; j < sizes[sizes.size() - 1]; ++ j) {
            if (res[j][0] > res[index][0]) {
                index = j;
            }
        }
        if (index == v_test_data[i]->y) {
            sum ++;
        }
    }
    return sum;
}

Matrix NetWork::cost_derivative(
    const Matrix &output_activations,
    const Matrix &y) {
    //return output_activations - y;
    return y;
}