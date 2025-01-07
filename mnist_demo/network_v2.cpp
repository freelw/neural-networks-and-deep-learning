#include "network_v2.h"
#include "matrix.h"
#include <math.h>

#include <iostream>

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
}

Matrix NetWork::feedforward(const Matrix &a) {
    Matrix res(a);
    for (auto i = 0; i < sizes.size()-1; ++ i) {
        res = sigmoid(weights[i].dot(res) + biases[i]);
    }
    return res;
}
