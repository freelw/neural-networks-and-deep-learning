#ifndef NETWORK_V2_H
#define NETWORK_V2_H

#include <vector>

class Matrix;

Matrix sigmoid(Matrix m);

Matrix sigmoid_prime(Matrix m);

class NetWork {

public:
    NetWork(const std::vector<int> &_sizes);
    Matrix feedforward(const Matrix &a);

private:
    std::vector<int> sizes;
    int num_layers;
    std::vector<Matrix> biases;
    std::vector<Matrix> weights;
};

#endif