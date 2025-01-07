#ifndef NETWORK_V2_H
#define NETWORK_V2_H

#include <vector>

class Matrix;

Matrix sigmoid(Matrix m);

Matrix sigmoid_prime(Matrix m);

class TrainingData;

class NetWork {

public:
    NetWork(const std::vector<int> &_sizes);
    Matrix feedforward(const Matrix &a);
    void SGD(std::vector<TrainingData*> &v_training_data, int epochs, int mini_batch_size, double eta);

private:
    std::vector<int> sizes;
    int num_layers;
    std::vector<Matrix> biases;
    std::vector<Matrix> weights;
};

#endif