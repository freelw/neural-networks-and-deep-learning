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
    void update_mini_batch(std::vector<TrainingData*> &mini_batch, double eta);
    void backprop(Matrix &x, Matrix &y, std::vector<Matrix> &delta_nabla_b, std::vector<Matrix> &delta_nabla_w);
    Matrix cost_derivative(const Matrix &output_activations, const Matrix &y);

private:
    std::vector<int> sizes;
    int num_layers;
    std::vector<Matrix> biases;
    std::vector<Matrix> weights;
};

#endif