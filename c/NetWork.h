#ifndef NETWORK_H
#define NETWORK_H

#include "mnist_loader.h"
#include <vector>

class NetWork
{
private:
    int num_layers;
    std::vector<int> sizes;
    float **bias;
    float ***weights;
    float **nabla_bias;
    float ***nabla_weights;
    float **backprop_nabla_bias;
    float ***backprop_nabla_weights;
    float **delta_bias;
    float ***delta_weights;
    int max_size;
    float *tmpv;

    float *delta;
    float *cost_der;
    float *sp;
    float *z;
    float **activations;
    float **zs;
    MnistLoader &loader;
public:
    NetWork(const std::vector<int> &_sizes, MnistLoader &_loader);
    ~NetWork();
    std::vector<float> NetWork::feedforward(const std::vector<float> & a);
    void sigmoid_array(float *arr, size_t size);
    float sigmoid(float z);
    void sigmoid_prime_array(float *arr, size_t size);
    float sigmoid_prime(float z);
    void dot(float **, const std::vector<float> & t, float *tmpv, size_t x, size_t y);
    void shuffle(float **p_training_data_x, float **p_training_data_y, size_t len);
    void SGD(
        int epochs,
        int mini_batch_size,
        float eta
    );
    void update_mini_batch(float **p_training_data_x, float **p_training_data_y, size_t offset, int mini_batch_size, size_t len, float eta);
    void initwb(float *** & weights, float ** & bias);
    void fill0wb(float *** & _weights, float ** & _bias);
    void add_nabla_delta();
    void minus_wb(int batch_cnt, float eta);
    void backprop(float *x, float *y);
    void cost_derivative(float *pa, float *py, float *c_d, size_t size);
};


#endif