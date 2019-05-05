#ifndef NETWORK_H
#define NETWORK_H

#include "mnist_loader.h"
#include <vector>

class NetWork
{
private:
    int num_layers;
    std::vector<int> sizes;
    double **bias;
    double ***weights;
    double **nabla_bias;
    double ***nabla_weights;
    double **backprop_nabla_bias;
    double ***backprop_nabla_weights;
    double **delta_bias;
    double ***delta_weights;
    int max_size;
    double *tmpv;

    double *delta;
    double *cost_der;
    double *sp;
    double *z;
    double **activations;
    double **zs;
    MnistLoader &loader;
public:
    NetWork(const std::vector<int> &_sizes, MnistLoader &_loader);
    ~NetWork();
    std::vector<double> feedforward(const std::vector<double> & a);
    void sigmoid_array(double *arr, size_t size);
    double sigmoid(double z);
    void sigmoid_prime_array(double *arr, size_t size);
    double sigmoid_prime(double z);
    void dot(double **, const std::vector<double> & t, double *tmpv, size_t x, size_t y);
    void shuffle(double **p_training_data_x, double **p_training_data_y, size_t len);
    void SGD(
        int epochs,
        int mini_batch_size,
        double eta
    );
    void update_mini_batch(double **p_training_data_x, double **p_training_data_y, size_t offset, int mini_batch_size, size_t len, double eta);
    void initwb(double *** & weights, double ** & bias);
    void fill0wb(double *** & _weights, double ** & _bias);
    void add_nabla_delta();
    void minus_wb(int batch_cnt, double eta);
    void backprop(double *x, double *y);
    int evaluate();
    void cost_derivative(double *pa, double *py, double *c_d, size_t size);
    void printwb();
    void printnwb();
};


#endif