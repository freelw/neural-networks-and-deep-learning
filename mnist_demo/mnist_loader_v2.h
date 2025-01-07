#ifndef MNIST_LOADER_V2_H
#define MNIST_LOADER_V2_H


#include <string>
#include "mnist_loader_base.h"

class MnistLoaderV2 {

public:
    MnistLoaderV2() {}
    ~MnistLoaderV2() {}

    void load_data();
public:
    int training_data_len;
    int training_data_x_len;
    int training_data_y_len;
    int test_data_len;
    int test_data_x_len;
    double **training_data_x;
    double **training_data_y;
    double **test_data_x;
    int *test_data_y;
    MnistLoaderBase base_loader;
};
#endif