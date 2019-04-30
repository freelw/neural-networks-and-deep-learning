#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H


#include <string>

class MnistLoader {

public:
    MnistLoader() {}
    ~MnistLoader() {}

    void load_data(const std::string & file_name);
public:
    int training_data_len;
    int training_data_x_len;
    int training_data_y_len;
    int test_data_len;
    int test_data_x_len;
    float **training_data_x;
    float **training_data_y;
    float **test_data_x;
    int *test_data_y;
};
#endif