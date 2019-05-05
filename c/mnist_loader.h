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
    double **training_data_x;
    double **training_data_y;
    double **test_data_x;
    int *test_data_y;
};
#endif