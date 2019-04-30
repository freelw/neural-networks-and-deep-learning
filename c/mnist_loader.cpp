#include "mnist_loader.h"

#include <iostream>
#include <fstream>

void MnistLoader::load_data(const std::string & file_name) {
    std::ifstream ifs(file_name.c_str());
    ifs >> training_data_len >> training_data_x_len >> training_data_y_len;
    training_data_x = new float* [training_data_len];
    training_data_y = new float* [training_data_len];
    for (size_t i = 0; i < training_data_len; ++ i) {
        training_data_x[i] = new float [training_data_x_len];
        training_data_y[i] = new float [training_data_y_len];
    }
    for (size_t i = 0; i < training_data_len; ++ i) {
        for (size_t j = 0; j < training_data_x_len; ++ j) {
            ifs >> training_data_x[i][j];
        }
        for (size_t j = 0; j < training_data_y_len; ++ j) {
            ifs >> training_data_y[i][j];
        }
    }
    ifs >> test_data_len >> test_data_x_len;
    test_data_x = new float* [test_data_len];
    test_data_y = new int [test_data_len];
    for (size_t i = 0; i < test_data_len; ++ i) {
        test_data_x[i] = new float [test_data_x_len];
    }
    for (size_t i = 0; i < test_data_len; ++ i) {
        for (size_t j = 0; j < test_data_x_len; ++ j) {
            ifs >> test_data_x[i][j];
        }
        ifs >> test_data_y[i];
    }
}