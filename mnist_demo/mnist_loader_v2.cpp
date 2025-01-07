#include "mnist_loader_v2.h"

#include <iostream>
void MnistLoaderV2::load_data() {

    base_loader.load();
    std::vector<std::vector<unsigned char>> trainImages = base_loader.getTrainImages();
    std::vector<unsigned char> trainLabels = base_loader.getTrainLabels();
    training_data_len = TRAIN_IMAGES_NUM;
    training_data_x_len = trainImages[0].size();

    std::cout << "dbg training_data_x_len : " << training_data_x_len << std::endl;
    training_data_y_len = 10;
    test_data_len = TEST_IMAGES_NUM;
    test_data_x_len = training_data_x_len;


    training_data_x = new double* [training_data_len];
    training_data_y = new double* [training_data_len];
    for (size_t i = 0; i < training_data_len; ++ i) {
        training_data_x[i] = new double [training_data_x_len];
        training_data_y[i] = new double [training_data_y_len];
    }

    for (size_t i = 0; i < training_data_len; ++ i) {
        for (size_t j = 0; j < training_data_x_len; ++ j) {
            training_data_x[i][j] = trainImages[i][j]*1./256;
        }
        for (size_t j = 0; j < training_data_y_len; ++ j) {
            if (trainLabels[i] == (unsigned char)j) {
                training_data_y[i][j] = 1;
            } else {
                training_data_y[i][j] = 0;
            }
        }
    }

    test_data_x = new double* [test_data_len];
    test_data_y = new int [test_data_len];
    for (size_t i = 0; i < test_data_len; ++ i) {
        test_data_x[i] = new double [test_data_x_len];
    }
    for (size_t i = 0; i < test_data_len; ++ i) {
        for (size_t j = 0; j < test_data_x_len; ++ j) {
            test_data_x[i][j] = trainImages[i+TRAIN_IMAGES_NUM][j]*1./256;
        }
        for (size_t j = 0; j < training_data_y_len; ++ j) {
            test_data_y[i] = trainLabels[i+TRAIN_IMAGES_NUM];
        }
    }
}