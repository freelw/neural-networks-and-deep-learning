#include "mnist_loader_base.h"
#include <fstream>
#include <iostream>


#define RESOURCE_BASE "/workspaces/neural-networks-and-deep-learning/mnist_demo/resources/"
#define IMAGES_PATH RESOURCE_BASE"train-images-idx3-ubyte"
#define EXPECTED_IMAGES_NUM 60000
#define TRAIN_IMAGES_NUM 50000
#define TEST_IMAGES_NUM 10000

const std::vector<std::vector<unsigned char>> & MnistLoaderBase::getTrainImages() {
    return train_images;
}

const std::vector<unsigned char> & MnistLoaderBase::getTrainLabels() {
    return train_labels;
}

int reverse_char(int input) {
    unsigned char a, b, c, d;
    a = input & 0xff;
    b = (input >> 8) & 0xff;
    c = (input >> 16) & 0xff;
    d = (input >> 24) & 0xff;
    return (a << 24) | (b << 16) | (c << 8) | d;
}

void MnistLoaderBase::load() {
    std::ifstream images_ifs(IMAGES_PATH, std::ios::binary);
    std::string images_data = std::string(std::istreambuf_iterator<char>(images_ifs), std::istreambuf_iterator<char>());
    unsigned char * p = (unsigned char *)(images_data.c_str());
    int magic = reverse_char(*((int *)p));
    int images_num = reverse_char(*((int *)(p+4)));
    int rows_num = reverse_char(*((int *)(p+8)));
    int cols_num = reverse_char(*((int *)(p+12)));

    std::cout << "magic : " << magic << std::endl;
    std::cout << "images_num : " << images_num << std::endl;
    std::cout << "rows_num : " << rows_num << std::endl;
    std::cout << "cols_num : " << cols_num << std::endl;

    if (images_num != EXPECTED_IMAGES_NUM) {
        std::cerr << "images_num = " << images_num << " not equal to " << EXPECTED_IMAGES_NUM << std::endl;
        exit(-1);
    }
    train_images.reserve(EXPECTED_IMAGES_NUM);
    int pos = 16;
    for (auto i = 0; i < EXPECTED_IMAGES_NUM; ++ i) {
        std::vector<unsigned char> tmp;
        tmp.reserve(rows_num*cols_num);
        unsigned char * start = p + pos;
        tmp.emplace_back(start[i]);
        pos += rows_num*cols_num;
        train_images.emplace_back(tmp);
    }
}
