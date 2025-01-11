#ifndef MNIST_LOADER_BASE_H
#define MNIST_LOADER_BASE_H

#include <vector>

//#define WANGLITEST
#ifdef WANGLITEST
#define EXPECTED_IMAGES_NUM 12
#define TRAIN_IMAGES_NUM 10
#define TEST_IMAGES_NUM 2
#else
#define EXPECTED_IMAGES_NUM 60000
#define TRAIN_IMAGES_NUM 50000
#define TEST_IMAGES_NUM 10000
#endif


class MnistLoaderBase {

public:
    MnistLoaderBase() {}
    ~MnistLoaderBase() {}
    const std::vector<std::vector<unsigned char>> & getTrainImages();
    const std::vector<unsigned char> & getTrainLabels();
    void load();

private:
    void load_images();
    void load_labels();

private:
    std::vector<std::vector<unsigned char>> train_images;
    std::vector<unsigned char> train_labels;
};

#endif