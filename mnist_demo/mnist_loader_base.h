#ifndef MNIST_LOADER_BASE_H
#define MNIST_LOADER_BASE_H

#include <vector>

class MnistLoaderBase {

public:
    MnistLoaderBase() {}
    ~MnistLoaderBase() {}
    const std::vector<std::vector<unsigned char>> & getTrainImages();
    const std::vector<unsigned char> & getTrainLabels();

    void load();

private:

    std::vector<std::vector<unsigned char>> train_images;
    std::vector<unsigned char> train_labels;


};

#endif