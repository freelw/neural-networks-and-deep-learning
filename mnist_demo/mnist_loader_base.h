#ifndef MNIST_LOADER_BASE_H
#define MNIST_LOADER_BASE_H

#include <vector>

class MnistLoaderBase {

public:
    MnistLoaderBase() {}
    ~MnistLoaderBase() {}
    const std::vector<std::vector<int>> & getTrainImages();
    const std::vector<int> & getTrainLabels();

private:

    std::vector<std::vector<int>> train_images;
    std::vector<int> train_labels;


};

#endif