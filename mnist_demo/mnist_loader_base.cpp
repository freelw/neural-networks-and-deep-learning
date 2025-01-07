#include "mnist_loader_base.h"


const std::vector<std::vector<int>> & MnistLoaderBase::getTrainImages() {
    return train_images;
}

const std::vector<int> & MnistLoaderBase::getTrainLabels() {
    return train_labels;
}
