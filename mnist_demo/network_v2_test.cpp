#include "matrix.h"
#include "network_v2.h"
#include "mnist_loader_base.h"
#include <iostream>
#include <vector>
#include <assert.h>

#define INPUT_LAYER_SIZE 784
using namespace std;
int main(int argc, char *argv[])
{
    vector<int> sizes;
    sizes.push_back(INPUT_LAYER_SIZE);
    sizes.push_back(30);
    sizes.push_back(10);
    NetWork mynet(sizes);
    MnistLoaderBase loader;
    loader.load();
    
    std::vector<TrainingData*> v_training_data;
    std::vector<TrainingData*> v_test_data;
    for (auto i = 0; i < TRAIN_IMAGES_NUM; ++ i) {
        TrainingData *p = new TrainingData(INPUT_LAYER_SIZE, loader.getTrainLabels()[i]);
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
            p->x[j][0] = loader.getTrainImages()[i][j];
        }
        v_training_data.emplace_back(p);
    }
    for (auto i = 0; i < TEST_IMAGES_NUM; ++ i) {
        int index = i + TRAIN_IMAGES_NUM;
        TrainingData *p = new TrainingData(INPUT_LAYER_SIZE, loader.getTrainLabels()[index]);
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
            p->x[j][0] = loader.getTrainImages()[index][j];
        }
        v_test_data.emplace_back(p);   
    }

    assert(v_training_data.size() == TRAIN_IMAGES_NUM);
    assert(v_test_data.size() == TEST_IMAGES_NUM);
    mynet.SGD(v_training_data, v_test_data, 30, 10, 0.1);
    return 0;
}