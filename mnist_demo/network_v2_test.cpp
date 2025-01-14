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
    #ifndef WANGLITEST
    sizes.push_back(INPUT_LAYER_SIZE);
    sizes.push_back(30);
    sizes.push_back(10);
    #else
    sizes.push_back(INPUT_LAYER_SIZE);
    sizes.push_back(2);
    sizes.push_back(10);
    #endif
    NetWork mynet(sizes);
    MnistLoaderBase loader;
    loader.load();
    
    std::vector<TrainingData*> v_training_data;
    std::vector<TrainingData*> v_test_data;
    for (auto i = 0; i < TRAIN_IMAGES_NUM; ++ i) {
        TrainingData *p = new TrainingData(INPUT_LAYER_SIZE, loader.getTrainLabels()[i]);
        // cout << p->y << endl;
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
            p->x[j][0] = loader.getTrainImages()[i][j]*1./256;
        }
        v_training_data.emplace_back(p);
    }
    for (auto i = 0; i < TEST_IMAGES_NUM; ++ i) {
        int index = i + TRAIN_IMAGES_NUM;
        TrainingData *p = new TrainingData(INPUT_LAYER_SIZE, loader.getTrainLabels()[index]);
        //cout << p->y << endl;
        for (auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
            p->x[j][0] = loader.getTrainImages()[index][j]*1./256;
        }
        // for (auto j = 0; j < 10; ++ j) {
        //     cout << p->x[j][0] << " ";
        // }
        // cout << endl;
        // if (i > 0) {
        //     bool eq = true;
        //     for (auto j = 0; j < INPUT_LAYER_SIZE; ++ j) {
        //         if (p->x[j][0] != v_test_data[i-1]->x[j][0]) {
        //             eq = false;
        //         }
        //     }
        //     if (eq) {
        //         cout << "eq !" << endl;
        //     } else {
        //         // cout << "neq !" << endl;
        //     }
        // }
        v_test_data.emplace_back(p);
    }
    cout << "data loaded." << endl;

    assert(v_training_data.size() == TRAIN_IMAGES_NUM);
    assert(v_test_data.size() == TEST_IMAGES_NUM);
    #ifndef WANGLITEST
    mynet.SGD(v_training_data, v_test_data, 30, 30, 0.1);
    #else
    mynet.SGD(v_training_data, v_test_data, 2, 1, 0.1);
    #endif

    for (auto i = 0; i < v_training_data.size(); ++ i) {
        delete v_training_data[i];
    }
    for (auto i = 0; i < v_test_data.size(); ++ i) {
        delete v_test_data[i];
    }
    return 0;
}