#include "mnist_loader.h"
#include "mnist_loader_v2.h"
#include "network.h"
#include <iostream>
#include <vector>


using namespace std;
int main(int argc, char *argv[])
{
    // MnistLoader loader;
    // loader.load_data(argv[1]);
    MnistLoaderV2 loader;
    loader.load_data();
    /*cout << loader.training_data_len << endl;
    cout << loader.training_data_x_len << endl;
    cout << loader.training_data_y_len << endl;
    cout << loader.test_data_len << endl;
    cout << loader.test_data_x_len << endl;*/
    vector<int> sizes;
    sizes.push_back(784);
    sizes.push_back(30);
    sizes.push_back(10);
    NetWork<MnistLoaderV2> mynet(sizes, loader);
    //cout << "load done." << endl;
    mynet.SGD(30, 10, 3.0);
    return 0;
}