#include "mnist_loader.h"
#include "network.h"
#include <iostream>
#include <vector>

using namespace std;
int main(int argc, char *argv[])
{
    if (argc <= 1) {
        cout << "too few args" << endl;
        return -1;
    }
    MnistLoader loader;
    loader.load_data(argv[1]);
    cout << loader.training_data_len << endl;
    cout << loader.training_data_x_len << endl;
    cout << loader.training_data_y_len << endl;
    cout << loader.test_data_len << endl;
    cout << loader.test_data_x_len << endl;
    vector<int> sizes;
    sizes.push_back(784);
    sizes.push_back(30);
    sizes.push_back(10);
    NetWork mynet(sizes, loader);
    cout << "load done." << endl;
    mynet.SGD(30, 10, 3.0);
    return 0;
}