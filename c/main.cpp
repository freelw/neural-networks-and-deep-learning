#include "mnist_loader.h"

#include <iostream>

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
    return 0;
}