#include "mnist_loader.h"
#include "mnist_loader_v2.h"
#include <iostream>
#include <vector>
using namespace std;
int main(int argc, char *argv[])
{
    MnistLoader loader1;
    loader1.load_data();
    MnistLoaderV2 loader2;
    loader2.load_data();
    cout << loader1.training_data_len << endl;
    cout << loader1.training_data_x_len << endl;
    cout << loader1.training_data_y_len << endl;
    cout << loader1.test_data_len << endl;
    cout << loader1.test_data_x_len << endl;
    return 0;
}