#include "matrix.h"
#include "network_v2.h"
#include "mnist_loader_base.h"
#include <iostream>
#include <vector>


using namespace std;
int main(int argc, char *argv[])
{
    vector<int> sizes;
    sizes.push_back(784);
    sizes.push_back(30);
    sizes.push_back(10);
    NetWork mynet(sizes);

    MnistLoaderBase loader;
    loader.load();
    // mynet.SGD(30, 10, 0.1);
    return 0;
}