#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

class NetWork
{
private:
    int num_layers;
    std::vector<int> sizes;
    float **bias;
    float ***weights;
public:
    NetWork(const std::vector<int> &_sizes);
    ~NetWork();

};


#endif