#include "network_v2.h"
#include "matrix.h"
#include <math.h>

#include <iostream>

double sigmoid_double(double z) {
    double ret = 1./(1.+exp(-z));
    // std::cout << "z : " << z << std::endl;
    // std::cout << "ret : " << ret << std::endl;
    return ret;
}

Matrix sigmoid(Matrix m) {
    Shape shape = m.getShape();
    Matrix res(m);
    for (auto i = 0; i < shape.rowCnt; ++i) {
        for (auto j = 0; j < shape.colCnt; ++j) {
            res[i][j] = sigmoid_double(res[i][j]);
        }
    }
    return res;
}
