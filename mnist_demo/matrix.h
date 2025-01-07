#ifndef MATRIX_H
#define MATRIX_H

#include <ostream>
#include <vector>

using namespace std;

struct Shape {
    int rowCnt;
    int colCnt;
    Shape(int r, int c): rowCnt(r), colCnt(c) {}
};

class Matrix {

public:
    Matrix(Shape _shape)
        : initialized(false),
        shape(_shape) {}

    Matrix& zero();
    friend ostream &operator<<(ostream &output, const Matrix &m );

private:
    bool initialized;
    Shape shape;
    std::vector<std::vector<double>> data;

};

#endif