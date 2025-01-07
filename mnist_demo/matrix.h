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

    Matrix(const Matrix &m);
    Matrix& zero();
    friend ostream &operator<<(ostream &output, const Matrix &m);
    Matrix operator+(const Matrix &m);
    Matrix operator+(int dt);
    Matrix operator-(int dt);
    Matrix operator-();
    friend Matrix operator-(int, const Matrix &m);
    std::vector<double>& operator[](unsigned int index);
    Matrix& setAll(double v);
    Shape getShape();
    Matrix dot(Matrix &m);

private:
    bool initialized;
    Shape shape;
    std::vector<std::vector<double>> data;

};

#endif