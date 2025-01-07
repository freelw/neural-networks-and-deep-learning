#include "matrix.h"

#include <iostream>

using namespace std;

void test1() {
    //normal
    Matrix a(Shape(5,2));
    cout << a.zero() << endl;
    Matrix b(Shape(3,2));
    cout << b << endl;
}

void test2() {
    // operator +
    Matrix c(Shape(4,3)), d(Shape(4, 3));
    c.setAll(1);
    d.setAll(2);
    cout << c + d << endl;
}

void test3() {
    // dot
    Matrix x(Shape(3,2)), y(Shape(2, 3));
    x.zero();
    y.zero();
    x[0][0] = 1;
    x[1][0] = 2;
    x[2][0] = 3;
    x[0][1] = 6;
    x[1][1] = 6;
    x[2][1] = 6;
    y[0][0] = 4;
    y[0][1] = 5;
    y[0][2] = 6;
    y[1][0] = 7;
    y[1][1] = 7;
    y[1][2] = 7;
    cout << y.dot(x) << endl;
}

int main() {
    test1();
    test2();
    test3();
    return 0;
}