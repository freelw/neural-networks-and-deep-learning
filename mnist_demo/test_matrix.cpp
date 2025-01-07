#include "matrix.h"
#include "network_v2.h"
#include <iostream>

using namespace std;

void test1() {
    //normal
    cout << "test normal start ... " << endl;
    Matrix a(Shape(5,2));
    cout << a.zero() << endl;
    Matrix b(Shape(3,2));
    cout << b << endl;
    cout << "test normal end ... " << endl;
}

void test2() {
    // operator +
    cout << "test operator + start ... " << endl;
    Matrix c(Shape(4,3)), d(Shape(4, 3));
    c.setAll(1);
    d.setAll(2);
    cout << c + d << endl;
    cout << c + 3 << endl;
    cout << "test operator + end ... " << endl;
}

void test3() {
    // dot
    cout << "test dot start ... " << endl;
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
    cout << "test dot end ... " << endl;
}

void test4() {
    // operator +
    cout << "test operator - start ... " << endl;
    Matrix c(Shape(4,3));
    c.setAll(2);
    cout << c << endl;
    cout << c - 1 << endl;
    cout << -c << endl;
    cout << 1 - c << endl;
    cout << "test operator - end ... " << endl;
}

void test5() {
    // dot
    cout << "test sigmoid start ... " << endl;
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
    cout << -y.dot(x) << endl;
    cout << sigmoid(-y.dot(x)) << endl;
    cout << "test sigmoid end ... " << endl;
}

int main() {
    test1();
    test2();
    test3();
    test4();
    test5();
    return 0;
}