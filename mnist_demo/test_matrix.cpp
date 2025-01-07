#include "matrix.h"

#include <iostream>

using namespace std;

int main() {

    Matrix a(Shape(5,2));
    cout << a.zero() << endl;

    Matrix b(Shape(3,2));
    cout << b << endl;

    Matrix c(Shape(4,3)), d(Shape(4, 3));

    c.setAll(1);
    d.setAll(2);

    cout << c + d << endl;

    return 0;
}