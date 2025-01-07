#include "matrix.h"

#include <iostream>

using namespace std;

int main() {

    Matrix a(Shape(5,2));
    cout << a.zero() << endl;

    Matrix b(Shape(3,2));
    cout << b << endl;

    return 0;
}