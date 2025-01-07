#include "mnist_loader.h"
#include "mnist_loader_v2.h"
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

#define EPS 0.000001

void check_eq_int(int a, int b) {
    if (a != b) {
        std::cerr << "check_eq_int failed " << a << " != " << b << std::endl;
    }
}

void check_eq_double(double a, double b) {
    if (fabs(a-b) > EPS) {
        std::cerr << "check_eq_double failed " << a << " != " << b << std::endl;
    }
}
void check_eq(MnistLoader &loader1, MnistLoaderV2 &loader2)
{
    check_eq_int(loader1.training_data_len, loader2.training_data_len);
    check_eq_int(loader1.training_data_x_len, loader2.training_data_x_len);
    check_eq_int(loader1.training_data_y_len, loader2.training_data_y_len);
    check_eq_int(loader1.test_data_len, loader2.test_data_len);
    check_eq_int(loader1.test_data_x_len, loader2.test_data_x_len);


     for (size_t i = 0; i < loader1.training_data_len; ++ i) {
        for (size_t j = 0; j < loader1.training_data_x_len; ++ j) {
            check_eq_double(loader1.training_data_x[i][j], loader2.training_data_x[i][j]);
        }
     }
}
int main(int argc, char *argv[])
{
    MnistLoaderV2 loader2;
    loader2.load_data();
    MnistLoader loader1;
    loader1.load_data();

    cout << "loader1..." << endl;
    cout << loader1.training_data_len << endl;
    cout << loader1.training_data_x_len << endl;
    cout << loader1.training_data_y_len << endl;
    cout << loader1.test_data_len << endl;
    cout << loader1.test_data_x_len << endl;

    cout << "loader2..." << endl;
    cout << loader2.training_data_len << endl;
    cout << loader2.training_data_x_len << endl;
    cout << loader2.training_data_y_len << endl;
    cout << loader2.test_data_len << endl;
    cout << loader2.test_data_x_len << endl;

    check_eq(loader1, loader2);

    cout << "checked. " << endl;
    return 0;
}