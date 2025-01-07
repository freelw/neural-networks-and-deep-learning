#include "matrix.h"


Matrix& Matrix::zero() {
    if (!initialized) {
        data.clear();
        for (auto i = 0; i < shape.rowCnt; ++i) {
            std::vector<double> tmp;
            for (auto j = 0; j < shape.colCnt; ++j) {
                tmp.emplace_back(0);
            }
            data.emplace_back(tmp);
        }
    }
    initialized = true;
    return *this;
}

ostream &operator<<(ostream &output, const Matrix &m) {
    output << "[";
    for (auto i = 0; i < m.shape.rowCnt; ++ i) {
        if (i > 0) {
            output << " ";
        }
        output << "[";
        for (auto j = 0; j < m.shape.colCnt-1; ++ j) {
            output << m.data[i][j] << ", ";
        }
        output << m.data[i][m.shape.colCnt-1] << "]";
        if (i < m.shape.rowCnt-1) {
            output << endl;
        }
    }
    output << "]" << endl;
    return output;
}
