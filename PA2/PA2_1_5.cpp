// PA2 第一题第5小问
// 实现随机矩阵100*100的A的QR分解和Cholesky分解求解Ax=b
#include<iostream>
#include<Eigen/Core>
#include<Eigen/Dense>

#define MATRIX_SIZE 100
using namespace std;
using namespace Eigen;

int main(int argc, char** argv)
{
    MatrixXd Apre = MatrixXd::Random(100, 100);
    // to make a symmetric matrix to mkaesure Cholesky Decomposition work
    MatrixXd A = Apre.transpose() * Apre;

    Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random( MATRIX_SIZE, 1);
    cout << "b(0) = " << v_Nd(0) << endl;
    // QR decomposition
    Eigen::Matrix<double, MATRIX_SIZE, 1> x1;
    x1 = A.colPivHouseholderQr().solve(v_Nd);
    // b1_ = A * x1
    Eigen::Matrix<double, MATRIX_SIZE, 1> b1_ = A *x1;
    cout << "b1_(0) = " << b1_(0) << endl;

    // Cholesky decomposition
    Eigen::Matrix<double, MATRIX_SIZE, 1> x2;
    x2 = A.llt().solve(v_Nd);
    // b2_ = A * x2
    Eigen::Matrix<double, MATRIX_SIZE, 1> b2_ = A * x2;
    cout << "b2_(0) = " << b2_(0) << endl;
    return 0;
}