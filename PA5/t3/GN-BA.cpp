//
// Created by xiang on 12/21/17.
//

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "sophus/se3.hpp"

using namespace std;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

string p3d_file = "./p3d.txt";
string p2d_file = "./p2d.txt";

int main(int argc, char **argv) {

    VecVector2d p2d;
    VecVector3d p3d;
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // load points in to p3d and p2d 
    // START YOUR CODE HERE
    ifstream p3d_fin(p3d_file);
    ifstream p2d_fin(p2d_file);
    Vector3d p3d_input;
    Vector2d p2d_input;
    if(!p3d_fin || !p2d_fin) // !(is open)?
    {
        cerr << "file read error" << endl;
    }
    while(!p3d_fin.eof())
    {
        p3d_fin >> p3d_input(0) >> p3d_input(1) >> p3d_input(2);
        p3d.push_back(p3d_input);
    }
    p3d_fin.close();
    
    while(!p2d_fin.eof())
    {
        p2d_fin >> p2d_input(0) >> p2d_input(1);
        p2d.push_back(p2d_input);
    }
    p2d_fin.close();
    // END YOUR CODE HERE
    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();
    cout << "points: " << nPoints << endl;

    Sophus::SE3d T_esti; // estimated pose, default value is R = diag(1,1,1), t=(0,0,0)

    for (int iter = 0; iter < iterations; iter++) {

        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < nPoints; i++) {
            // compute cost for p3d[I] and p2d[I]
            // START YOUR CODE HERE 
            Vector3d pc = T_esti * p3d[i]; // SE3 product with v3d
            double inv_z = 1.0 / pc(2);
            double inv_z2 = inv_z * inv_z;
            Vector3d kpc = K * pc;
            Vector2d proj(kpc(0)/pc(2), kpc(1)/pc(2));

            Vector2d e = p2d[i] - proj;
            cost += e.squaredNorm();
	    // END YOUR CODE HERE

	    // compute jacobian
            Matrix<double, 2, 6> J;
            // START YOUR CODE HERE 
            J << -fx * inv_z,
                0,
                fx * pc(0) * inv_z2,
                fx * pc(0) * pc(1) * inv_z2,
                -(fx + fx * pc(0) * pc(0) * inv_z2),
                fx * pc(1) * inv_z,
                0,
                -fy * inv_z,
                fy * pc(1) * inv_z2,
                fy + fy * pc(1) * pc(1) * inv_z2,
                -fy * pc(0) * pc(1) * inv_z2,
                -fy * pc(0) * inv_z;
	    // END YOUR CODE HERE

            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

	// solve dx 
        Vector6d dx;

        // START YOUR CODE HERE 
        dx = H.ldlt().solve(b);
        // END YOUR CODE HERE

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        // START YOUR CODE HERE 
        T_esti = Sophus::SE3d::exp(dx) * T_esti;
        // END YOUR CODE HERE
        
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    return 0;
}
