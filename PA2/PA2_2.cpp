// Practice Two Question 2
// use Geometry
#include<iostream>
#include<Eigen/Core>
#include<Eigen/Dense>
#include<Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    Quaterniond q1(0.55, 0.3, 0.2, 0.2);
    Quaterniond q2(-0.1, 0.3, -0.7, 0.2);
    // Normalize the Quaternion before using
    q1.normalize();
    q2.normalize();
    Vector3d t1(0.7, 1.1, 0.2);
    Vector3d t2(-0.1, 0.4, 0.8);
    Isometry3d T_c1w = Isometry3d::Identity();
    Isometry3d T_c2w = Isometry3d::Identity();
    T_c1w.rotate( q1 );
    T_c1w.pretranslate(t1);
    T_c2w.rotate( q2 );
    T_c2w.pretranslate(t2);
    Vector3d p1_c1(0.5, -0.1, 0.2);
    Vector3d p1_c2;
    p1_c2 = T_c2w * T_c1w.inverse() * p1_c1;
    cout << p1_c2 << endl;
    return 0;
}