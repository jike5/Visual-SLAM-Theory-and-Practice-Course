#include <sophus/se3.hpp>
#include <string>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;
using namespace Eigen;
using namespace cv;

string trajectory_file = "./compare.txt";

void pose_estimation_3d3d(const vector<Point3f> &pts1,const vector<Point3f> &pts2, Eigen::Matrix3d &R_, Eigen::Vector3d &t_);
void DrawTrajectory(vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_e,
        vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_g,
        const string& ID);

int main(int argc, char **argv) {

    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_e;
    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_g;
    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_gt;
    vector<Point3f> pts_e,pts_g;
    ifstream fin(trajectory_file);
    if(!fin){
        cerr<<"can't find file at "<<trajectory_file<<endl;
        return 1;
    }
    while(!fin.eof()){
        double t1,tx1,ty1,tz1,qx1,qy1,qz1,qw1;
        double t2,tx2,ty2,tz2,qx2,qy2,qz2,qw2;
        fin>>t1>>tx1>>ty1>>tz1>>qx1>>qy1>>qz1>>qw1>>t2>>tx2>>ty2>>tz2>>qx2>>qy2>>qz2>>qw2;
        pts_e.push_back(Point3f(tx1,ty1,tz1));
        pts_g.push_back(Point3f(tx2,ty2,tz2));
        poses_e.push_back(Sophus::SE3d(Quaterniond(qw1,qx1,qy1,qz1),Vector3d(tx1,ty1,tz1)));
        poses_g.push_back(Sophus::SE3d(Quaterniond(qw2,qx2,qy2,qz2),Vector3d(tx2,ty2,tz2)));
    }

    Matrix3d R;
    Vector3d t;
    pose_estimation_3d3d(pts_e,pts_g,R,t);
    cout << "R = " << R << endl;
    cout <<"t = " << t << endl;
    Sophus::SE3d T_eg(R,t);
    for(auto SE_g:poses_g)    {
        Sophus::SE3d T_e=T_eg*SE_g;
        poses_gt.push_back(T_e);
    }
    DrawTrajectory(poses_e,poses_g," Before Align");
    DrawTrajectory(poses_e,poses_gt," After Align");
    return 0;
}

void pose_estimation_3d3d(const vector<Point3f> &pts1,
                          const vector<Point3f> &pts2,
                          Eigen::Matrix3d &R_, Eigen::Vector3d &t_) {
    Point3f p1, p2;     // center of mass
    int N = pts1.size();
    cout << "Number of The Points: " << N << endl;
    cout << "pts1[30] = " << pts1[30] << endl;
    for (int i = 0; i < N; i++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    cout << "p1 = " <<p1 << endl;
    cout << "p2 = " <<p2 << endl;
    p1 = Point3f(Vec3f(p1) / N);
    p2 = Point3f(Vec3f(p2) / N);
    cout << "p1 = " <<p1 << endl;
    cout << "p2 = " <<p2 << endl;
    vector<Point3f> q1(N), q2(N); // remove the center
    for (int i = 0; i < N; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout << "W=" << W << endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    cout << "U=" << U << endl;
    cout << "V=" << V << endl;

    R_ = U * (V.transpose());
    if (R_.determinant() < 0) {
        R_ = -R_;
    }
    t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);
}

void DrawTrajectory(vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_e,
        vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> poses_g,
        const string& ID) {
    if (poses_e.empty() || poses_g.empty()) {
        cerr << "Trajectory is empty!" << endl;
        return;
    }

    string windowtitle = "Trajectory Viewer" + ID;
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind(windowtitle, 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for (size_t i = 0; i < poses_e.size() - 1; i++) {
            glColor3f(1.0f, 0.0f, 0.0f);
            glBegin(GL_LINES);
            auto p1 = poses_e[i], p2 = poses_e[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        for (size_t i = 0; i < poses_g.size() - 1; i++) {
            glColor3f(0.0f, 0.0f, 1.0f);
            glBegin(GL_LINES);
            auto p1 = poses_g[i], p2 = poses_g[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}
