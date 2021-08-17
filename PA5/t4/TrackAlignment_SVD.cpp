// Creat by zhenxinzhu 20210816
// TrackAlignment using SVD method
#include<Eigen/Core>
#include<Eigen/Geometry>
#include<Eigen/Dense>
using namespace Eigen;
#include "sophus/se3.hpp"
#include<iostream>
#include<fstream>
#include<vector>
using namespace std;
#include<pangolin/pangolin.h>
#include<unistd.h> // usleep

void pose_estimation_3d3d_svd(const vector<Vector3d, Eigen::aligned_allocator<Vector3d>> &vec_pe,
                              const vector<Vector3d, Eigen::aligned_allocator<Vector3d>> &vec_pg,
                              Matrix3d &R, Vector3d &t);
void DrawTrajectory(vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> pose_e,
                    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> pose_g,
                    const string& ID);

int main(int argc, char ** argv)
{
    vector<double> time_e;
    vector<Vector3d, Eigen::aligned_allocator<Vector3d>> vec_te;
    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vec_Te;
    vector<double> time_g;
    vector<Vector3d, Eigen::aligned_allocator<Vector3d>> vec_tg;
    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vec_Tg;
    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vec_Tge;
    // Read The TXT File
    ifstream fin("./compare.txt");
    if(!fin){
        cerr << "file read error! " << endl;
    }
    
    while(!fin.eof())
    {
        double te, txe, tye, tze, qxe, qye, qze, qwe,
            tg, txg, tyg, tzg, qxg, qyg, qzg, qwg;
        fin >> te >> txe >> tye >> tze >> qxe >> qye >> qze >> qwe
            >> tg >> txg >> tyg >> tzg >> qxg >> qyg >> qzg >> qwg;
        time_e.push_back(te);
        time_g.push_back(tg);
        vec_te.push_back(Vector3d(txe, tye, tze));
        vec_tg.push_back(Vector3d(txg, tyg, tzg));
        vec_Te.push_back(Sophus::SE3d(Quaterniond(qwe,qxe,qye,qze), Vector3d(txe, tye, tze)));
        vec_Tg.push_back(Sophus::SE3d(Quaterniond(qwg,qxg,qyg,qzg), Vector3d(txg, tyg, tzg)));
    }
    fin.close();
    // Calculate Rotation Matrix R  & Translation t by using SVD
    Matrix3d R;
    Vector3d t;
    pose_estimation_3d3d_svd(vec_te, vec_tg, R, t);
    cout << R << endl;
    cout << t << endl;

    // transform pg to pe's coordinate
    Sophus::SE3d T_eg(R,t);
    for(auto SE_g : vec_Tg)
    {
        vec_Tge.push_back(T_eg * SE_g);
    }
    // Draw
    DrawTrajectory(vec_Te, vec_Tg, "Before Align");
    DrawTrajectory(vec_Te, vec_Tge, "After Align");
    return 0;
}

void pose_estimation_3d3d_svd(const vector<Vector3d, Eigen::aligned_allocator<Vector3d>> &vec_pe,
                              const vector<Vector3d, Eigen::aligned_allocator<Vector3d>> &vec_pg,
                              Matrix3d &R, Vector3d &t){
    double n = vec_pe.size();
    cout << "Number of The Points: " << n << endl;
    assert(n == vec_pg.size());
    cout << "vec_pe[30] = " << vec_pe[30] << endl;

    Vector3d center_e, center_g; // Center of the points
    for(int i = 0; i < n; i++)
    {
        center_e += vec_pe[i];
        center_g += vec_pg[i];
    }
    cout << "sum center_e = " << center_e << endl;
    cout << "sum center_g = " << center_g << endl;
    center_e = center_e / n;
    cout << "center_e = " <<center_e << endl;

    center_g = center_g / n;
    cout << "center_g = " <<center_g << endl;
    Matrix3d W = Matrix3d::Zero();
    for(int i = 0; i < n; i++)
    {
        W += (vec_pe[i] - center_e) * (vec_pg[i] - center_g).transpose();
    }
    cout << "W=" << W << endl;

    JacobiSVD<MatrixXd> svd(W, ComputeThinU | ComputeFullV);
    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();

    cout << "U=" << U << endl;
    cout << "V=" << V << endl;

    R = U * V.transpose();
    if(R.determinant() < 0)
        R = -R;
    cout << "determinant of R: " << R.determinant() << endl;
    t = center_e - R * center_g;
}

void DrawTrajectory(vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> pose_e,
                    vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> pose_g,
                    const string& ID){
    if(pose_e.empty() || pose_g.empty()){
        cerr << "Trajectory is empty!" << endl;
        return;
    }

    string windowtitle = "Trajectory Viewer" + ID;
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind(windowtitle, 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC0_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
            );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while(pangolin::ShouldQuit() == false){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glLineWidth(2);
        for(size_t i = 0; i < pose_e.size() - 1; i++){
            glColor3f(1.0f, 0.0f, 0.0f);
            glBegin(GL_LINES);
            auto p1 = pose_e[i], p2 = pose_e[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        for(size_t i = 0; i < pose_g.size() - 1; i++){
            glColor3f(0.0f, 1.0f, 0.0f);
            glBegin(GL_LINES);
            auto p1 = pose_g[i], p2 = pose_g[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000); // sleep 5ms
    }
}