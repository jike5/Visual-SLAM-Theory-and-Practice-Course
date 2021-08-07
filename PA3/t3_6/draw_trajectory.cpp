#include <sophus/se3.h>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/StdVector>

// need pangolin for plotting trajectory
#include <pangolin/pangolin.h>

using namespace std;

// path to trajectory file
string estimation_file = "../estimated.txt";
string groundtruth_file = "../groundtruth.txt";

// function for plotting trajectory, don't edit this code
// start point is red and end point is blue
void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>>);
vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> ReadTrajectory(const string &path);

int main(int argc, char **argv) {

    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses1; // estimation
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses2; // truth

    /// implement pose reading code
    poses1 = ReadTrajectory(estimation_file);
    poses2 = ReadTrajectory(groundtruth_file);
    
    // calculate RMSE
    double nRMSE = 0;
    double n = 0;
    for (size_t i = 0; i < poses1.size() - 1; i++) {
        auto p1 = poses1[i], p2 = poses2[i];
        auto temp = p1.inverse() * p2;
        Eigen::Matrix<double, 6, 1> e = temp.log();
        double e_norm = e.norm();
        nRMSE += e_norm * e_norm;
        n = (double)i + 1;
    }
    double RMSE = sqrt(nRMSE / n);
    cout << "RMSE = " << RMSE << endl;
    // draw trajectory in pangolin
    DrawTrajectory(poses1);
    DrawTrajectory(poses2);
    return 0;
}

/*******************************************************************************************/
vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> ReadTrajectory(const string &path)
{
    ifstream fin(path);
    vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> trajectory;
    if(!fin){
        cerr << "trajectory " << path << " not found " << endl;
        return trajectory;
    }
    while(!fin.eof()){
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Sophus::SE3 p1(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));
        trajectory.push_back(p1);
    }
    return trajectory;
}

void DrawTrajectory(vector<Sophus::SE3, Eigen::aligned_allocator<Sophus::SE3>> poses) {
    if (poses.empty()) {
        cerr << "Trajectory is empty!" << endl;
        return;
    }

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
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
        for (size_t i = 0; i < poses.size() - 1; i++) {
            glColor3f(1 - (float) i / poses.size(), 0.0f, (float) i / poses.size());
            glBegin(GL_LINES);
            auto p1 = poses[i], p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

}