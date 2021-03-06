//
// Created by xiang on 1/4/18.
// this program shows how to perform direct bundle adjustment
//
#include <iostream>

using namespace std;

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>


#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

#include <pangolin/pangolin.h>
#include <boost/format.hpp>

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> VecSE3;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVec3d;

// global variables
string pose_file = "./poses.txt";
string points_file = "./points.txt";

// intrinsics
float fx = 277.34;
float fy = 291.402;
float cx = 312.234;
float cy = 239.777;

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

// g2o vertex that use sophus::SE3 as pose
class VertexSophus : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSophus() {}

    ~VertexSophus() {}

    bool read(std::istream &is) {}

    bool write(std::ostream &os) const {}

    virtual void setToOriginImpl() {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update_) {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
        setEstimate(Sophus::SE3d::exp(update) * estimate());
    }
};

// TODO edge of projection error, implement it
// 16x1 error, which is the errors in patch
typedef Eigen::Matrix<double,16,1> Vector16d;
class EdgeDirectProjection : public g2o::BaseBinaryEdge<16, Vector16d, g2o::VertexSBAPointXYZ, VertexSophus> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeDirectProjection(float *color, cv::Mat &target) {
        this->origColor = color;
        this->targetImg = target;
    }

    ~EdgeDirectProjection() {}

    virtual void computeError() override {
        // TODO START YOUR CODE HERE
        // compute projection error ...
        auto v0 = (g2o::VertexSBAPointXYZ *) _vertices[0];
        auto v1 = (VertexSophus *) _vertices[1];
        Eigen::Vector3d pc = v1->estimate() * v0->estimate();
        double px = pc[0] / pc[2];
        double py = pc[1] / pc[2];
        cv::Point2d pro = cv::Point2d(fx*px + cx, fy*py + cy);
        Vector16d CurColor, OriColor;
        if (pro.x - 2 < 0 || pro.y - 2 < 0 || pro.x + 1 > targetImg.cols || pro.y + 1 > targetImg.rows){
            _error = Vector16d::Zero();
            ifOut = true;
        }
        else {
            int i = 0;
            for(int du = -2; du < 2; ++du){
                for(int dv = -2; dv < 2; ++dv){
                    CurColor[i] =  GetPixelValue(targetImg, pro.x + du, pro.y + dv);
                    OriColor[i] = origColor[i];
                    ++i;
                }
            }
            _error = OriColor -CurColor;
        }
        // END YOUR CODE HERE
    }

    // Let g2o compute jacobian for you
    // my jacobin
/*
    virtual void linearizeOplus() override {
        if(ifOut){
            _jacobianOplusXi = Eigen::Matrix<double, 16, 3>::Zero();
            _jacobianOplusXj = Eigen::Matrix<double, 16, 6>::Zero();
        }
        else{
            auto v0 = (g2o::VertexSBAPointXYZ *) _vertices[0];
            auto v1 = (VertexSophus *) _vertices[1];
            Eigen::Vector3d P = v0->estimate();
            Eigen::Vector3d Pc = v1->estimate() * P;
            Eigen::Matrix<double, 1, 2> J_I_u;
            Eigen::Matrix<double, 2, 3> J_u_q;
            Eigen::Matrix<double, 2, 6> J_u_ksi;

            float X = Pc[0], Y = Pc[1], Z = Pc[2];
            float Z_inv = 1.0 / Z, Z2 = Z * Z;
            float Z2_inv = Z_inv * Z_inv;

            double x = fx * X * Z_inv + cx;
            double y = fy * Y * Z_inv + cy;

            J_u_q << fx * Z_inv,
                    0,
                    -fx * X * Z2_inv,
                    0,
                    fy * Z_inv,
                    -fy * Y * Z2_inv;
            J_u_ksi << fx * Z_inv,
                        0,
                        -fx * X * Z2_inv,
                        -fx * X * Y * Z2_inv,
                        fx + fx * X * X * Z2_inv,
                        -fx * Y * Z_inv,
                        0,
                        fy * Z_inv,
                        -fy * Y * Z2_inv,
                        -fy - fy * Y * Y * Z2_inv,
                        fy * X * Y * Z2_inv,
                        fy * X * Z_inv;
            int i = 0;
            for(int du = -2; du < 2; ++du){
                for(int dv = -2; dv < 2; ++dv){
                    J_I_u[0] = 0.5 * (GetPixelValue(targetImg, du + x - 1, dv + y) -
                            GetPixelValue(targetImg, du + x + 1, dv + y));
                    J_I_u[1] = 0.5 * (GetPixelValue(targetImg, du + x, dv + y - 1) -
                            GetPixelValue(targetImg, du + x, dv + y + 1));
                    _jacobianOplusXi.block<1, 3>(i, 0) = -J_I_u * J_u_q * v1->estimate().rotationMatrix();
                    _jacobianOplusXj.block<1, 6>(i, 0) = -J_I_u * J_u_ksi;
                    ++i;
                }
            }
        }
    }
*/

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

private:
    cv::Mat targetImg;  // the target image
    float *origColor = nullptr;   // 16 floats, the color of this point
    bool ifOut = false; // point3D_project out of the image
};

// plot the poses and points for you, need pangolin
void Draw(const VecSE3 &poses, const VecVec3d &points);

int main(int argc, char **argv) {
    // read poses and points
    VecSE3 poses;
    VecVec3d points;
    ifstream fin(pose_file);

    while (!fin.eof()) {
        double timestamp = 0;
        fin >> timestamp;
        if (timestamp == 0) break;
        double data[7];
        for (auto &d: data) fin >> d;
        poses.push_back(Sophus::SE3d(
                Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                Eigen::Vector3d(data[0], data[1], data[2])
        ));
        if (!fin.good()) break;
    }
    fin.close();

    vector<float *> color;
    fin.open(points_file);
    while (!fin.eof()) {
        double xyz[3] = {0};
        for (int i = 0; i < 3; i++) fin >> xyz[i];
        if (xyz[0] == 0) break;
        points.push_back(Eigen::Vector3d(xyz[0], xyz[1], xyz[2]));
        float *c = new float[16];
        for (int i = 0; i < 16; i++) fin >> c[i];
        color.push_back(c);

        if (fin.good() == false) break;
    }
    fin.close();

    cout << "poses: " << poses.size() << ", points: " << points.size() << endl;

    // read images
    vector<cv::Mat> images;
    boost::format fmt("./%d.png");
    for (int i = 0; i < 7; i++) {
        images.push_back(cv::imread((fmt % i).str(), 0));
    }

    // build optimization problem
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
            );

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // TODO add vertices, edges into the graph optimizer
    // START YOUR CODE HERE
    vector<VertexSophus *> vector_VertexSophus;
    vector<g2o::VertexSBAPointXYZ *> vector_VertexSBAPointXYZ;
    // add vertex
    for (int i = 0; i < points.size(); ++i){
        g2o::VertexSBAPointXYZ *v = new g2o::VertexSBAPointXYZ;
        v->setId(i);
        v->setEstimate(points[i]);
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vector_VertexSBAPointXYZ.push_back(v);
    }
    for (int i = 0; i < poses.size(); ++i){
        VertexSophus *v = new VertexSophus;
        v->setId(i + points.size());
        v->setEstimate(poses[i]);
        optimizer.addVertex(v);
        vector_VertexSophus.push_back(v);
    }
    // add edge
    for (int i = 0; i < poses.size(); ++i){
        for (int j = 0; j < points.size(); ++j){
            EdgeDirectProjection * edge = new EdgeDirectProjection(color[j], images[i]);
            edge->setVertex(0, vector_VertexSBAPointXYZ[j]);
            edge->setVertex(1, vector_VertexSophus[i]);
            edge->setInformation(Eigen::Matrix<double, 16, 16>::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber());
            optimizer.addEdge(edge);
        }
    }
    // END YOUR CODE HERE

    // perform optimization
    optimizer.initializeOptimization(0);
    optimizer.optimize(200);

    // TODO fetch data from the optimizer
    // START YOUR CODE HERE
    for(int i = 0; i < points.size(); ++i){
        auto vertex = vector_VertexSBAPointXYZ[i];
        auto estimate = vertex->estimate();
        points[i] = estimate;
    }
    for(int j = 0; j < poses.size(); ++j){
        auto vertex = vector_VertexSophus[j];
        auto estimate = vertex->estimate();
        poses[j] = estimate;
    }
    // END YOUR CODE HERE

    // plot the optimized points and poses
    std::cout << "Start Draw ..." << endl;
    Draw(poses, points);

    // delete color data
    for (auto &c: color) delete[] c;
    return 0;
}

void Draw(const VecSE3 &poses, const VecVec3d &points) {
    if (poses.empty() || points.empty()) {
        cerr << "parameter is empty!" << endl;
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
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.1;
        int width = 640, height = 480;
        for (auto &Tcw: poses) {
            glPushMatrix();
            Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
            glMultMatrixf((GLfloat *) m.data());
            glColor3f(1, 0, 0);
            glLineWidth(2);
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(0, 0, 0);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
            glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
            glEnd();
            glPopMatrix();
        }

        // points
        glPointSize(2);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < points.size(); i++) {
            glColor3f(0.0, points[i][2]/4, 1.0-points[i][2]/4);
            glVertex3d(points[i][0], points[i][1], points[i][2]);
        }
        glEnd();

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}

