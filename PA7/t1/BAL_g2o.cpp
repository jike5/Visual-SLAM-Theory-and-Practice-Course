//
// Created by zzx on 2021/8/24.
//

#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include<iostream>

// common.h after include
#include "common.h"
#include "sophus/se3.hpp"

using namespace Eigen;
using namespace std;
using namespace Sophus;

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv){
    if(argc != 2){
        cout << "usage: bal_g2o bal_data.txt"  << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("initial.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}

// vertex and edges
class VertexPoint : public g2o::BaseVertex<3, Vector3d>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override {
        _estimate = Vector3d(0, 0, 0);
    }

    virtual void oplusImpl(const double *update) override{
        _estimate += Vector3d(update[0], update[1], update[2]);
    }

    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}
};


struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}

    explicit PoseAndIntrinsics(double * data_addr) {
        rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    // 将结构体内的值返回到data_addr中
    void set_to(double *data_addr) {
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) data_addr[i+3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    SO3d rotation;
    Vector3d translation = Vector3d::Zero();
    double focal = 0;
    double k1 = 0;
    double k2 = 0;
};


class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoseAndIntrinsics() {}

    virtual void setToOriginImpl() override {
        _estimate = PoseAndIntrinsics();
    }

    virtual void oplusImpl(const double *update) override {
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation = Vector3d(update[3], update[4], update[5]) + _estimate.translation;
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    Vector2d project(const Vector3d &point){
        Vector3d pc = _estimate.rotation * point + _estimate.translation;
        pc = -pc / pc[2];
        double r2 = pc.squaredNorm();
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
        return Vector2d(_estimate.focal * distortion * pc[0],
                        _estimate.focal * distortion * pc[1]);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {};
};


class EdgeProjection : public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void computeError() override {
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0];
        auto v1 = (VertexPoint *) _vertices[1];
        auto proj = v0->project(v1->estimate());
        _error = proj - _measurement;
    }

    // use numeric derivatives

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

};


void SolveBA(BALProblem &bal_problem){
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();
    const double *observations = bal_problem.observations();

    // PoseandIntrinsics in 9, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;
    // #include <g2o/solvers/csparse/linear_solver_csparse.h>
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer; 
    optimizer.setAlgorithm(solver); 
    optimizer.setVerbose(true); // 打开调试输出

    // 添加顶点
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
    vector<VertexPoint *> vertex_points;
    for(int i = 0; i < bal_problem.num_cameras(); ++i){
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics(); // 添加的是顶点的指针
        double *camera = cameras + camera_block_size * i;
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera));
        optimizer.addVertex(v);
        vertex_pose_intrinsics.push_back(v);
    }

    for(int i = 0; i < bal_problem.num_points(); ++i){
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Vector3d(point[0], point[1], point[2]));
        // g2o在BA中需要手动设置待Marg的顶点
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }
    // 添加边
    for(int i = 0; i < bal_problem.num_observations(); ++i){
        EdgeProjection *edge = new EdgeProjection();

        // setVertex(size_t i, Vertex *v) ith vertex on the edge，顶点的指针
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);
        edge->setMeasurement(Vector2d(observations[2*i + 0], observations[2*i + 1]));
        edge->setInformation(Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    // 开始优化
    optimizer.initializeOptimization();
    optimizer.optimize(40); // iterations = 40

    // 将结果返回到bal_problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i){
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate(); // estimate：PoseAndIntrinsics
        estimate.set_to(camera); // 调用结构体PoseAndIntrinsics的成员函数，将优化结果返回到camrea指针指向的位置
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
    }

}

/*
 // myVertexCamera
class VertexCamera : public g2o::BaseVertex<9, Eigen::Matrix3d>{
 public:
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

     VertexCamera() {}

     virtual void setToOriginImpl() override {
         _estimate << 0, 0, 0, 0, 0, 0, 0, 0, 0;
     }

     virtual void oplusImpl(const double *update) override {
         Vector3d R_log = Vector3d(_estimate(0, 0), _estimate(0, 1), _estimate(0, 2));
         SO3d R = SO3d(R_log);
         SO3d R_updated = SO3d::exp(Vector3d(update[0],update[1],update[2])) * R;
         Vector3d R_updated_log = R_updated.log();
         _estimate(0, 0) = R_updated_log(0);
         _estimate(0, 1) = R_updated_log(1);
         _estimate(0, 2) = R_updated_log(2);

         _estimate(1, 0) += update[3];
         _estimate(1, 1) += update[4];
         _estimate(1, 2) += update[5];

         _estimate(2, 0) += update[6];
         _estimate(2, 1) += update[7];
         _estimate(2, 2) += update[8];
     }

     Vector2d project(const Vector3d &point){
         Vector3d R_log = Vector3d(_estimate(0, 0), _estimate(0, 1), _estimate(0, 2));
         SO3d R = SO3d(R_log);
         Vector3d pc = R * point + Vector3d(_estimate(1,0), _estimate(1,1), _estimate(1, 2));
         pc = -pc / pc[2];
         double r2 = pc.squaredNorm();
         double distortion = 1.0 + r2 * (_estimate(2, 1) + _estimate(2, 2) * r2);
         return Vector2d(_estimate(2, 0) * distortion * pc[0],
                         _estimate(2, 0) * distortion * pc[1]);
     }

     Eigen::Matrix3d setMatrix(double * data_addr){
         Matrix3d M;
         M << data_addr[0], data_addr[1], data_addr[2],
                data_addr[3], data_addr[4], data_addr[5],
                data_addr[6], data_addr[7], data_addr[8];
         return M;
     }

     virtual bool read(istream &in) {}

     virtual bool write(ostream &out) const {};

 };

// myEdgeProjection
class EdgeProjection : public g2o::BaseBinaryEdge<2, Vector2d, VertexCamera, VertexPoint>{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void computeError() override {
        auto v0 = (VertexCamera *) _vertices[0];
        auto v1 = (VertexPoint *) _vertices[1];
        auto proj = v0->project(v1->estimate());
        _error = proj - _measurement;
    }

    // use numeric derivatives

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

};

 // mySolveBA
void SolveBA(BALProblem &bal_problem){
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();
    const double *observations = bal_problem.observations();

    // PoseandIntrinsics in 9, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;
    // #include <g2o/solvers/csparse/linear_solver_csparse.h>
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true); // 打开调试输出

    // 添加顶点
    vector<VertexCamera *> vertex_pose_intrinsics;
    vector<VertexPoint *> vertex_points;
    for(int i = 0; i < bal_problem.num_cameras(); ++i){
        VertexCamera *v = new VertexCamera(); // 添加的是顶点的指针
        double *camera = cameras + camera_block_size * i;
        v->setId(i);
        v->setEstimate(v->setMatrix(camera));
        optimizer.addVertex(v);
        vertex_pose_intrinsics.push_back(v);
    }

    for(int i = 0; i < bal_problem.num_points(); ++i){
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Vector3d(point[0], point[1], point[2]));
        // g2o在BA中需要手动设置待Marg的顶点
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }
    // 添加边
    for(int i = 0; i < bal_problem.num_observations(); ++i){
        EdgeProjection *edge = new EdgeProjection();

        // setVertex(size_t i, Vertex *v) ith vertex on the edge，顶点的指针
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);
        edge->setMeasurement(Vector2d(observations[2*i + 0], observations[2*i + 1]));
        edge->setInformation(Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    // 开始优化
    optimizer.initializeOptimization();
    optimizer.optimize(40); // iterations = 40

    // 将结果返回到bal_problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i){
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate(); // estimate：PoseAndIntrinsics
        estimate.set_to(camera); // 调用结构体PoseAndIntrinsics的成员函数，将优化结果返回到camrea指针指向的位置
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
    }

}

 */

