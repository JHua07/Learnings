#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
    // 沿Z轴转90度的旋转矩阵
    Matrix3d R = AngleAxisd(M_PI/2, Vector3d(0, 0, 1)).toRotationMatrix();

    //四元数
    Quaterniond q(R);

    Sophus::SO3d SO3_R(R);  // 通过旋转矩阵构造SO(3)
    Sophus::SO3d SO3_q(q);  // 通过四元数构造SO(3)

    // 二者是等价的
    cout << "SO(3) from matrix: \n" << SO3_R.matrix() << endl;
    cout << "SO(3) from quaternion: \n" << SO3_q.matrix() << endl;
    cout << "they are equal" << endl;

    // 使用对数映射获得它的李代数
    Vector3d so3_r = SO3_R.log();
    Vector3d so3_q = SO3_q.log();
    cout << "so3_r = " << so3_r.transpose() << endl;
    cout << "so3_q = " << so3_q.transpose() << endl;

    //hat 为向量到反对称矩阵
    cout << "so3_r hat = \n" << Sophus::SO3d::hat(so3_r) << endl;
    cout << "so3_q hat = \n" << Sophus::SO3d::hat(so3_q) << endl;

    // 相对的，vee为反对称到向量
    cout << "so3_r hat vee = " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3_r)).transpose() << endl;
    cout << "so3_q hat vee = " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3_q)).transpose() << endl;

    //增量扰动模型的更新,左乘更新
    Vector3d update_so3(0.01, 0.02, 0.03); //假设更新量为这么多
    Sophus::SO3d SO3_R_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
    Sophus::SO3d SO3_Q_updated = Sophus::SO3d::exp(update_so3) * SO3_q;
    

    cout << "SO3_R updated = \n" << SO3_R_updated.matrix() << endl;
    cout << "SO3_Q updated = \n" << SO3_Q_updated.matrix() << endl;

    return 0;
}