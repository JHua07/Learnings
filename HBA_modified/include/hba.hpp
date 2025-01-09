#ifndef HBA_HPP
#define HBA_HPP

#include <iostream>
#include <thread>
#include <fstream>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "tools.hpp"
#include "layer.hpp"
#include "mypcl.hpp"
#include "voxel.hpp"

class HBA
{
public:
    int thread_num, total_layer_num;
    std::vector<LAYER> layers;
    std::string data_path;
    
    // 初始化图层状态, 主要作用是传入底层的位姿以及更新后续层的位姿
    HBA(int total_layer_num_, std::string data_path_, int thread_num_)
    {
        total_layer_num = total_layer_num_;
        data_path = data_path_;
        thread_num = thread_num_;

        // 更新层信息
        layers.resize(total_layer_num);
        for(int i = 0; i < total_layer_num; i++)
        {
            layers[i].layer_num = i + 1; //索引
            layers[i].thread_num = thread_num;
        }
        layers[0].data_path = data_path;
        layers[0].pose_vec = mypcl::read_pose(data_path + "pose.json");
        layers[0].init_layer_param();
        layers[0].init_storage(total_layer_num);

        // 初始化非底层的参数
        for(int i = 1; i < total_layer_num; i++)
        {
            // 上一层除开最后一个线程之外的窗口数量，这里相当于, 从第二层开始，每层的数量都是上一层的数量的1/WIN_SIZE倍
            int pose_size_ = (layers[i - 1].thread_num - 1) * layers[i - 1].part_length;

            // 加上最后一个线程的窗口数量
            pose_size_ += layers[i - 1].tail == 0 ? layers[i - 1].left_gap_num : (layers[i - 1].left_gap_num + 1);

            layers[i].data_path = layers[i - 1].data_path + "process1/";
            layers[i].init_layer_param(pose_size_);
            layers[i].init_storage(total_layer_num);
        }
        printf("HBA init layer done\n");
    }

    void update_next_layer_state(int cur_layer_num)
    {
        // 遍历当前层的所有线程
        for(int i = 0; i < layers[cur_layer_num].thread_num; i++)
        {
            if(i < layers[cur_layer_num].thread_num - 1) // 非最后一个线程
            {   
                // 遍历第i个线程的每个窗口
                for(int j = 0; j < layers[cur_layer_num].part_length; j++)
                {
                    int index = (i * layers[cur_layer_num].part_length + j) * GAP; // 当前层的第i个线程的第j个窗口的索引
                    
                    // 当前层每个窗口内的第一个点的索引作为下一层的位姿
                    layers[cur_layer_num + 1].pose_vec[i * layers[cur_layer_num].part_length + j] = layers[cur_layer_num].pose_vec[index];
                }
            }
            else
            {
                for(int j = 0; j < layers[cur_layer_num].j_upper; j++)
                {
                    int index = (i * layers[cur_layer_num].part_length + j) * GAP;
                    layers[cur_layer_num + 1].pose_vec[i * layers[cur_layer_num].part_length + j] = layers[cur_layer_num].pose_vec[index];
                }
            }
        }
    }

    /*
    理解：hessian是怎么来的
    首先，定义Hess是一个60*60的矩阵，分别存放了窗口内10个点的位姿的Hessian矩阵的上三角部分元素
    1. 在hba.cpp中，通过damping_iter()计算每一个线程的每个窗口的Hessian，
    1. 通过ba.hpp中的divide_thread()函数，将任务分配给每个线程，调用acc_evaluate2()计算每一个体素内的Hessian矩阵，
        就是60*60的,然后所有体素的残差，雅各布矩阵和Hessian矩阵累加得到总的Hessian矩阵，雅各布矩阵和残差。（因为Hessian具有线性累加性质，因为 Hessian 矩阵本质上描述的是二阶导数的累积特性）
    2. 
    */
    // 使用GTSAM进行位姿图优化
    void pose_graph_optimization()
    {
        // 初始化位姿和hessian矩阵
        std::vector<mypcl::pose> upper_pose, init_pose; // 上层位姿和初始位姿(q,t)
        upper_pose = layers[total_layer_num - 1].pose_vec; // 顶层位姿
        init_pose = layers[0].pose_vec;                    // 底层位姿
        std::vector<VEC(6)> upper_cov, init_cov;           // 上层和初始的协方差矩阵
        upper_cov = layers[total_layer_num - 1].hessians;
        init_cov = layers[0].hessians;

        // 初始化gtsam
        int cnt = 0;
        gtsam::Values initial;
        gtsam::NonlinearFactorGraph graph;
        gtsam::Vector Vector6(6); // 定义一个6维向量
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8; // 初始化6维向量

        // 先验噪声模型
        gtsam::noiseModel::Diagonal::shared_ptr priorModel = gtsam::noiseModel::Diagonal::Variances(Vector6);

        //插入底层的第一个位姿作为初始位姿
        initial.insert(0, gtsam::Pose3(gtsam::Rot3(init_pose[0].q.toRotationMatrix()), gtsam::Point3(init_pose[0].t)));

        // 添加先验因子
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(gtsam::Rot3(init_pose[0].q.toRotationMatrix()), gtsam::Point3(init_pose[0].t)), priorModel));

        // 遍历底层的位姿，添加因子到因子图中
        for(uint i = 0; i < init_pose.size(); i++)
        {
            // 逐个添加底层的位姿
            if(i > 0)
            {
                initial.insert(i, gtsam::Pose3(gtsam::Rot3(init_pose[i].q.toRotationMatrix()), gtsam::Point3(init_pose[i].t)));
            }
            
            // 添加每个窗口内第一个位姿的因子 i%GAP==0确保了每个窗口的第一个位姿
            if(i % GAP == 0 && cnt < init_cov.size())
            {
                // 遍历窗口中的所有hessian
                for(int j = 0; j < WIN_SIZE - 1; j++)
                {
                    for(int k = j + 1; k < WIN_SIZE; k++)
                    {
                        // 保证调用不越界
                        if(i + j + 1 >= init_pose.size() || i + k >= init_pose.size())
                        {
                            break;
                        }

                        cnt++;//这里的cnt是用来计数的，用来计算Hessian矩阵的数量，也就是窗口内两两之间的Hessian矩阵的数量，比如一个窗口内就是10个位姿，那么就有10*9/2个Hessian矩阵
                        
                        if(init_cov[cnt - 1].norm() < 1e-20)
                        {
                            continue; // 如果Hessian矩阵的范数小于1e-20（即Hessian过小），跳过
                        }

                        // 设置窗口内第j个位姿相对于第一个位姿的位移和旋转
                        Eigen::Vector3d t_ab = init_pose[i + j].t;
                        Eigen::Matrix3d R_ab = init_pose[i + j].q.toRotationMatrix();

                        // 更新为，窗口内第k个位姿相对于第j个位姿的位移和旋转, 需要这个矩阵主要是为了添加两个节点之间的约束
                        t_ab = R_ab.transpose() * (init_pose[i + k].t - t_ab);
                        R_ab = R_ab.transpose() * init_pose[i + k].q.toRotationMatrix();
                        
                        gtsam::Rot3 R_sam(R_ab);
                        gtsam::Point3 t_sam(t_ab);

                        // 根据位姿间的hessians设置里程计噪声模型
                        Vector6 << fabs(1.0 / init_cov[cnt - 1](0)), fabs(1.0 / init_cov[cnt - 1](1)), fabs(1.0 / init_cov[cnt - 1](2)),
                            fabs(1.0 / init_cov[cnt - 1](3)), fabs(1.0 / init_cov[cnt - 1](4)), fabs(1.0 / init_cov[cnt - 1](5));
                        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

                        // 添加里程计两两位姿间的factor
                        gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(i + j, i + k, gtsam::Pose3(R_sam, t_sam), odometryNoise));
                        graph.push_back(factor);
                    }
                }
            }

        }

        // 遍历顶层的位姿，添加因子到因子图中
        int pose_size = upper_pose.size();
        cnt = 0;
        
        // 不按窗口添加因子，而是直接两两之间添加因子
        for(int i = 0; i < pose_size - 1; i++)
        {
            for(int j = i + 1; j < pose_size; j++)
            {
                cnt++;
                if(upper_cov[cnt - 1].norm() < 1e-20)
                {
                    continue;
                }

                Eigen::Vector3d t_ab = upper_pose[i].t;
                Eigen::Matrix3d R_ab = upper_pose[i].q.toRotationMatrix();
                t_ab = R_ab.transpose() * (upper_pose[j].t - t_ab);
                R_ab = R_ab.transpose() * upper_pose[j].q.toRotationMatrix();
                gtsam::Rot3 R_sam(R_ab);
                gtsam::Point3 t_sam(t_ab);

                Vector6 << fabs(1.0 / upper_cov[cnt - 1](0)), fabs(1.0 / upper_cov[cnt - 1](1)), fabs(1.0 / upper_cov[cnt - 1](2)),
                    fabs(1.0 / upper_cov[cnt - 1](3)), fabs(1.0 / upper_cov[cnt - 1](4)), fabs(1.0 / upper_cov[cnt - 1](5));
                gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
                // 这里使用pow()函数是因为在上一层的时候，每个窗口内的位姿数量是下一层的GAP倍，所以这里需要乘以GAP的total_layer_num - 1次方，这里就是要追溯到底层的位姿索引
                gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(i * pow(GAP, total_layer_num - 1),
                                                                                                 j * pow(GAP, total_layer_num - 1), gtsam::Pose3(R_sam, t_sam), odometryNoise));
                graph.push_back(factor);
            }
        }
        // 此时已经完成了所有因子的添加，接下来就是使用gtsam进行优化
        gtsam::ISAM2Params parameters;
        // 重新线性化阈值,由于非线性函数的特性，随着优化的进行，误差函数可能会发生较大的变化，导致线性化的误差逐渐增大。为了避免这种误差积累影响优化过程，通常会在优化的过程中引入 重新线性化 的机制
        parameters.relinearizeThreshold = 0.01;
        // 重新线性化跳过,设置=1，确保每次迭代都会重新线性化
        parameters.relinearizeSkip = 1;
        gtsam::ISAM2 isam(parameters); // 创建ISAM2对象
        isam.update(graph, initial);   // 引入因子图和初始值
        isam.update();                 // 进行优化

        gtsam::Values results = isam.calculateEstimate(); // 计算估计值

        cout << "vertex size" << results.size() << endl;

        for(uint i = 0; i < results.size(); i++)
        {
            // 使用results中优化后的位姿来更新初始位姿
            gtsam::Pose3 pose = results.at(i).cast<gtsam::Pose3>();
            assign_qt(init_pose[i].q, init_pose[i].t, Eigen::Quaterniond(pose.rotation().toQuaternion()), pose.translation());
        }
        mypcl::write_pose(init_pose, data_path); // 保存优化后的位姿
        printf("pgo complete!\n");
    }

};

#endif