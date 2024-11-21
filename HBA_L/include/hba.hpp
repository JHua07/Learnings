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
#include "mypcl.hpp"
#include "ba.hpp"


class LAYER
{
public:
    /************变量定义*****************/
    int pose_size;          // 位姿数量
    int layer_num;          // 层索引
    int max_iter;           // 最大迭代次数
    int part_length;        // 除开最后一个线程，每个线程分配的任务量,也就是每个线程处理的窗口
    int left_size;          // 最后一个线程需要处理的所有位姿数量
    int left_h_size;        //
    int j_upper;            // 最后一个线程剩下的窗口移动次数
    int tail;               // 一层之内所有窗口按步长移动完成后，剩下的不足一次移动步长的位姿数量, 因为如果大于5，那么窗口就能再移动一次, 拿23个位姿举例，WIN_SIZE = 10, GAP = 5, thread_num = 2举例，tail == 3
    int thread_num;         // 线程数量
    int gap_num;            // 满窗口移动次数
    int last_win_size;      // 最后不足一个窗口大小的位姿数量，比如23个点，WIN_SIZE = 10 ,GAP = 5, 此时last_win_size = 8
    int left_gap_num;       // 最后一个线程剩余满窗口任务量(PS: 这里不要陷入误区，因为最后一个线程不一定能分到33个任务)，拿23个位姿举例，WIN_SIZE = 10, GAP = 5, thread_num = 2举例，满足窗口内10个点的移动一共只移动两次，此时最后还剩下3个点（移动后窗口内有8个点），那么就还需要移动一次，以遍历所有的点
    double downsample_size; // 降采样大小
    double voxel_size;      // 体素大小
    double eigen_ratio;     // 特征值比率
    double reject_ratio;    // 拒绝比率

    std::string data_path;          // 数据路径
    vector<mypcl::pose> pose_vec;   // 位姿向量
    std::vector<thread *> mthreads; // 线程向量
    std::vector<double> mem_costs;  // 内存消耗成本

    std::vector<VEC(6)> hessians;                      // Hessian矩阵, 定义6个6*1的向量
    std::vector<pcl::PointCloud<PointType>::Ptr> pcds; // 点云指针

    // 构造函数, 初始化变量
    LAYER()
    {
        pose_size = 0;
        layer_num = 1;
        max_iter = 10;
        downsample_size = 0.1;
        voxel_size = 4.0;
        eigen_ratio = 0.1;
        reject_ratio = 0.05;
        pose_vec.clear();
        mthreads.clear();
        pcds.clear();
        hessians.clear();
        mem_costs.clear();
    }

    // 初始化存储
    void init_storage(int total_layer_num_)
    {
        // 初始化线程和内存消耗
        mthreads.resize(thread_num);
        mem_costs.resize(thread_num);

        // 初始化点云和位姿
        pcds.resize(pose_size);
        pose_vec.resize(pose_size);

// 如果定义了FULL_HESS，那么初始化Hessian矩阵
#ifdef FULL_HESS
        if (layer_num < total_layer_num_) // 非顶层
        {
            //hessian_size的大小的计算 = 每个窗口内的位姿两两之间的都有一个hessian,那么每个窗口就是n*(n-1)/2
            //以23个pose，WIN_SIZE=10，GAP=5，thread_num=2来举例，那么hessian_size = 10*(10-1)/2 * 2 + 10*(10-1)/2 + 8*(8-1)/2 = 163
            int hessian_size = (thread_num - 1) * (WIN_SIZE - 1) * WIN_SIZE / 2 * part_length;
            hessian_size += (WIN_SIZE - 1) * WIN_SIZE / 2 * left_gap_num;
            if(tail > 0)
            {
                hessian_size += (last_win_size - 1) * last_win_size / 2;
            }
            hessians.resize(hessian_size);
            printf("hessian_size: %d\n", hessian_size);
        }
        else// 顶层
        {
            int hessian_size = pose_size * (pose_size - 1) / 2;
            hessians.resize(hessian_size);
            printf("hessian_size: %d\n", hessian_size);
        }

#endif
        for(int i = 0; i < thread_num; i++)
        {
            mem_costs.push_back(0);
        }
    }

    // 初始化参数
    void init_parameter(int pose_size_ = 0)
    {
        /***** 图片链接: ./home/jhua/ws_hba_learning/src/HBA_L/learnings/understand1.jpeg and understand2.jpeg ******/
        if (layer_num == 1) // 底层初始化
        {
            pose_size = pose_vec.size(); // 读取的初始位姿数量
        }
        else // 非底层初始化
        {
            pose_size = pose_size_; // 传入的
        }
        tail = (pose_size - WIN_SIZE) % GAP;
        gap_num = (pose_size - WIN_SIZE) / GAP;
        last_win_size = pose_size - GAP * (gap_num + 1);
        part_length = ceil((gap_num + 1) / double(thread_num));

        // 如果计算出的任务量超出了总任务量，则使用 floor 函数向下取整，重新计算每个线程的任务量
        if (gap_num - (thread_num - 1) * part_length < 0)
        {
            part_length = floor((gap_num + 1) / double(thread_num));
        }

        // 这个循环确保每个线程分配到的任务量是合理的。如果任务量为0或者分配不均匀，则减少线程数量并重新计算任务量。
        while (part_length == 0 || (gap_num - (thread_num - 1) * part_length + 1) / double(part_length) > 2)
        {
            thread_num -= 1;
            part_length = ceil((gap_num + 1) / double(thread_num));
            if (gap_num - (thread_num - 1) * part_length < 0)
            {
                part_length = floor((gap_num + 1) / double(thread_num));
            }
        }

        // 最后一个线程剩下的任务量
        left_gap_num = gap_num - (thread_num - 1) * part_length + 1;

        // 如果没有剩下的位姿，即刚好所有的位姿都被窗口移动完，且每个窗口中的位姿数量都是WIN_SIZE
        if (tail == 0)
        {
            // 拿20个位姿举例，WIN_SIZE = 10, GAP = 5, thread_num = 2举例，left_size = 10
            left_size = (gap_num - (thread_num - 1) * part_length + 1) * WIN_SIZE;
            // 最后一个线程需要处理的所有位姿数量减去 1 的结果，不知道用来干嘛
            left_h_size = (gap_num - (thread_num - 1) * part_length) * GAP + WIN_SIZE - 1;
            // 最后一个线程的窗口移动次数
            j_upper = gap_num - (thread_num - 1) * part_length + 1;
        }
        else
        {
            // 拿23个位姿举例，WIN_SIZE = 10, GAP = 5, thread_num = 2举例，left_size = 18，结合上面的tail == 0的情况，这个值代表的最后一个线程需要处理的所有位姿数量
            left_size = (gap_num - (thread_num - 1) * part_length + 1) * WIN_SIZE + GAP + tail;
            left_h_size = (gap_num - (thread_num - 1) * part_length + 1) * GAP + last_win_size - 1;
            j_upper = gap_num - (thread_num - 1) * part_length + 2;
        }
        printf("init parameter:\n");
        printf("layer_num %d | thread_num %d | pose_size %d | max_iter %d | part_length %d | gap_num %d | last_win_size %d | "
               "left_gap_num %d | tail %d | left_size %d | left_h_size %d | j_upper %d | "
               "downsample_size %f | voxel_size %f | eigen_ratio %f | reject_ratio %f\n",
               layer_num, thread_num, pose_size, max_iter, part_length, gap_num, last_win_size,
               left_gap_num, tail, left_size, left_h_size, j_upper,
               downsample_size, voxel_size, eigen_ratio, reject_ratio);
    }
};

class HBA
{
public:
    int thread_num, total_layer_num; // 线程数量, 总层数量
    std::vector<LAYER> layers;       // 层向量
    std::string data_path;           // 数据路径

    // HBA构造函数，初始化HBA的参数，主要包括层的各种属性
    HBA(int total_layer_num_, std::string data_path_, int thread_num_)
    {
        total_layer_num = total_layer_num_;
        data_path = data_path_;
        thread_num = thread_num_;

        layers.resize(total_layer_num);
        for (int i = 0; i < total_layer_num; i++)
        {
            layers[i].layer_num = i + 1;       // 更新层的索引从1开始
            layers[i].thread_num = thread_num; // 更新层的线程数量
        }
        // 初始化底层的属性
        layers[0].data_path = data_path;
        layers[0].pose_vec = mypcl::read_pose(data_path + "pose.json"); // 读取底层的初始位姿
        layers[0].init_parameter();                                     // 初始化底层参数
        layers[0].init_storage(total_layer_num);                        // 初始化底层存储

        // 初始化底层以上的属性
        for (int i = 1; i < total_layer_num; i++)
        {
            // 上一层最后一个线程之外的位姿数量
            int pose_size_ = (layers[i - 1].thread_num - 1) * layers[i - 1].part_length;

            // 若上一层窗口移动完成后刚好没有剩余位姿，即layers[i - 1].tail == 0, 此时pose_size_ + 0 ； 否则，pose_size_ + layers[i - 1].left_gap_num + 1
            pose_size_ += layers[i - 1].tail == 0 ? layers[i - 1].left_gap_num : (layers[i - 1].left_gap_num + 1);

            layers[i].init_parameter(pose_size_);
            layers[i].init_storage(total_layer_num);
            layers[i].data_path = layers[i - 1].data_path + "process1/";
        }
        printf("HBA init done!\n");
    }

    void update_next_layer_state(int cur_layer_num)
    {
        // 遍历当前层的所有线程
        for(int i = 0; i < layers[cur_layer_num].thread_num; i++)
        {
            if(i < layers[cur_layer_num].thread_num - 1)//非最后一个线程
            {
                //遍历第i个线程的每个窗口
                for(int j = 0; j < layers[cur_layer_num].part_length; j++)
                {
                    int index = (i * layers[cur_layer_num].part_length + j) * GAP;// 当前层的第i个线程的第j个窗口的索引
                    layers[cur_layer_num + 1].pose_vec[i * layers[cur_layer_num].part_length + j] = layers[cur_layer_num].pose_vec[index];//上一层的位姿数量是下一层的GAP倍，相当于按GAP进行降采样
                }
            }
            else // 最后一个线程
            {
                for (int j = 0; j < layers[cur_layer_num].j_upper; j++)
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
        // 初始化位姿和Hessian矩阵
        std::vector<mypcl::pose> upper_pose, init_pose; // 上层位姿和初始位姿(q,t)
        upper_pose = layers[total_layer_num - 1].pose_vec; // 顶层位姿
        init_pose = layers[0].pose_vec;                    // 底层位姿
        std::vector<VEC(6)> upper_cov, init_cov;           // 上层和初始的协方差矩阵
        upper_cov = layers[total_layer_num - 1].hessians;  // 顶层的Hessian矩阵
        init_cov = layers[0].hessians;                     // 底层的Hessian矩阵

        // 初始化gtsam
        int cnt = 0;
        gtsam::Values initial;//初始值
        gtsam::NonlinearFactorGraph graph;//定义一个非线性因子图
        gtsam::Vector Vector6(6);//定义一个6维向量
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8;//初始化6维向量
        gtsam::noiseModel::Diagonal::shared_ptr priorModel = gtsam::noiseModel::Diagonal::Variances(Vector6); // 先验噪声模型
        initial.insert(0, gtsam::Pose3(gtsam::Rot3(init_pose[0].q.toRotationMatrix()), gtsam::Point3(init_pose[0].t))); // 插入底层的第一个位姿作为初始位姿
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(gtsam::Rot3(init_pose[0].q.toRotationMatrix()), 
                    gtsam::Point3(init_pose[0].t)), priorModel)); // 添加先验因子，包括位姿和噪声模型

        // 遍历底层的位姿，添加因子到因子图中
        for(uint i = 0; i < init_pose.size(); i++)
        {
            if(i > 0)//添加位姿
            {
                initial.insert(i, gtsam::Pose3(gtsam::Rot3(init_pose[i].q.toRotationMatrix()), gtsam::Point3(init_pose[i].t)));
            }
            
            // 这里为什么能直接这么用，是因为在保存窗口内位姿的时候，已经将窗口内的位姿都转换到了第一个位姿的坐标系下，因此每一个位姿都是相对于第一个位姿的
            if(i % GAP == 0 && cnt < init_cov.size())// 逐窗口添加因子, 一个窗口10个位姿，那么每个窗口内的位姿两两之间都有一个Hessian矩阵
            {
                for(int j = 0; j < WIN_SIZE - 1; j++)
                {
                    for(int k = j + 1; k < WIN_SIZE; k++)
                    {
                        if(i + j + 1 >= init_pose.size() || i + k >= init_pose.size())
                        {
                            break; // 如果超出了位姿数量，跳出循环
                        }

                        cnt++;//这里的cnt是用来计数的，用来计算Hessian矩阵的数量，也就是窗口内两两之间的Hessian矩阵的数量，比如一个窗口内就是10个位姿，那么就有10*9/2个Hessian矩阵
                        if(init_cov[cnt - 1].norm() < 1e-20)
                        {
                            continue; // 如果Hessian矩阵的范数小于1e-20（即Hessian过小），跳过
                        }
                        
                        // 窗口内第j个位姿
                        Eigen::Vector3d t_ab = init_pose[i + j].t;
                        Eigen::Matrix3d R_ab = init_pose[i + j].q.toRotationMatrix();

                        // 更新为，窗口内第j个位姿相对于第k个位姿的位移和旋转，其中k = j + ?
                        t_ab = R_ab.transpose() * (init_pose[i + k].t - t_ab);           // 计算两个位姿之间的相对位移
                        R_ab = R_ab.transpose() * init_pose[i + k].q.toRotationMatrix(); // 计算两个位姿之间的相对旋转
                        gtsam::Rot3 R_sam(R_ab);
                        gtsam::Point3 t_sam(t_ab);

                        Vector6 << fabs(1.0 / init_cov[cnt - 1](0)), fabs(1.0 / init_cov[cnt - 1](1)), fabs(1.0 / init_cov[cnt - 1](2)),
                            fabs(1.0 / init_cov[cnt - 1](3)), fabs(1.0 / init_cov[cnt - 1](4)), fabs(1.0 / init_cov[cnt - 1](5));
                        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6); // 里程计噪声模型
                        gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(i + j, i + k, gtsam::Pose3(R_sam, t_sam), odometryNoise)); // 添加里程计因子

                        graph.push_back(factor);
                    }
                }
            }

        }

        // 遍历顶层的位姿，添加因子到因子图中
        int pose_size = upper_pose.size();
        cnt = 0;

        // 不按窗口添加因子，而是直接两两之间添加因子
        for (int i = 0; i < pose_size - 1; i++)
        {
            for (int j = i + 1; j < pose_size; j++)
            {
                cnt++;
                if (upper_cov[cnt - 1].norm() < 1e-20)
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

#endif // HBA_HPP
