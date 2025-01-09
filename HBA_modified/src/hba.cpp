#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include <mutex>
#include <assert.h>
#include <ros/ros.h>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <tf/transform_broadcaster.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include "hba.hpp"
#include "tools.hpp"
#include "voxel.hpp"
#include "layer.hpp"
#include "mypcl.hpp"

using namespace std;
using namespace Eigen;

int pcd_name_fill_num = 0;

// 体素树根节点
void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> &feat_map, pcl::PointCloud<PointType> &feat_pt, Eigen::Quaterniond q, Eigen::Vector3d t, int fnum, double voxel_size, int window_size, float eigen_ratio)
{
    float loc_xyz[3];
    for(PointType &p_c : feat_pt.points)
    {
        Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
        Eigen::Vector3d pvec_tran = q * pvec_orig + t;

        for(int j = 0; j < 3; j++)
        {
            loc_xyz[j] = pvec_tran[j] / voxel_size;
            if(loc_xyz[j] < 0)
            {
                loc_xyz[j] -= 1.0;
            }
        }

        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);

        if(iter != feat_map.end())
        {
            iter->second->vec_orig[fnum].push_back(pvec_orig);
            iter->second->vec_tran[fnum].push_back(pvec_tran);

            iter->second->sig_orig[fnum].push(pvec_orig);
            iter->second->sig_tran[fnum].push(pvec_tran);
        }
        else // 如果未找到，创建一个新的体素树根节点，并将其添加到feat_map中
        {
            OCTO_TREE_ROOT *ot = new OCTO_TREE_ROOT(window_size, eigen_ratio);
            ot->vec_orig[fnum].push_back(pvec_orig);
            ot->vec_tran[fnum].push_back(pvec_tran);
            ot->sig_orig[fnum].push(pvec_orig);
            ot->sig_tran[fnum].push(pvec_tran);

            ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
            ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
            ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
            ot->quater_length = voxel_size / 4.0;
            ot->layer = 0;
            feat_map[position] = ot;
        }
    }
}


void parallel_compute_tool(LAYER &layer, int thread_id, LAYER &next_layer, int i, int win_size)
{
    // raw_pc为当前窗口原始的点云，src_pc为降采样+转换+合并后的点云
    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(win_size);
    raw_pc.resize(win_size);

    double residual_cur = 0, residual_pre = 0; // 当前残差，上一次残差
    vector<IMUST> x_buf(win_size);             // 状态量

    // 计算每个窗口内的位姿
    for (int j = 0; j < win_size; j++)
    {
        x_buf[j].R = layer.pose_vec[i * GAP + j].q.toRotationMatrix();
        x_buf[j].p = layer.pose_vec[i * GAP + j].t;
    }

    size_t mem_cost = 0;
    for (int loop = 0; loop < layer.max_iter; loop++)
    {
        if (layer.layer_num == 1)
        {
            // 从路径中读取点云pcd文件, 逐窗口读取
            for (int j = i * GAP; j < i * GAP + win_size; j++)
            {
                if (loop == 0)
                {
                    pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>); // 创建点云指针
                    mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
                    raw_pc[j - i * GAP] = pc;
                }
                src_pc[j - i * GAP] = (*raw_pc[j - i * GAP]).makeShared();
            }
        }
        else
        {
            for (int j = i * GAP; j < i * GAP + win_size; j++)
            {
                src_pc[j - i * GAP] = (*layer.pcds[j]).makeShared();
            }
        }

        // 建立体素地图
        unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

        // 对该窗口进行体素降采样并划分这个窗口的根节点体素
        for (size_t j = 0; j < win_size; j++)
        {
            if (layer.downsample_size > 0)
            {
                downsample_voxel(*src_pc[j], layer.downsample_size);
            }
            cut_voxel(surf_map, *src_pc[j], Eigen::Quaterniond(x_buf[j].R), x_buf[j].p, j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio);
        }

        // 继续划分体素，确定体素的特征，完成体素地图的建立
        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        {
            iter->second->recut();
        }

        // 进行优化过程
        VOX_HESS voxhess(win_size);
        for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        {
            // 向体素容器中添加体素的特征、点云数据(ifdef ENABLE_FILTER)
            iter->second->tras_opt(voxhess);
        }

        VOX_OPTIMIZER opt_lsv(win_size);
        // 去除该窗口内的离群点，一个窗口内去除最多ratio比例的离群点
        opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
        PLV(6)
        hess_vec;

        // 窗口内进行阻尼迭代
        opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        {
            // 释放体素地图的内存，其实是只删除了second，也就是octo_tree_root的部分

            delete iter->second;
        }

        // 更新hess和内存占用量
        if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1)
        {
            // 将内存占用更新为当前的最大值
            if (layer.mem_costs[thread_id] < mem_cost)
            {
                layer.mem_costs[thread_id] = mem_cost;
            }

            // 更新hessians, 一个窗口内有WIN_SIZE * (WIN_SIZE - 1) / 2个hessian矩阵
            for (int j = 0; j < win_size * (win_size - 1) / 2; j++)
            {
                // 存储hessian矩阵
                layer.hessians[i * (win_size - 1) * win_size / 2 + j] = hess_vec[j];
            }
            break;
        }
        residual_pre = residual_cur; // 更新上一次的残差
    }

    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>); // 一个窗口的总点云
    for (size_t j = 0; j < win_size; j++)
    {
        Eigen::Quaterniond q_tmp;
        Eigen::Vector3d t_tmp;
        // 将x_buf中的R和t，先转到窗口内的第一个位姿的坐标系下后，赋值给q_tmp和t_tmp, 这里的.inverse()等价于.transpose()
        assign_qt(q_tmp, t_tmp, Eigen::Quaterniond(x_buf[0].R.inverse() * x_buf[j].R), x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

        // 一个位姿对应的点云
        pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);

        // 将点云src_pc按照q_tmp和t_tmp进行变换,也就是把每个窗口的每个位姿的点云转换到第一个位姿的坐标系下，结果存储在pc_oneframe中
        mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
        pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe); // 将每一个位姿对应的点云合并到窗口内的点云
    }
    downsample_voxel(*pc_keyframe, 0.05); // 体素降采样
    // 这个窗口内的点云赋值给下一层的pcd，直观来讲，就是将这个窗口的所有点云集中到第一个位姿的点云中，因为下一层只取这层的每个窗口的第一个位姿
    next_layer.pcds[i] = pc_keyframe;
}

// 执行某层非最后一个线程的并行计算
void parallel_head(LAYER &layer, int thread_id, LAYER &next_layer)
{
    int &part_length = layer.part_length;
    int &layer_num = layer.layer_num;

    // 处理当前层每一个线程的任务，也就是处理每个窗口
    for (int i = thread_id * part_length; i < (thread_id + 1) * part_length; i++)
    {
        parallel_compute_tool(layer, thread_id, next_layer, i, WIN_SIZE);
    }
}

// 执行最后一个线程的并行计算
void parallel_tail(LAYER &layer, int thread_id, LAYER &next_layer)
{
    int &part_length = layer.part_length;
    int &layer_num = layer.layer_num;
    int &left_gap_num = layer.left_gap_num;

    // 记录各种事件的时间
    // double load_t = 0, undis_t = 0, dsp_t = 0, cut_t = 0, recut_t = 0, total_t = 0, tran_t = 0, sol_t = 0, save_t = 0;

    if (layer.gap_num - (layer.thread_num - 1) * part_length + 1 != left_gap_num)
    {
        printf("This layer's left_gap_num is wrong!\n");
    }

    // 处理最后一个线程的满窗口
    for (uint i = thread_id * part_length; i < thread_id * part_length + left_gap_num; i++)
    {
        parallel_compute_tool(layer, thread_id, next_layer, i, WIN_SIZE);
    }

    if (layer.tail > 0)
    {
        int i = thread_id * part_length + left_gap_num;
        parallel_compute_tool(layer, thread_id, next_layer, i, layer.last_win_size);
    }
}

void global_ba(LAYER &layer)
{
    int window_size = layer.pose_vec.size();
    vector<IMUST> x_buf(window_size);
    for (int i = 0; i < window_size; i++)
    {
        x_buf[i].R = layer.pose_vec[i].q.toRotationMatrix();
        x_buf[i].p = layer.pose_vec[i].t;
    }

    vector<pcl::PointCloud<PointType>::Ptr> src_pc;
    src_pc.resize(window_size);
    for (int i = 0; i < window_size; i++)
    {
        src_pc[i] = (*layer.pcds[i]).makeShared();
    }

    double residual_cur = 0, residual_pre = 0;
    size_t mem_cost = 0, max_mem = 0;

    std::cout << "---------------------" << std::endl;
    std::cout << "Global BA Iteration Start:" << std::endl;
    for (int loop = 0; loop < layer.max_iter; loop++)
    {
        std::cout << "---------------------" << std::endl;
        std::cout << "Iteration " << loop << std::endl;

        unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

        for (int i = 0; i < window_size; i++)
        {
            if (layer.downsample_size > 0)
            {
                downsample_voxel(*src_pc[i], layer.downsample_size);
            }
            cut_voxel(surf_map, *src_pc[i], Eigen::Quaterniond(x_buf[i].R), x_buf[i].p, i, layer.voxel_size, window_size, layer.eigen_ratio);
        }

        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        {
            iter->second->recut();
        }

        VOX_HESS voxhess(window_size);
        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        {
            iter->second->tras_opt(voxhess);
        }

        VOX_OPTIMIZER opt_lsv(window_size);
        opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
        PLV(6)
        hess_vec;
        opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        {
            delete iter->second;
        }

        cout << "Residual absolute: " << abs(residual_pre - residual_cur) << " | "
             << "percentage: " << abs(residual_pre - residual_cur) / abs(residual_cur) << endl;

        if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1)
        {
            if (mem_cost > max_mem)
            {
                max_mem = mem_cost;
            }
#ifdef FULL_HESS
            for (int i = 0; i < window_size * (window_size - 1) / 2; i++)
            {
                layer.hessians[i] = hess_vec[i];
            }
#else
            for (int i = 0; i < window_size - 1; i++)
            {
                Matrix6d hess = Hess_cur.block(6 * i, 6 * i + 6, 6, 6);
                for (int row = 0; row < 6; row++)
                {
                    for (int col = 0; col < 6; col++)
                    {
                        hessFile << hess(row, col) << ((row * col == 25) ? "" : " ");
                    }
                }
                if (i < window_size - 2)
                {
                    hessFile << "\n";
                }
            }
#endif
            break;
        }
        residual_pre = residual_cur;
    }
    // 这里不知道为什么只有顶层BA会更新pose_vec, 后面试试前面的也更新
    for (int i = 0; i < window_size; i++)
    {
        layer.pose_vec[i].q = Quaterniond(x_buf[i].R);
        layer.pose_vec[i].t = x_buf[i].p;
    }
}

void distribute_thread(LAYER &layer, LAYER &next_layer)
{
    int &thread_num = layer.thread_num;
    // 创建线程任务
    for(int i =0; i < thread_num; i++)
    {
        if(i < thread_num - 1)
        {
            layer.mthreads[i] = new thread(parallel_head, ref(layer), i, ref(next_layer));
        }
        else
        {
            layer.mthreads[i] = new thread(parallel_tail, ref(layer), i, ref(next_layer));
        }
    }

    //分线程处理
    for(int i = 0; i < thread_num; i++)
    {
        layer.mthreads[i]->join();// 这会阻塞当前线程，直到 mthreads[i] 指向的线程完成执行。
        delete layer.mthreads[i];// 在等待线程完成后，使用 delete 释放 mthreads[i] 指向的线程对象的内存。这是为了避免内存泄漏。
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "hba_modified"); // 初始化ROS节点
    ros::NodeHandle nh("~");      // 创建节点句柄

    /******* NodeHandle的作用 ******/
    /*
    1. 初始化ROS节点
    2. 管理通信功能(包括管理节点、创建节点、创建话题、创建服务等)
    3. 为节点提供一个命名空间
    4. 为节点提供参数服务器
    5. 为节点提供日志记录
    6. 节点的生命周期管理
    */

    int total_layer_num, thread_num; // 总层数量，线程数量
    string data_path;

    // 从参数服务器中获取参数
    nh.getParam("total_layer_num", total_layer_num);     // 获取总层数量
    nh.getParam("pcd_name_fill_num", pcd_name_fill_num); // 获取pcd文件名填充数量
    nh.getParam("data_path", data_path);                 // 获取数据路径
    nh.getParam("thread_num", thread_num);               // 获取线程数量

    // 创建HBA对象
    HBA hba(total_layer_num, data_path, thread_num);

    // 并行计算优化
    for(int i = 0; i < total_layer_num - 1; i++)
    {
        std::cout << "---------------------" << std::endl;
        distribute_thread(hba.layers[i], hba.layers[i + 1]);
        hba.update_next_layer_state(i);
    }

    // 全局BA
    global_ba(hba.layers[total_layer_num - 1]);
    hba.pose_graph_optimization();

    printf("HBA Iteration Complete!\n");

    return 0;
}
