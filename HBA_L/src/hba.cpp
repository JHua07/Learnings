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

#include "ba.hpp"
#include "hba.hpp"
#include "tools.hpp"
#include "mypcl.hpp"

using namespace std;
using namespace Eigen;

int pcd_name_fill_num = 0;

void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> &feat_map, pcl::PointCloud<PointType> &feat_pt, Eigen::Quaterniond q, Eigen::Vector3d t, int fnum, double voxel_size, int window_size, float eigen_ratio)
{
    float loc_xyz[3];
    for (PointType &p_c : feat_pt.points)
    {
        Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
        Eigen::Vector3d pvec_tran = q * pvec_orig + t;

        for (int j = 0; j < 3; j++)
        {
            loc_xyz[j] = pvec_tran[j] / voxel_size;
            if (loc_xyz[j] < 0)
            {
                loc_xyz[j] -= 1.0;
            }
        }

        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        if (iter != feat_map.end()) // 如果找到, 将原始体素点集，转换体素点集，原始体素因子集，转换体素因子集，分别存入
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

/// @brief 执行某层非最后一个线程的并行计算
/// @param layer 
/// @param thread_id 
/// @param next_layer 
void parallel_comp(LAYER &layer, int thread_id, LAYER &next_layer)
{
    int& part_length = layer.part_length; // 每个线程分配的任务量
    int& layer_num = layer.layer_num;     // 层索引,从1开始

    //处理当前线程的任务, 分别处理每个线程的每个任务，即分别处理每一个窗口
    for(int i = thread_id * part_length; i < (thread_id + 1) * part_length; i++)
    {
        vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;// ？？点云帧()，原始点的帧(从pcd文件读取)
        src_pc.resize(WIN_SIZE); // 点云帧的大小初始化为窗口大小
        raw_pc.resize(WIN_SIZE); 

        double residual_cur = 0, residual_pre = 0; // 当前残差，上一次残差
        vector<IMUST> x_buf(WIN_SIZE); // 位姿向量(t, R, p, v, bg, ba, g)

        //计算窗口内的位姿
        for(int j = 0; j < WIN_SIZE; j++)
        {
            // 将当前层的当前线程的任务的位姿赋值给x_buf
            x_buf[j].R = layer.pose_vec[i * GAP + j].q.toRotationMatrix(); // 旋转矩阵
            x_buf[j].p = layer.pose_vec[i * GAP + j].t; // 位移
        }

        if(layer_num != 1)//非底层
        {
            for(int j = i * GAP; j < i * GAP + WIN_SIZE; j++)
            {
                src_pc[j - i * GAP] = (*layer.pcds[j]).makeShared(); // 将点云复制给当前窗口
            }
        }

        size_t mem_cost = 0; // 内存消耗
        for(int loop = 0; loop < layer.max_iter; loop++)//循环，开始ba
        {
            if(layer_num == 1)//底层
            {
                // 这个时候底层还没有读入点云，所以需要从路径中读取pcd文件, 一个窗口一个窗口进行读取
                for(int j = i * GAP; j < i * GAP + WIN_SIZE; j++)
                {
                    if(loop == 0)
                    {
                        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
                        mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
                        raw_pc[j - i * GAP] = pc; // 读取pcd文件，将每帧的点云赋值给raw_pc
                    }
                    src_pc[j - i * GAP] = (*raw_pc[j - i * GAP]).makeShared(); // 将点云赋值
                }
            }

            unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map; // 体素地图
            
            // 窗口内的每一帧点云进行体素降采样
            for(size_t j = 0; j < WIN_SIZE; j++)
            {
                if(layer.downsample_size > 0)
                {
                    downsample_voxel(*src_pc[j], layer.downsample_size); // 体素降采样
                    cut_voxel(surf_map, *src_pc[j], Eigen::Quaterniond(x_buf[j].R), x_buf[j].p, j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio); // 分割体素
                }
            }

            for(auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
            {
                iter->second->recut(); // 重新分割, 为了确定每个体素的特征
            }

            VOX_HESS voxhess(WIN_SIZE);
            for(auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
            {
                iter->second->tras_opt(voxhess); // 体素优化器
            }

            VOX_OPTIMIZER opt_lsv(WIN_SIZE);
            opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio); // 移除离群点
            PLV(6) hess_vec;

            // 阻尼迭代,其核心思想是在每次迭代更新时引入一个“阻尼”因子，通过控制步长来确保迭代过程的稳定性，并避免过大的跳跃或震荡。
            opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost); // 计算出当前的残差、内存成本和hessian矩阵

            for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
            {
                delete iter->second;// 释放内存
            }

            // 输出残差和更新hessian矩阵，循环残差相对增量小于0.05或者达到最大迭代次数时，结束循环
            if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1)
            {
                if (layer.mem_costs[thread_id] < mem_cost)
                {
                    layer.mem_costs[thread_id] = mem_cost; // 如果当前线程的内存成本小于计算的内存成本，则更新内存成本。
                }
                for (int j = 0; j < WIN_SIZE * (WIN_SIZE - 1) / 2; j++)
                {
                    layer.hessians[i * (WIN_SIZE - 1) * WIN_SIZE / 2 + j] = hess_vec[j]; // 将hessian矩阵赋值给当前线程的hessian矩阵
                }
                break;
            }
            residual_pre = residual_cur; // 更新上一次的残差
        }

        pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>); // 关键帧点云
        for(size_t j = 0; j < WIN_SIZE; j++)
        {
            Eigen::Quaterniond q_tmp;
            Eigen::Vector3d t_tmp;
            assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R), x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p)); // 计算相对于窗口内第一个点的相对位移和旋转
            
            pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
            mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp); // 将点云src_pc按照q_tmp和t_tmp进行变换，结果存储在pc_oneframe中
            pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe); // 合并点云
        }
        downsample_voxel(*pc_keyframe, 0.05); // 体素降采样
        next_layer.pcds[i] = pc_keyframe; // 将关键帧点云赋值给下一层的pcd
    }

}

/// @brief 该层最后一个线程的并行计算
/// @param layer 
/// @param thread_id 
/// @param next_layer 
void parallel_tail(LAYER& layer, int thread_id, LAYER& next_layer)
{
    int &part_length = layer.part_length;   // 每个线程分配的任务量
    int &layer_num = layer.layer_num;       // 层索引,从1开始
    int &left_gap_num = layer.left_gap_num; // 剩余的位姿数量

    double load_t = 0, undis_t = 0, dsp_t = 0, cut_t = 0, recut_t = 0, total_t = 0,
           tran_t = 0, sol_t = 0, save_t = 0; // 计时

    if (layer.gap_num - (layer.thread_num - 1) * part_length + 1 != left_gap_num)
    {
        printf("THIS IS WRONG!\n");
    }

    // 处理最后一个线程的所有满窗口
    for (uint i = thread_id * part_length; i < thread_id * part_length + left_gap_num; i++)
    {
        printf("parallel computing %d\n", i);
        double t0, t1;
        double t_begin = ros::Time::now().toSec();

        vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
        src_pc.resize(WIN_SIZE);
        raw_pc.resize(WIN_SIZE);

        double residual_cur = 0, residual_pre = 0;
        vector<IMUST> x_buf(WIN_SIZE);
        for (int j = 0; j < WIN_SIZE; j++) // 计算每个窗口内的位姿
        {
            x_buf[j].R = layer.pose_vec[i * GAP + j].q.toRotationMatrix();
            x_buf[j].p = layer.pose_vec[i * GAP + j].t;
        }

        if (layer_num != 1) // 非底层
        {
            t0 = ros::Time::now().toSec();
            for (int j = i * GAP; j < i * GAP + WIN_SIZE; j++)
                src_pc[j - i * GAP] = (*layer.pcds[j]).makeShared();
            load_t += ros::Time::now().toSec() - t0; // 复制上一层的pcd并记录加载时间
        }

        size_t mem_cost = 0;
        for (int loop = 0; loop < layer.max_iter; loop++)
        {
            if (layer_num == 1) // 底层
            {
                t0 = ros::Time::now().toSec();
                for (int j = i * GAP; j < i * GAP + WIN_SIZE; j++)
                {
                    if (loop == 0)
                    {
                        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
                        mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
                        raw_pc[j - i * GAP] = pc;
                    }
                    src_pc[j - i * GAP] = (*raw_pc[j - i * GAP]).makeShared();
                }
                load_t += ros::Time::now().toSec() - t0;
            }

            unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

            for (size_t j = 0; j < WIN_SIZE; j++)
            {
                t0 = ros::Time::now().toSec();
                if (layer.downsample_size > 0)
                    downsample_voxel(*src_pc[j], layer.downsample_size); // 体素降采样
                dsp_t += ros::Time::now().toSec() - t0;

                t0 = ros::Time::now().toSec();
                cut_voxel(surf_map, *src_pc[j], Quaterniond(x_buf[j].R), x_buf[j].p,
                          j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio); // 分割体素
                cut_t += ros::Time::now().toSec() - t0;
            }

            t0 = ros::Time::now().toSec();
            for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
            {
                iter->second->recut(); // 重新分割
            }
            recut_t += ros::Time::now().toSec() - t0;

            t0 = ros::Time::now().toSec();
            VOX_HESS voxhess(WIN_SIZE);
            for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
            {
                iter->second->tras_opt(voxhess); // 体素优化器
            }
            tran_t += ros::Time::now().toSec() - t0;

            VOX_OPTIMIZER opt_lsv(WIN_SIZE);
            t0 = ros::Time::now().toSec();
            opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio); // 去除离群点
            PLV(6)
            hess_vec;
            opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost); // 阻尼迭代,计算当前的残差，内存成本和hessian矩阵
            sol_t += ros::Time::now().toSec() - t0;

            for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
            {
                delete iter->second;
            }

            if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1)
            {
                if (layer.mem_costs[thread_id] < mem_cost)
                {
                    layer.mem_costs[thread_id] = mem_cost; // 如果当前线程的内存成本小于计算的内存成本，则更新内存成本。
                }

                if (i < thread_id * part_length + left_gap_num)
                {
                    for (int j = 0; j < WIN_SIZE * (WIN_SIZE - 1) / 2; j++)
                    {
                        layer.hessians[i * (WIN_SIZE - 1) * WIN_SIZE / 2 + j] = hess_vec[j]; // 更新hessian矩阵
                    }
                }
                break;
            }
            residual_pre = residual_cur;
        }
        pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
        for (size_t j = 0; j < WIN_SIZE; j++)
        {
            t1 = ros::Time::now().toSec();
            Eigen::Quaterniond q_tmp;
            Eigen::Vector3d t_tmp;
            assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                      x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p)); // 计算相对于窗口内第一个点的相对位移和旋转

            pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
            mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp); // 变换点云
            pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);        // 合并点云
            save_t += ros::Time::now().toSec() - t1;
        }
        t0 = ros::Time::now().toSec();
        downsample_voxel(*pc_keyframe, 0.05); // 体素降采样
        dsp_t += ros::Time::now().toSec() - t0;

        t0 = ros::Time::now().toSec();
        next_layer.pcds[i] = pc_keyframe; // 将关键帧点云赋值给下一层的pcd
        save_t += ros::Time::now().toSec() - t0;

        total_t += ros::Time::now().toSec() - t_begin;
    }

    // 这里可以看做是最后一次处理
    if (layer.tail > 0)
    {
        // 起始索引
        int i = thread_id * part_length + left_gap_num;

        vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
        src_pc.resize(layer.last_win_size);
        raw_pc.resize(layer.last_win_size);
        double residual_cur = 0, residual_pre = 0;
        vector<IMUST> x_buf(layer.last_win_size);
        for (int j = 0; j < layer.last_win_size; j++)
        {
            x_buf[j].R = layer.pose_vec[i * GAP + j].q.toRotationMatrix();
            x_buf[j].p = layer.pose_vec[i * GAP + j].t;
        }

        if (layer_num != 1)
        {
            for (int j = i * GAP; j < i * GAP + layer.last_win_size; j++)
                src_pc[j - i * GAP] = (*layer.pcds[j]).makeShared();
        }

        size_t mem_cost = 0;
        for (int loop = 0; loop < layer.max_iter; loop++)
        {
            if (layer_num == 1)
                for (int j = i * GAP; j < i * GAP + layer.last_win_size; j++)
                {
                    if (loop == 0)
                    {
                        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
                        mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
                        raw_pc[j - i * GAP] = pc;
                    }
                    src_pc[j - i * GAP] = (*raw_pc[j - i * GAP]).makeShared();
                }

            unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

            for (size_t j = 0; j < layer.last_win_size; j++)
            {
                if (layer.downsample_size > 0)
                    downsample_voxel(*src_pc[j], layer.downsample_size);
                cut_voxel(surf_map, *src_pc[j], Quaterniond(x_buf[j].R), x_buf[j].p,
                          j, layer.voxel_size, layer.last_win_size, layer.eigen_ratio);
            }
            for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
                iter->second->recut();

            VOX_HESS voxhess(layer.last_win_size);
            for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
                iter->second->tras_opt(voxhess);

            VOX_OPTIMIZER opt_lsv(layer.last_win_size);
            opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
            PLV(6)
            hess_vec;
            opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

            for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
                delete iter->second;

            if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1)
            {
                if (layer.mem_costs[thread_id] < mem_cost)
                    layer.mem_costs[thread_id] = mem_cost;

                for (int j = 0; j < layer.last_win_size * (layer.last_win_size - 1) / 2; j++)
                    layer.hessians[i * (WIN_SIZE - 1) * WIN_SIZE / 2 + j] = hess_vec[j];

                break;
            }
            residual_pre = residual_cur;
        }

        pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
        for (size_t j = 0; j < layer.last_win_size; j++)
        {
            Eigen::Quaterniond q_tmp;
            Eigen::Vector3d t_tmp;
            assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                      x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

            pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
            mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
            pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
        }
        downsample_voxel(*pc_keyframe, 0.05);
        next_layer.pcds[i] = pc_keyframe;
    }
    printf("total time: %.2fs\n", total_t);
    printf("load pcd %.2fs %.2f%% | undistort pcd %.2fs %.2f%% | "
           "downsample %.2fs %.2f%% | cut voxel %.2fs %.2f%% | recut %.2fs %.2f%% | trans %.2fs %.2f%% | solve %.2fs %.2f%% | "
           "save pcd %.2fs %.2f%%\n",
           load_t, load_t / total_t * 100, undis_t, undis_t / total_t * 100,
           dsp_t, dsp_t / total_t * 100, cut_t, cut_t / total_t * 100, recut_t, recut_t / total_t * 100, tran_t, tran_t / total_t * 100,
           sol_t, sol_t / total_t * 100, save_t, save_t / total_t * 100);
}

/// @brief 这个函数实现了某一层的线程的并行计算，即某一层中完成ba
/// @param layer
/// @param next_layer
void distribute_thread(LAYER &layer, LAYER &next_layer)
{
    int &thread_num = layer.thread_num;   // 线程数量
    double t0 = ros::Time::now().toSec(); // 记录开始时间

    // 遍历线程，进行每个线程的并行计算
    for (int i = 0; i < thread_num - 1; i++)
    {
        // 非最后线程并行计算
        if (i < thread_num - 1)
        {
            layer.mthreads[i] = new thread(parallel_comp, ref(layer), i, ref(next_layer)); // 创建线程
        }
        else
        {
            layer.mthreads[i] = new thread(parallel_tail, ref(layer), i, ref(next_layer)); // 创建线程
        }
        // printf("Thread distribution time: %f\n", ros::Time::now().toSec()-t0);
        // 此时就完成了底层向上的ba
    }
    t0 = ros::Time::now().toSec();
    for (int i = 0; i < thread_num; i++)
    {
        layer.mthreads[i]->join(); // 这会阻塞当前线程，直到 mthreads[i] 指向的线程完成执行。换句话说，当前线程会等待 mthreads[i] 线程结束。
        delete layer.mthreads[i];  // 在等待线程完成后，使用 delete 释放 mthreads[i] 指向的线程对象的内存。这是为了避免内存泄漏。
    }
    // printf("Thread join time: %f\n", ros::Time::now().toSec()-t0);
}

/// @brief 全局ba
void global_ba(LAYER &layer)
{
    int window_size = layer.pose_vec.size(); // 顶层直接处理所有的位姿
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
        src_pc[i] = (*layer.pcds[i]).makeShared(); // 读取顶层的点云pcd
    }

    double residual_cur = 0, residual_pre = 0;
    size_t mem_cost = 0, max_mem = 0;
    double dsp_t = 0, cut_t = 0, recut_t = 0, tran_t = 0, sol_t = 0, t0;
    for (int loop = 0; loop < layer.max_iter; loop++)
    {
        std::cout << "---------------------" << std::endl;
        std::cout << "Iteration " << loop << std::endl;

        unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

        for (int i = 0; i < window_size; i++)
        {
            t0 = ros::Time::now().toSec();
            if (layer.downsample_size > 0)
                downsample_voxel(*src_pc[i], layer.downsample_size);
            dsp_t += ros::Time::now().toSec() - t0;
            t0 = ros::Time::now().toSec();
            cut_voxel(surf_map, *src_pc[i], Quaterniond(x_buf[i].R), x_buf[i].p, i,
                      layer.voxel_size, window_size, layer.eigen_ratio * 2);
            cut_t += ros::Time::now().toSec() - t0;
        }
        t0 = ros::Time::now().toSec();
        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
            iter->second->recut();
        recut_t += ros::Time::now().toSec() - t0;

        t0 = ros::Time::now().toSec();
        VOX_HESS voxhess(window_size);
        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
            iter->second->tras_opt(voxhess);
        tran_t += ros::Time::now().toSec() - t0;

        t0 = ros::Time::now().toSec();
        VOX_OPTIMIZER opt_lsv(window_size);
        opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
        PLV(6)
        hess_vec;
        opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
        sol_t += ros::Time::now().toSec() - t0;

        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
            delete iter->second;

        cout << "Residual absolute: " << abs(residual_pre - residual_cur) << " | "
             << "percentage: " << abs(residual_pre - residual_cur) / abs(residual_cur) << endl;

        if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1)
        {
            if (max_mem < mem_cost)
                max_mem = mem_cost;
#ifdef FULL_HESS // 记录完整hessian矩阵
            for (int i = 0; i < window_size * (window_size - 1) / 2; i++) // 遍历hessian矩阵的上三角部分元素
                layer.hessians[i] = hess_vec[i];
#else // 记录对角线和上三角部分
            for (int i = 0; i < window_size - 1; i++)
            {
                Matrix6d hess = Hess_cur.block(6 * i, 6 * i + 6, 6, 6);
                for (int row = 0; row < 6; row++)
                    for (int col = 0; col < 6; col++)
                        hessFile << hess(row, col) << ((row * col == 25) ? "" : " ");
                if (i < window_size - 2)
                    hessFile << "\n";
            }
#endif
            break;
        }
        residual_pre = residual_cur;
    }

    // 更新顶层的变化矩阵
    for(int i = 0; i < window_size; i++)
    {
        layer.pose_vec[i].q = Eigen::Quaterniond(x_buf[i].R);
        layer.pose_vec[i].t = x_buf[i].p;
    }
    printf("Downsample: %f, Cut: %f, Recut: %f, Tras: %f, Sol: %f\n", dsp_t, cut_t, recut_t, tran_t, sol_t);
}

    int main(int argc, char **argv)
    {
        ros::init(argc, argv, "hba"); // 初始化ROS节点
        ros::NodeHandle nh("~");      // 创建节点句柄

        /******* NodeHandle的作用 ******/
        /*
        1. 初始化ROS节点
        2. 管理通信功能(包括管理节点、创建节点、创建话题、创建服务等)、
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

        HBA hba(total_layer_num, data_path, thread_num); // 创建HBA对象, 初始化各层的参数

        // 底层向上ba
        for (int i = 0; i < total_layer_num - 1; i++)
        {
            std::cout << "---------------------" << std::endl;
            distribute_thread(hba.layers[i], hba.layers[i + 1]); // 分配线程
            hba.update_next_layer_state(i);                      // 更新下一层的状态
        }

        // 顶层向下ba
        global_ba(hba.layers[total_layer_num - 1]); // 全局ba
        hba.pose_graph_optimization();              // 位姿图优化
        printf("Iteration Complete!\n");
    }