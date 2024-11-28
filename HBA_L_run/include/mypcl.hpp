#ifndef MYPCL_HPP
#define MYPCL_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>

// 定义一个pose结构体，包含一个四元数和一个三维向量
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vector_vec3d;
typedef std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> vector_quad;
// typedef pcl::PointXYZINormal PointType;
typedef pcl::PointXYZ PointType;
// typedef pcl::PointXYZI PointType;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

// 雷达时间戳
std::vector<double> lidar_time;

namespace mypcl
{
    // 自定义位姿结构体
    struct pose
    {
        // pose(q,t),q是四元数，t是三维向量，先初始化再定义，确保变量被初始化
        pose(Eigen::Quaterniond _q = Eigen::Quaterniond(1, 0, 0, 0),
             Eigen::Vector3d _t = Eigen::Vector3d(0, 0, 0)) : q(_q), t(_t) {}
        Eigen::Quaterniond q;
        Eigen::Vector3d t;
    };

    // 读取pcd文件, 输入文件路径，pcd文件名，点云指针，文件编号，文件前缀
    void loadPCD(std::string filePath, int pcd_fill_num, pcl::PointCloud<PointType>::Ptr &pc, int num, std::string prefix = "")
    {
        std::stringstream ss;
        // 这个是pcd文件名的填充数，比如文件名是0001.pcd，那么这个填充数就是4, launch文件中设置
        if (pcd_fill_num > 0)
        {
            ss << std::setw(pcd_fill_num) << std::setfill('0') << num;
        }
        else
        {
            ss << num;
        }
        pcl::io::loadPCDFile(filePath + prefix + ss.str() + ".pcd", *pc);
    }

    // 保存pcd文件
    void savdPCD(std::string filePath, int pcd_fill_num, pcl::PointCloud<PointType>::Ptr &pc, int num)
    {
        std::stringstream ss;
        if (pcd_fill_num > 0)
        {
            ss << std::setw(pcd_fill_num) << std::setfill('0') << num;
        }
        else
        {
            ss << num;
        }
        pcl::io::savePCDFileBinary(filePath + ss.str() + ".pcd", *pc);
    }

    // 读取位姿文件, 输入文件名，四元数，三维向量
    std::vector<pose> read_pose(std::string filename, Eigen::Quaterniond qe = Eigen::Quaterniond(1, 0, 0, 0), Eigen::Vector3d te = Eigen::Vector3d(0, 0, 0))
    {
        std::vector<pose> pose_vec;
        std::fstream file;
        file.open(filename);
        double lt, tx, ty, tz, w, x, y, z;

        while (!file.eof()) // 是否达到文件流的末尾
        {
            file >>lt >> tx >> ty >> tz >> w >> x >> y >> z;
            Eigen::Quaterniond q(w, x, y, z);
            Eigen::Vector3d t(tx, ty, tz);
            pose_vec.push_back(pose(qe * q, qe * t + te)); // 这个是为了方便转换坐标？
            lidar_time.push_back(lt);
        }

        file.close();
        return pose_vec;
    }

    // 作用是将点云pc_in按照位姿t和q进行变换，变换后的点云存储在pt_out中
    void transform_pointcloud(pcl::PointCloud<PointType> const &pc_in, pcl::PointCloud<PointType> &pt_out, Eigen::Vector3d t, Eigen::Quaterniond q)
    {
        size_t size = pc_in.points.size();
        pt_out.points.resize(size);
        for (size_t i = 0; i < size; i++)
        {
            Eigen::Vector3d pt_cur(pc_in.points[i].x, pc_in.points[i].y, pc_in.points[i].z);
            Eigen::Vector3d pt_to;
            // if(pt_cur.norm()<0.3) continue; // 这个是为了过滤掉距离原点太近的点
            pt_to = q * pt_cur + t;
            pt_out.points[i].x = pt_to.x();
            pt_out.points[i].y = pt_to.y();
            pt_out.points[i].z = pt_to.z();
            // pt_out.points[i].r = pc_in.points[i].r;
            // pt_out.points[i].g = pc_in.points[i].g;
            // pt_out.points[i].b = pc_in.points[i].b;
        }
    }

    // 作用是将两个点云合并,这个是合并彩色点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr append_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc1, pcl::PointCloud<pcl::PointXYZRGB> pc2)
    {
        size_t size1 = pc1->points.size();
        size_t size2 = pc2.points.size();
        pc1->points.resize(size1 + size2);
        for (size_t i = size1; i < size1 + size2; i++)
        {
            // 将pc2的点云赋值给pc1
            pc1->points[i].x = pc2.points[i - size1].x;
            pc1->points[i].y = pc2.points[i - size1].y;
            pc1->points[i].z = pc2.points[i - size1].z;
            pc1->points[i].r = pc2.points[i - size1].r;
            pc1->points[i].g = pc2.points[i - size1].g;
            pc1->points[i].b = pc2.points[i - size1].b;
            // pc1->points[i].intensity = pc2.points[i-size1].intensity;//点云强度
        }

        return pc1;
    }

    // 合并非彩色点云
    pcl::PointCloud<PointType>::Ptr append_cloud(pcl::PointCloud<PointType>::Ptr pc1, pcl::PointCloud<PointType> pc2)
    {
        size_t size1 = pc1->points.size();
        size_t size2 = pc2.points.size();
        pc1->points.resize(size1 + size2);
        for (size_t i = size1; i < size1 + size2; i++)
        {
            pc1->points[i].x = pc2.points[i - size1].x;
            pc1->points[i].y = pc2.points[i - size1].y;
            pc1->points[i].z = pc2.points[i - size1].z;
            // pc1->points[i].r = pc2.points[i-size1].r;
            // pc1->points[i].g = pc2.points[i-size1].g;
            // pc1->points[i].b = pc2.points[i-size1].b;
            // pc1->points[i].intensity = pc2.points[i-size1].intensity;
        }
        return pc1;
    }

    // 这行代码的作用是从 std::set<double> 容器 dis_vec 中获取一个特定位置的元素，并返回该元素。
    double compute_inlier_ratio(std::vector<double> residuals, double ratio)
    {
        std::set<double> dis_vec; // set容器，具有唯一性，自动排序，insert插入，find查找
        for (size_t i = 0; i < (size_t)(residuals.size() / 3); i++)
        {
            dis_vec.insert(fabs(residuals[3 * i + 0]) + fabs(residuals[3 * i + 1]) + fabs(residuals[3 * i + 2]));
        }

        return *(std::next(dis_vec.begin(), (int)((ratio)*dis_vec.size())));
    }

    // 写入位姿文件
    void write_pose(std::vector<pose> &pose_vec, std::string path)
    {
        std::ofstream file;
        file.open(path + "pose.json", std::ofstream::trunc);
        file.close();
        Eigen::Quaterniond q0(pose_vec[0].q.w(), pose_vec[0].q.x(), pose_vec[0].q.y(), pose_vec[0].q.z());
        Eigen::Vector3d t0(pose_vec[0].t(0), pose_vec[0].t(1), pose_vec[0].t(2));
        file.open(path + "pose.json", std::ofstream::app);

        for (size_t i = 0; i < pose_vec.size(); i++)
        {
            pose_vec[i].t << q0.inverse() * (pose_vec[i].t - t0);
            pose_vec[i].q.w() = (q0.inverse() * pose_vec[i].q).w();
            pose_vec[i].q.x() = (q0.inverse() * pose_vec[i].q).x();
            pose_vec[i].q.y() = (q0.inverse() * pose_vec[i].q).y();
            pose_vec[i].q.z() = (q0.inverse() * pose_vec[i].q).z();
            file << pose_vec[i].t(0) << " "
                 << pose_vec[i].t(1) << " "
                 << pose_vec[i].t(2) << " "
                 << pose_vec[i].q.w() << " " << pose_vec[i].q.x() << " "
                 << pose_vec[i].q.y() << " " << pose_vec[i].q.z();
            if (i < pose_vec.size() - 1)
                file << "\n";
        }
        lidar_time.clear();
        file.close();
    }

    // 写入EVO格式的位姿文件
    void writeEVOPose(std::vector<double> &lidar_times, std::vector<pose> &pose_vec, std::string path)
    {
        std::ofstream file;
        file.open(path + "evo_pose.txt", std::ofstream::trunc);
        for (size_t i = 0; i < pose_vec.size(); i++)
        {
            file << std::setprecision(18) << lidar_times[i] << " " << std::setprecision(6)
                 << pose_vec[i].t(0) << " " << pose_vec[i].t(1) << " " << pose_vec[i].t(2) << " "
                 << pose_vec[i].q.x() << " " << pose_vec[i].q.y() << " "
                 << pose_vec[i].q.z() << " " << pose_vec[i].q.w();
            if (i < pose_vec.size() - 1)
                file << "\n";
        }
        lidar_time.clear();
        file.close();
    }

}

#endif