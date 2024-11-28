#ifndef BA_HPP
#define BA_HPP

#include <thread>
#include <fstream>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "tools.hpp"

#define WIN_SIZE 10 // 窗口大小，即一个窗口内的位姿数量
#define GAP 5       // 步长
#define AVG_THR
#define FULL_HESS
// #define ENABLE_RVIZ
// #define ENABLE_FILTER

const double one_three = (1.0 / 3.0);

int layer_limit = 2;
int MIN_PT = 15;
int thd_num = 16;

// 现在有个问题，是一个体素内只有一个窗口还是多个窗口？
// VOX_HESS类
class VOX_HESS
{
public:
    vector<const vector<VOX_FACTOR> *> plvec_voxels; // 体素因子向量，元素包括（P, v, N）, 方法有clear, push, cov, operator+=, transform
    vector<PLV(3)> origin_points;                    // 原始点(3,1)的向量
    int win_size;                                    // 窗口大小

    // 构造函数，初始化变量
    VOX_HESS(int _win_size = WIN_SIZE) : win_size(_win_size) { origin_points.resize(win_size); }

    // 析构函数，释放内存
    ~VOX_HESS()
    {
        vector<const vector<VOX_FACTOR> *>().swap(plvec_voxels);
    }

    /// @brief  将vec_orig中的点添加到origin_points_中
    /// @param  vec_orig 输入的点集
    /// @param  origin_points_ 全部原始点点集
    void get_center(const PLV(3) & vec_orig, PLV(3) & origin_points_)
    {
        size_t pt_size = vec_orig.size();
        for (size_t i = 0; i < pt_size; i++)
        {
            origin_points_.emplace_back(vec_orig[i]); // 将vec_orig[i]添加到origin_points_中，emplace_back是在容器的末尾直接构造一个对象
        }
        return;
    }

    /// @brief  将输入体素因子向量添加到 plvec_voxels 容器中
    /// @param  sig_orig 体素因子向量
    /// @param  vec_orig 体素内点集
    void push_voxel(const vector<VOX_FACTOR> *sig_orig, const vector<PLV(3)> *vec_orig)
    {
        int process_size = 0; //
        // 计算有效体素的数量
        for (int i = 0; i < win_size; i++)
        {
            // 若原始的体素因子的数量不为0，则process_size加1
            if ((*sig_orig)[i].N != 0)
            {
                process_size++;
            }
        }
#ifdef ENABLE_FILTER
        // 如果有效体素的数量小于1，则直接返回
        if (process_size < 1)
        {
            return;
        }

        // 将vec_orig[i]中的点添加到origin_points[i]中
        for (int i = 0; i < win_size; i++)
        {
            if ((*sig_orig)[i].N != 0)
            {
                // 将vec_orig[i]中的点添加到origin_points[i]中
                get_center((*vec_orig)[i], origin_points[i]);
            }
        }
#endif

        // 如果有效体素的数量小于2，直接返回
        if (process_size < 2)
        {
            return;
        }

        // 将输入的原始信号向量添加到 plvec_voxels 容器中
        plvec_voxels.push_back(sig_orig);
    }

    // 这个好像没用到？可能是计算雅各布矩阵？
    Eigen::Matrix<double, 6, 1> lam_f(Eigen::Vector3d *u, int m, int n)
    {
        Eigen::Matrix<double, 6, 1> jac;
        jac[0] = u[m][0] * u[n][0];
        jac[1] = u[m][0] * u[n][1] + u[m][1] * u[n][0];
        jac[2] = u[m][0] * u[n][2] + u[m][2] * u[n][0];
        jac[3] = u[m][1] * u[n][1];
        jac[4] = u[m][1] * u[n][2] + u[m][2] * u[n][1];
        jac[5] = u[m][2] * u[n][2];
        return jac;
    }

    /// @brief 计算体素内各窗口的Hessian矩阵，雅各布矩阵和残差，两个矩阵见formula1.png和formula2.png推导见Reference，这个残差是每个窗口的最小的特征值的和
    /// @param xs
    /// @param head
    /// @param end
    /// @param Hess
    /// @param JacT
    /// @param residual
    void acc_evaluate2(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
    {
        // 初始化
        Hess.setZero();
        JacT.setZero();
        residual = 0;
        vector<VOX_FACTOR> sig_tran(win_size);
        const int kk = 0;

        PLV(3)
        viRiTuk(win_size);
        PLM(3)
        viRiTukukT(win_size);

        vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
        Eigen::Matrix3d umumT;

        // 体素存储了一个或多个窗口内的位姿信息
        for (int a = head; a < end; a++)
        {
            // 对于体素 a，获取其原始数据 sig_orig
            const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[a];

            VOX_FACTOR sig;

            // 对于每个窗口 i，将 sig_orig[i] 通过 xs[i] 转换到 xs的坐标系下，包括协方差，质心和点数量
            for (int i = 0; i < win_size; i++)
            {
                if (sig_orig[i].N != 0)
                {
                    sig_tran[i].transform(sig_orig[i], xs[i]);
                    sig += sig_tran[i];
                }
            }
            // 质心坐标
            const Eigen::Vector3d &vBar = sig.v / sig.N;

            // 体素内点云的协方差矩阵,并进行特征值分解
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P / sig.N - vBar * vBar.transpose());
            // 特征值，这个特征值是按照从小到大排列的
            const Eigen::Vector3d &lmbd = saes.eigenvalues();
            // 特征向量组成的矩阵
            const Eigen::Matrix3d &U = saes.eigenvectors();
            // 记录体素内点的数量
            int NN = sig.N;

            // 特征向量
            Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};

            // 计算雅各布矩阵
            const Eigen::Vector3d &uk = u[kk]; // kk=0, 这里就是uk = u(0)，也就是第一个特征向量
            Eigen::Matrix3d ukukT = uk * uk.transpose();
            umumT.setZero();
            for (int i = 0; i < 3; i++)
            {
                if (i != kk)
                {
                    umumT += 2.0 / (lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();
                }
            }

            // 计算hessian矩阵，这个hessian矩阵在formula1.png和formula2.png中，具体推导参考论文Reference的第8到11页
            for (int i = 0; i < win_size; i++)
            {
                if (sig_orig[i].N != 0)
                {
                    Eigen::Matrix3d Pi = sig_orig[i].P;
                    Eigen::Vector3d vi = sig_orig[i].v;
                    Eigen::Matrix3d Ri = xs[i].R;
                    double ni = sig_orig[i].N;

                    Eigen::Matrix3d vihat;
                    vihat << SKEW_SYM_MATRX(vi); // 反对称矩阵
                    Eigen::Vector3d RiTuk = Ri.transpose() * uk;
                    Eigen::Matrix3d RiTukhat;
                    RiTukhat << SKEW_SYM_MATRX(RiTuk);

                    Eigen::Vector3d PiRiTuk = Pi * RiTuk;
                    viRiTuk[i] = vihat * RiTuk;
                    viRiTukukT[i] = viRiTuk[i] * uk.transpose();

                    Eigen::Vector3d ti_v = xs[i].p - vBar;
                    double ukTti_v = uk.dot(ti_v);

                    Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
                    Eigen::Vector3d combo2 = Ri * vi + ni * ti_v;
                    Auk[i].block<3, 3>(0, 0) = (Ri * Pi + ti_v * vi.transpose()) * RiTukhat - Ri * combo1;
                    Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
                    Auk[i] /= NN;

                    const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
                    JacT.block<6, 1>(6 * i, 0) += jjt;

                    const Eigen::Matrix3d &HRt = 2.0 / NN * (1.0 - ni / NN) * viRiTukukT[i];
                    Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
                    Hb.block<3, 3>(0, 0) += 2.0 / NN * (combo1 - RiTukhat * Pi) * RiTukhat - 2.0 / NN / NN * viRiTuk[i] * viRiTuk[i].transpose() - 0.5 * hat(jjt.block<3, 1>(0, 0));
                    Hb.block<3, 3>(0, 3) += HRt;
                    Hb.block<3, 3>(3, 0) += HRt.transpose();
                    Hb.block<3, 3>(3, 3) += 2.0 / NN * (ni - ni * ni / NN) * ukukT;

                    Hess.block<6, 6>(6 * i, 6 * i) += Hb;
                }
            }

            for (int i = 0; i < win_size - 1; i++)
            {
                if (sig_orig[i].N != 0)
                {
                    double ni = sig_orig[i].N;
                    for (int j = i + 1; j < win_size; j++)
                    {
                        if (sig_orig[j].N != 0)
                        {
                            double nj = sig_orig[j].N;
                            Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
                            Hb.block<3, 3>(0, 0) += -2.0 / NN / NN * viRiTuk[i] * viRiTuk[j].transpose();
                            Hb.block<3, 3>(0, 3) += -2.0 * nj / NN / NN * viRiTukukT[i];
                            Hb.block<3, 3>(3, 0) += -2.0 * ni / NN / NN * viRiTukukT[j].transpose();
                            Hb.block<3, 3>(3, 3) += -2.0 * ni * nj / NN / NN * ukukT;

                            Hess.block<6, 6>(6 * i, 6 * j) += Hb;
                        }
                    }
                }
            }

            residual += lmbd[kk];
        }

        for (int i = 1; i < win_size; i++)
        {
            for (int j = 0; j < i; j++)
            {
                Hess.block<6, 6>(6 * i, 6 * j) = Hess.block<6, 6>(6 * j, 6 * i).transpose();
            }
        }
    }

    // 只计算残差，不计算雅各布矩阵和Hessian矩阵，这个计算的也是所有体素内每个窗口的最小特征值的和？？
    void evaluate_only_residual(const vector<IMUST> &xs, double &residual)
    {
        residual = 0;
        vector<VOX_FACTOR> sig_tran(win_size);
        int kk = 0; // The kk-th lambda value

        int gps_size = plvec_voxels.size();

        for (int a = 0; a < gps_size; a++)
        {
            const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[a];
            VOX_FACTOR sig;

            for (int i = 0; i < win_size; i++)
            {
                sig_tran[i].transform(sig_orig[i], xs[i]);
                sig += sig_tran[i];
            }

            Eigen::Vector3d vBar = sig.v / sig.N;
            Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
            Eigen::Vector3d lmbd = saes.eigenvalues();

            residual += lmbd[kk];
        }
    }

    // 和上面的evaluate_only_residual类似，只是这个函数返回的是一个vector<double>，这个vector<double>是所有体素内每个窗口的最小特征值的和
    std::vector<double> evaluate_residual(const vector<IMUST> &xs)
    {
        /* for outlier removal usage */
        std::vector<double> residuals;
        vector<VOX_FACTOR> sig_tran(win_size);
        int kk = 0; // The kk-th lambda value
        int gps_size = plvec_voxels.size();

        for (int a = 0; a < gps_size; a++)
        {
            const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[a];
            VOX_FACTOR sig;

            for (int i = 0; i < win_size; i++)
            {
                sig_tran[i].transform(sig_orig[i], xs[i]);
                sig += sig_tran[i];
            }

            Eigen::Vector3d vBar = sig.v / sig.N;
            Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
            Eigen::Vector3d lmbd = saes.eigenvalues();

            residuals.push_back(lmbd[kk]);
        }

        return residuals;
    }

    /// @brief 这个函数是用于去除离群点的，这个函数的作用是去除体素内的残差大于阈值的体素，这个函数的实现是遍历所有体素，然后计算体素内所有窗口的残差，如果有一个窗口的残差大于阈值，则去除这个体素
    /// @param xs
    /// @param threshold
    /// @param reject_num
    void remove_residual(const vector<IMUST> &xs, double threshold, double reject_num)
    {
        vector<VOX_FACTOR> sig_tran(win_size);
        int kk = 0; // The kk-th lambda value
        int rej_cnt = 0;
        size_t i = 0;
        for (; i < plvec_voxels.size();)
        {
            const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[i];
            VOX_FACTOR sig;

            for (int j = 0; j < win_size; j++)
            {
                sig_tran[j].transform(sig_orig[j], xs[j]);
                sig += sig_tran[j];
            }

            Eigen::Vector3d vBar = sig.v / sig.N;
            Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
            Eigen::Vector3d lmbd = saes.eigenvalues();

            if (lmbd[kk] >= threshold)
            {
                plvec_voxels.erase(plvec_voxels.begin() + i);
                rej_cnt++;
                continue;
            }
            i++;
            if (rej_cnt == reject_num)
            {
                break;
            }
        }
    }
};

int BINGO_CNT = 0;
enum OCTO_STATE
{
    UNKNOWN,
    MID_NODE,
    PLANE
}; // 八叉树的状态, 未知，中间节点，平面

// 八叉树节点类，目前看来，这个八叉树主要是存储体素的特征，当判断完体素的特征后，就将这个节点里存储的数据清空，然后将这个节点的八个子节点存储数据
class OCTO_TREE_NODE
{
public:
    OCTO_STATE octo_state;                 // 八叉树状态的枚举变量
    int layer, win_size;                   // 层数，窗口大小
    vector<PLV(3)> vec_orig, vec_tran;     // 这里指的应该都是体素集合，原始集，转换后的体素集
    vector<VOX_FACTOR> sig_orig, sig_tran; // 原始体素因子，转换后的体素因子, 里面存放的代表了体素内的点云的协方差矩阵，质心和点的数量

    OCTO_TREE_NODE *leaves[8]; // 八叉树的八个子节点
    float voxel_center[3];     // 体素中心坐标, 用数组表示
    float quater_length;       // 体素的四分之一边长
    float eigen_thr;           // 特征值阈值

    Eigen::Vector3d center, direct, value_vector; // 质心坐标，特征向量，特征值
    double eigen_ratio;

// 选择性发布残差以及方向的值
#ifdef ENABLE_RVIZ
    ros::NodeHandle nh;
    ros::Publisher pub_residual = nh.advertise<sensor_msgs::PointCloud2>("/residual", 1000);
    ros::Publisher pub_direct = nh.advertise<visualization_msgs::MarkerArray>("/direct", 1000);
#endif

    // 构造函数，初始化变量
    OCTO_TREE_NODE(int _win_size = WIN_SIZE, float _eigen_thr = 1.0 / 10) : win_size(_win_size), eigen_thr(_eigen_thr)
    {
        octo_state = UNKNOWN;
        layer = 0;
        vec_orig.resize(win_size);
        vec_tran.resize(win_size);
        sig_orig.resize(win_size);
        sig_tran.resize(win_size);
        for (int i = 0; i < 8; i++)
        {
            leaves[i] = nullptr;
        }
    }

    // 析构函数，释放内存
    virtual ~OCTO_TREE_NODE()
    {
        for (int i = 0; i < 8; i++)
        {
            if (leaves[i] != nullptr)
            {
                delete leaves[i];
            }
        }
    }

    /// @brief  判断体素是否是平面特征,0表示不是平面特征，1表示是平面特征
    bool judge_eigen()
    {
        VOX_FACTOR covMat; // 体素内点云的协方差矩阵
        for (int i = 0; i < win_size; i++)
        {
            // 坐标转换后的体素因子数量大于0
            if (sig_tran[i].N > 0)
            {
                covMat += sig_tran[i];
            }
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
        value_vector = saes.eigenvalues();   // 特征值
        center = covMat.v / covMat.N;        // 质心
        direct = saes.eigenvectors().col(0); // 特征向量

        eigen_ratio = saes.eigenvalues()[0] / saes.eigenvalues()[2]; // 特征值比值,最小特征值/最大特征值，比值小，说明可能是个平面
        if (eigen_ratio > eigen_thr)
        {
            return 0;
        }

        double eva0 = saes.eigenvalues()[0];                          // 最小特征值
        double sqr_eva0 = sqrt(eva0);                                 // 最小特征值的平方根
        Eigen::Vector3d center_turb = center + 5 * sqr_eva0 * direct; // 体素中心坐标加上5倍最小特征值的平方根乘以特征向量，扰动后的体素中心坐标

        vector<VOX_FACTOR> covMats(8); // 存储中心位置外8个子体素的点云的协方差矩阵
        for (int i = 0; i < win_size; i++)
        {
            // 遍历每个窗口内的点云数据
            for (Eigen::Vector3d ap : vec_tran[i])
            {
                int xyz[3] = {0, 0, 0};
                for (int k = 0; k < 3; k++)
                {
                    // 将每个点云数据的坐标与体素中心坐标扰动后的坐标进行比较，如果大于体素中心坐标扰动后的坐标，则xyz[k] = 1
                    if (ap(k) > center_turb[k])
                    {
                        xyz[k] = 1;
                    }
                }

                Eigen::Vector3d pvec(ap(0), ap(1), ap(2)); // 体素内的点云数据

                int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2]; // 体素的索引
                covMats[leafnum].push(pvec);
            }
        }

        // 根据子体素特征值判断父体素的特征
        double ratios[2] = {1.0 / (3.0 * 3.0), 2.0 * 2.0}; // 特征值比值的范围
        int num_all = 0, num_qua = 0;
        for (int i = 0; i < 8; i++)
        {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMats[i].cov()); // 计算8个子体素内的协方差矩阵的特征值
            double child_eva0 = (saes.eigenvalues()[0]);                           // 子体素内的最小特征值
            if (child_eva0 > ratios[0] * eva0 && child_eva0 < ratios[1] * eva0)    // 如果子体素内的最小特征值在特定范围内
            {
                num_qua++;
            }
            num_all++;
        }

        double prop = 1.0 * num_qua / num_all; // 比例

        if (prop < 0.5)
        {
            return 0;
        }
        return 1;
    }

    // 在8叉树内划分子体素
    void cut_func(int ci)
    {
        PLV(3) &pvec_orig = vec_orig[ci];
        PLV(3) &pvec_tran = vec_tran[ci];

        // 将8个子体素存进8叉树的子树中
        uint a_size = pvec_tran.size(); // 父体素内的点云数量
        for (uint j = 0; j < a_size; j++)
        {
            int xyz[3] = {0, 0, 0};
            for (uint k = 0; k < 3; k++)
            {
                if (pvec_tran[j](k) > voxel_center[k])
                {
                    xyz[k] = 1;
                }
            }
            int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
            if (leaves[leafnum] == nullptr) // 存子体素的中心点三维坐标
            {
                leaves[leafnum] = new OCTO_TREE_NODE(win_size, eigen_thr);
                leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
                leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
                leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
                leaves[leafnum]->quater_length = quater_length / 2.0; // 子体素的四分之一边长是父体素的二分之一
                leaves[leafnum]->layer = layer + 1;                   // 子体素的层数是父体素的层数加1, 递归
            }

            leaves[leafnum]->vec_orig[ci].push_back(vec_orig[ci][j]); // 将父体素内的点云数据存进子体素中
            leaves[leafnum]->vec_tran[ci].push_back(vec_tran[ci][j]);

            leaves[leafnum]->sig_orig[ci].push(pvec_orig[j]); // 将父体素内的体素因子存进子体素中
            leaves[leafnum]->sig_tran[ci].push(pvec_tran[j]);
        }

        PLV(3)
        ().swap(pvec_orig);
        PLV(3)
        ().swap(pvec_tran);
    }

    void recut()
    {
        if (octo_state == UNKNOWN)
        {
            int point_size = 0;
            for (int i = 0; i < win_size; i++)
            {
                point_size += sig_orig[i].N;
            }

            if (point_size < MIN_PT) // 如果体素内的点云数量小于15，则将八叉树状态设置为中间节点, 这个节点的数据就直接丢弃了
            {
                octo_state = MID_NODE;
                vector<PLV(3)>().swap(vec_orig); // 释放内存
                vector<PLV(3)>().swap(vec_tran);
                vector<VOX_FACTOR>().swap(sig_orig);
                vector<VOX_FACTOR>().swap(sig_tran);
                return;
            }

            if (judge_eigen())
            {
                octo_state = PLANE;
#ifndef ENABLE_FILTER
#ifndef ENABLE_RVIZ
                vector<PLV(3)>().swap(vec_orig);
                vector<PLV(3)>().swap(vec_tran);
#endif
#endif
                return;
            }

            else
            {
                if (layer == layer_limit)
                {
                    octo_state = MID_NODE;
                    vector<PLV(3)>().swap(vec_orig);
                    vector<PLV(3)>().swap(vec_tran);
                    vector<VOX_FACTOR>().swap(sig_orig);
                    vector<VOX_FACTOR>().swap(sig_tran);
                    return;
                }
                vector<VOX_FACTOR>().swap(sig_orig);
                vector<VOX_FACTOR>().swap(sig_tran);
                for (int i = 0; i < win_size; i++)
                {
                    cut_func(i);
                }
            }
        }
        for (int i = 0; i < 8; i++)
        {
            if (leaves[i] != nullptr)
            {
                leaves[i]->recut();
            }
        }
    }

    // 体素优化
    void tras_opt(VOX_HESS &vox_opt)
    {
        // 如果当前节点的状态是平面特征
        if (octo_state == PLANE)
        {
            // 将当前节点的体素因子和点云信息传递到体素优化器中
            vox_opt.push_voxel(&sig_orig, &vec_orig);
        }
        else
        {
            // 遍历当前节点的8个子节点
            for (int i = 0; i < 8; i++)
            {
                // 如果子节点不为空
                if (leaves[i] != nullptr)
                {
                    // 对子节点递归调用tras_opt函数
                    leaves[i]->tras_opt(vox_opt);
                }
            }
        }
    }

    // 展示体素？
    void tras_display(int layer = 0)
    {
        float ref = 255.0 * rand() / (RAND_MAX + 1.0f);
        pcl::PointXYZINormal ap; // 数据结构有x,y,z,intensity, norm(), curvature()
        ap.intensity = ref;      // 体素随机着色

        if (octo_state == PLANE)
        {
            // std::vector<unsigned int> colors;
            // colors.push_back(static_cast<unsigned int>(rand() % 256));
            // colors.push_back(static_cast<unsigned int>(rand() % 256));
            // colors.push_back(static_cast<unsigned int>(rand() % 256));
            pcl::PointCloud<pcl::PointXYZINormal> color_cloud;

            for (int i = 0; i < win_size; i++) // 遍历窗口
            {
                // 遍历体素内的点云数据
                for (size_t j = 0; j < vec_tran[i].size(); j++)
                {
                    Eigen::Vector3d &pvec = vec_tran[i][j]; // 第i个体素内的第j个点云数据
                    ap.x = pvec.x();
                    ap.y = pvec.y();
                    ap.z = pvec.z();
                    // ap.b = colors[0];
                    // ap.g = colors[1];
                    // ap.r = colors[2];
                    ap.normal_x = sqrt(value_vector[1] / value_vector[0]);
                    ap.normal_y = sqrt(value_vector[2] / value_vector[0]);
                    ap.normal_z = sqrt(value_vector[0]);
                    // ap.curvature = total;
                    color_cloud.push_back(ap);
                }
            }

#ifdef ENABLE_RVIZ
            sensor_msgs::PointCloud2 dbg_msg;
            pcl::toROSMsg(color_cloud, dbg_msg);
            dbg_msg.header.frame_id = "camera_init";
            pub_residual.publish(dbg_msg);

            visualization_msgs::Marker marker;
            visualization_msgs::MarkerArray marker_array;
            marker.header.frame_id = "camera_init";
            marker.header.stamp = ros::Time::now();
            marker.ns = "basic_shapes";
            marker.id = BINGO_CNT;
            BINGO_CNT++;
            marker.action = visualization_msgs::Marker::ADD;
            marker.type = visualization_msgs::Marker::ARROW;
            marker.color.a = 1;
            marker.color.r = layer == 0 ? 1 : 0;
            marker.color.g = layer == 1 ? 1 : 0;
            marker.color.b = layer == 2 ? 1 : 0;
            marker.scale.x = 0.01;
            marker.scale.y = 0.05;
            marker.scale.z = 0.05;
            marker.lifetime = ros::Duration();
            geometry_msgs::Point apoint;
            apoint.x = center(0);
            apoint.y = center(1);
            apoint.z = center(2);
            marker.points.push_back(apoint);
            apoint.x += 0.2 * direct(0);
            apoint.y += 0.2 * direct(1);
            apoint.z += 0.2 * direct(2);
            marker.points.push_back(apoint);
            marker_array.markers.push_back(marker);
            pub_direct.publish(marker_array);
#endif
        }
        else
        {
            if (layer == layer_limit)
            {
                return;
            }
            layer++;
            for (int i = 0; i < 8; i++)
            {
                if (leaves[i] != nullptr)
                {
                    leaves[i]->tras_display(layer);
                }
            }
        }
    }
};

// 根节点
class OCTO_TREE_ROOT : public OCTO_TREE_NODE
{
public:
    OCTO_TREE_ROOT(int _winsize, float _eigen_thr) : OCTO_TREE_NODE(_winsize, _eigen_thr) {}
};

class VOX_OPTIMIZER
{
public:
    int win_size, jac_leng, imu_leng;
    VOX_OPTIMIZER(int _win_size = WIN_SIZE) : win_size(_win_size)
    {
        // 体素优化器的雅各布矩阵维度和IMU维度
        jac_leng = DVEL * win_size;
        imu_leng = DIM * win_size;
    }

    /// @brief 分割阈值的计算，这个函数是用于计算分割阈值的，这个分割阈值是用于去除离群点的，这个函数的实现是遍历所有体素，然后计算体素内所有窗口的残差，如果有一个窗口的残差大于阈值，则去除这个体素
    /// @param x_stats 
    /// @param voxhess 
    /// @param x_ab 
    /// @param Hess 
    /// @param JacT 
    /// @return 
    double divide_thread(vector<IMUST> &x_stats, VOX_HESS &voxhess, vector<IMUST> &x_ab, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
    {
        double residual = 0;
        Hess.setZero();
        JacT.setZero();
        PLM(-1) hessians(thd_num);//thd_num = 16
        PLV(-1) jacobins(thd_num);

        for (int i = 0; i < thd_num; i++)
        {
            hessians[i].resize(jac_leng, jac_leng);
            jacobins[i].resize(jac_leng);
        }

        int tthd_num = thd_num;
        vector<double> resis(tthd_num, 0);// 存储每个线程的残差
        int g_size = voxhess.plvec_voxels.size();// 体素的数量
        if (g_size < tthd_num)
            tthd_num = 1;

        vector<thread *> mthreads(tthd_num);
        double part = 1.0 * g_size / tthd_num; // 每个线程处理的体素数量
        // 开启线程计算每个体素的残差，雅各布矩阵和Hessian矩阵, 然后所有体素的残差，雅各布矩阵和Hessian矩阵累加得到总的Hessian矩阵，雅各布矩阵和残差
        for (int i = 0; i < tthd_num; i++) 
        {
            mthreads[i] = new thread(&VOX_HESS::acc_evaluate2, &voxhess, x_stats, part * i, part * (i + 1),
                                     ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));
        }

        // 等待所有线程结束，更新Hessian矩阵，雅各布矩阵和残差
        for (int i = 0; i < tthd_num; i++)
        {
            mthreads[i]->join();
            Hess += hessians[i];
            JacT += jacobins[i];
            residual += resis[i];
            delete mthreads[i];
        }
#ifdef AVG_THR
        return residual / g_size;
#else
        return residual;
#endif
    }

    double only_residual(vector<IMUST> &x_stats, VOX_HESS &voxhess, vector<IMUST> &x_ab, bool is_avg = false)
    {
        double residual2 = 0;
        voxhess.evaluate_only_residual(x_stats, residual2);
        if (is_avg)
            return residual2 / voxhess.plvec_voxels.size();
        return residual2;
    }

    // 去除离群点
    void remove_outlier(vector<IMUST> &x_stats, VOX_HESS &voxhess, double ratio)
    {
        // 计算残差
        std::vector<double> residuals = voxhess.evaluate_residual(x_stats);

        // 对残差进行升序排序
        std::sort(residuals.begin(), residuals.end());

        // 计算阈值
        double threshold = residuals[std::floor((1 - ratio) * voxhess.plvec_voxels.size()) - 1];

        // 计算需要移除的离群点数量
        int reject_num = std::floor(ratio * voxhess.plvec_voxels.size());

        // 移除离群点
        voxhess.remove_residual(x_stats, threshold, reject_num);
    }

    /// @brief 阻尼迭代，计算当前的残差，内存成本和hessian矩阵，见formula3.png
    /// @param x_stats
    /// @param voxhess
    /// @param residual
    /// @param
    /// @param mem_cost
    void damping_iter(vector<IMUST> &x_stats, VOX_HESS &voxhess, double &residual, PLV(6) & hess_vec, size_t &mem_cost)
    {
        double u = 0.01, v = 2;
        Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng),
            HessuD(jac_leng, jac_leng);
        Eigen::VectorXd JacT(jac_leng), dxi(jac_leng), new_dxi(jac_leng);

        D.setIdentity();
        double residual1, residual2, q;
        bool is_calc_hess = true;
        vector<IMUST> x_stats_temp;

        vector<IMUST> x_ab(win_size);
        x_ab[0] = x_stats[0];
        for (int i = 1; i < win_size; i++)
        {
            x_ab[i].p = x_stats[i - 1].R.transpose() * (x_stats[i].p - x_stats[i - 1].p);
            x_ab[i].R = x_stats[i - 1].R.transpose() * x_stats[i].R;
        }

        double hesstime = 0;
        double solvtime = 0;
        size_t max_mem = 0;
        double loop_num = 0;
        for (int i = 0; i < 10; i++)
        {
            if (is_calc_hess)
            {
                double tm = ros::Time::now().toSec();
                residual1 = divide_thread(x_stats, voxhess, x_ab, Hess, JacT);
                hesstime += ros::Time::now().toSec() - tm;
            }

            double tm = ros::Time::now().toSec();
            D.diagonal() = Hess.diagonal();
            HessuD = Hess + u * D;
            double t1 = ros::Time::now().toSec();
            Eigen::SparseMatrix<double> A1_sparse(jac_leng, jac_leng);
            std::vector<Eigen::Triplet<double>> tripletlist;
            for (int a = 0; a < jac_leng; a++)
                for (int b = 0; b < jac_leng; b++)
                    if (HessuD(a, b) != 0)
                    {
                        tripletlist.push_back(Eigen::Triplet<double>(a, b, HessuD(a, b)));
                        // A1_sparse.insert(a, b) = HessuD(a, b);
                    }
            A1_sparse.setFromTriplets(tripletlist.begin(), tripletlist.end());
            A1_sparse.makeCompressed();
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Solver_sparse;
            Solver_sparse.compute(A1_sparse);
            size_t temp_mem = check_mem();
            if (temp_mem > max_mem)
                max_mem = temp_mem;
            dxi = Solver_sparse.solve(-JacT);
            temp_mem = check_mem();
            if (temp_mem > max_mem)
                max_mem = temp_mem;
            solvtime += ros::Time::now().toSec() - tm;
            // new_dxi = Solver_sparse.solve(-JacT);
            // printf("new solve time cost %f\n",ros::Time::now().toSec() - t1);
            // relative_err = ((Hess + u*D)*dxi + JacT).norm()/JacT.norm();
            // absolute_err = ((Hess + u*D)*dxi + JacT).norm();
            // std::cout<<"relative error "<<relative_err<<std::endl;
            // std::cout<<"absolute error "<<absolute_err<<std::endl;
            // std::cout<<"delta x\n"<<(new_dxi-dxi).transpose()/dxi.norm()<<std::endl;

            x_stats_temp = x_stats;
            for (int j = 0; j < win_size; j++)
            {
                x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DVEL * j, 0));
                x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DVEL * j + 3, 0);
            }

            double q1 = 0.5 * dxi.dot(u * D * dxi - JacT);
#ifdef AVG_THR
            residual2 = only_residual(x_stats_temp, voxhess, x_ab, true);
            q1 /= voxhess.plvec_voxels.size();
#else
            residual2 = only_residual(x_stats_temp, voxhess, x_ab);
#endif
            residual = residual2;
            q = (residual1 - residual2);
            // printf("iter%d: (%lf %lf) u: %lf v: %lf q: %lf %lf %lf\n",
            //        i, residual1, residual2, u, v, q/q1, q1, q);
            loop_num = i + 1;
            // if(hesstime/loop_num > 1) printf("Avg. Hessian time: %lf ", hesstime/loop_num);
            // if(solvtime/loop_num > 1) printf("Avg. solve time: %lf\n", solvtime/loop_num);
            // if(double(max_mem/1048576.0) > 2.0) printf("Max mem: %lf\n", double(max_mem/1048576.0));

            if (q > 0)
            {
                x_stats = x_stats_temp;
                q = q / q1;
                v = 2;
                q = 1 - pow(2 * q - 1, 3);
                u *= (q < one_three ? one_three : q);
                is_calc_hess = true;
            }
            else
            {
                u = u * v;
                v = 2 * v;
                is_calc_hess = false;
            }
#ifdef AVG_THR
            if ((fabs(residual1 - residual2) / residual1) < 0.05 || i == 9)
            {
                if (mem_cost < max_mem)
                    mem_cost = max_mem;
                for (int j = 0; j < win_size - 1; j++)
                    for (int k = j + 1; k < win_size; k++)
                        hess_vec.push_back(Hess.block<DVEL, DVEL>(DVEL * j, DVEL * k).diagonal().segment<DVEL>(0));
                break;
            }
#else
            if (fabs(residual1 - residual2) < 1e-9)
                break;
#endif
        }
    }

    //
    size_t check_mem()
    {
        FILE *file = fopen("/proc/self/status", "r");
        int result = -1;
        char line[128];

        while (fgets(line, 128, file) != nullptr)
        {
            if (strncmp(line, "VmRSS:", 6) == 0)
            {
                int len = strlen(line);

                const char *p = line;
                for (; std::isdigit(*p) == false; ++p)
                {
                }

                line[len - 3] = 0;
                result = atoi(p);

                break;
            }
        }
        fclose(file);

        return result;
    }
};

#endif