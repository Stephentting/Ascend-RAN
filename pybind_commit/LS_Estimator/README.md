# Ascend Custom LS Channel Estimation Operator

## 1. 算子简介
本样例实现了一个基于 **Ascend C** 的 OFDM 系统 LS（Least Square，最小二乘）信道估计与插值算子。

与传统的基于循环或 FFT 的插值实现不同，本算子采用了 **基于矩阵运算的插值（Matrix-based Interpolation）** 方案。通过离线预计算插值权重矩阵，将复杂的信道恢复过程转化为一次高效的矩阵乘法（Matmul），充分利用了 NPU Cube 单元的高计算密度特性，极大地提升了信道估计的吞吐率。

### 1.1 数学原理
算子核心算法利用线性变换将导频处的信道响应映射至所有数据子载波。

假设 $H_{p}$ 为导频位置的信道估计向量（输入），$W$ 为预计算的插值权重矩阵，$H_{all}$ 为所有子载波的完整信道响应（输出）。

#### 步骤 1：复数域线性变换
信道插值本质上是导频信号的加权求和。通过矩阵乘法直接从导频向量恢复完整频域响应：

$$
H_{all} = H_{p} \times W_{interp}
$$

* **代码对应**：核心 Matmul 接口调用 (`mm.IterateAll`)。
* **物理含义**：$W_{interp}$ 矩阵内部隐含了导频位置信息以及插值算法（如线性插值、DFT 插值或 MMSE 滤波系数）。

#### 步骤 2：实数域维度映射 (Real-Domain Mapping)
由于 NPU 的矩阵乘法器原生处理实数，本算子采用 **实部虚部平铺（Flattened Real-Imaginary）** 的方式处理复数信号。

假设导频数量为 $N_p$ (16)，子载波数量为 $N_c$ (256)。
* **输入 A (导频)**：维度为 $[Batch, 2N_p]$。即 $[Batch, 32]$，存储格式为 $[Re_0, Im_0, Re_1, Im_1, ...]$。
* **输入 B (权重)**：维度为 $[2N_p, 2N_c]$。即 $[32, 512]$。该矩阵通过特殊的块状结构设计，一次性完成复数乘法运算：
  $$
  \begin{bmatrix} Re(W) & Im(W) \\ -Im(W) & Re(W) \end{bmatrix}
  $$
* **输出 C (信道)**：维度为 $[Batch, 2N_c]$。即 $[Batch, 512]$，直接输出完整的复数信道响应。

---

## 2. 工程目录结构
```text
LDPC_Encoder
├── cmake/                      # 编译工程配置文件
编译工程文件
├── CMakeLists.txt              // 编译工程文件
算子实现
├── matmul_LS_custom.cpp          # 算子 Kernel 核心实现 (Ascend C)
├── matmul_custom_tiling.cpp   # 算子 Tiling 策略实现
封装以及调用测试
├── pybind11.cpp                # Python C++ 接口封装 (Pybind11)
├── test_matmul_pybind.py    # Python 端测试脚本
└── run_pybind.sh               # 自动化编译与运行脚本
```
### 说明  
- 可以直接运行 bash run_pybind.sh 