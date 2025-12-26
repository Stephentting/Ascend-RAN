# Ascend Custom QAM64 Demodulation Operator (Demapper)

## 1. 算子简介
本样例实现了一个基于 **Ascend C** 的 QAM64（64-Quadrature Amplitude Modulation）硬判决解调算子。

该算子接收接收端的 I 路和 Q 路浮点信号，采用 **最小欧氏距离法则（Minimum Euclidean Distance）** 进行判决，将模拟信号映射回二进制比特流。为了处理非对齐的数据长度，算子实现了 **Tiling（分块处理）** 与 **Tail Handling（尾部标量处理）** 相结合的策略，确保了数据的完整性与计算的正确性。

### 1.1 数学原理
QAM64 解调的核心是寻找接收符号在星座图上距离最近的标准点，并输出该点对应的比特索引。

#### 步骤 1：电平恢复与距离计算
接收到的信号 $R$（包含 $R_I$ 和 $R_Q$）首先与 8 个标准星座点电平进行比较。
标准电平集合 $L$ 由下式生成（归一化因子 $K = \frac{1}{\sqrt{42}} \approx 0.1543$）：

$$
L_i = (-7 + 2 \times i) \times K, \quad i \in \{0, 1, ..., 7\}
$$

对于每一个接收符号，算子分别计算 I 路和 Q 路到这 8 个电平的绝对距离：

$$
D_{I, i} = |R_I - L_i| \\
D_{Q, i} = |R_Q - L_i|
$$

#### 步骤 2：硬判决 (Hard Decision)
根据最小距离准则，找到最接近的电平索引：

$$
Idx_I = \underset{i}{\arg\min}(D_{I, i}) \\
Idx_Q = \underset{i}{\arg\min}(D_{Q, i})
$$

#### 步骤 3：比特映射 (Bit Mapping)
将找到的十进制索引 $Idx$（范围 0-7）直接转换为 3 位二进制数据。
> **注意**：本算子采用 **自然二进制映射 (Natural Binary Mapping)**，即索引值直接对应比特值（例如索引 2 对应 `010`）。

* **输出 I 路比特**：$Idx_I$ 的高、中、低位对应符号的前 3 个比特。
* **输出 Q 路比特**：$Idx_Q$ 的高、中、低位对应符号的后 3 个比特。
* **最终输出**：每个符号输出 6 个 `uint8` 类型的比特（0 或 1）。

---

## 2. 工程目录结构
```text
QAM64_Demodulator
├── cmake/                      # 编译工程配置文件
├── CMakeLists.txt              # 编译工程文件
算子实现
├── qamdemapper_custom.cpp            # 算子 Kernel 核心实现 (Ascend C)
封装以及调用测试
├── pybind11.cpp                # Python C++ 接口封装 (Pybind11)
├── test_qam_demod.py        # Python 端测试脚本 (含正确性验证)
└── run_pybind.sh               # 自动化编译与运行脚本
```
### 说明  
- 可以直接运行 bash run_pybind.sh 