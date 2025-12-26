# Ascend Custom QAM64 Modulation Operator

## 1. 算子简介
本样例实现了一个基于 **Ascend C** 的 QAM64（Quadrature Amplitude Modulation，64正交振幅调制）调制算子。

算子采用了 **向量化查表（Vectorized Look-Up Table）** 的高效实现方案。利用 NPU 的 Vector 计算单元，并行地将输入的二进制比特流映射为复数星座点。通过将 Gray 码映射逻辑固化为 Local Memory 中的查找表，避免了复杂的条件分支判断（If-Else），极大提升了符号生成的吞吐率。

### 1.1 数学原理
QAM64 调制的核心是将每 6 个输入比特映射为一个复数符号 $S = I + jQ$。

#### 步骤 1：比特分组 (Bit Grouping)
输入数据为 `uint8` 类型的比特流（0 或 1）。代码中将每 6 个连续比特划分为一个符号（Symbol）：
* **高 3 位** ($b_0, b_1, b_2$)：决定同相分量（In-phase, **Real** 部分）。
* **低 3 位** ($b_3, b_4, b_5$)：决定正交分量（Quadrature, **Imag** 部分）。

#### 步骤 2：Gray 码映射 (Standard Gray Mapping)
算子内部维护了一个长度为 8 的查找表（LUT），实现了标准的 Gray 码映射逻辑。3 位二进制输入索引与幅度电平的对应关系如下：

| 输入比特 (Binary) | 十进制索引 | 原始电平 (Level) | 归一化输出 ($Level \times K$) |
| :--- | :--- | :--- | :--- |
| **000** | 0 | -7 | $-7 \times K$ |
| **001** | 1 | -5 | $-5 \times K$ |
| **010** | 2 | -1 | $-1 \times K$ |
| **011** | 3 | -3 | $-3 \times K$ |
| **100** | 4 | +7 | $+7 \times K$ |
| **101** | 5 | +5 | $+5 \times K$ |
| **110** | 6 | +1 | $+1 \times K$ |
| **111** | 7 | +3 | $+3 \times K$ |

> **归一化说明**：为了保证星座图的平均功率为 1，所有电平值均乘以归一化因子 $K = \frac{1}{\sqrt{42}} \approx 0.15430335$。

#### 步骤 3：输出格式
最终输出被组织为两个独立的 Tensor，分别存储实部（Real）和虚部（Imag），数据类型为 `half` (FP16)。

---

## 2. 工程目录结构
```text
QAM64_Modulator
├── cmake/                      # 编译工程配置文件
├── CMakeLists.txt              # 编译工程文件
算子实现
├── qam64_modulation_vectorized.cpp # 算子 Kernel 核心实现 (Ascend C)
封装以及调用测试
├── pybind11.cpp                # Python C++ 接口封装 (Pybind11)
├── test_qam64_mod.py       # Python 端测试脚本 (含正确性验证)
└── run_pybind.sh               # 自动化编译与运行脚本
```
### 说明  
- 可以直接运行 bash run_pybind.sh 