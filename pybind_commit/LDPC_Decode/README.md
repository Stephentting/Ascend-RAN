# Ascend Custom LDPC Decoder Operator

## 1. 算子简介
本样例实现了一个基于 **Ascend C** 的 LDPC 硬判决译码算子（Hard-Decision Bit-Flipping Decoder）。

与传统实现不同，本算子采用了 **Kernel 内循环（Loop-in-Kernel）** 机制，将 LDPC 译码的多次迭代逻辑完全下沉至 NPU 侧执行。Host 侧只需启动一次 Task 即可完成完整的译码过程，极大地减少了 Host-Device 之间的交互开销，提升了端到端性能。

### 1.1 数学原理
算子核心算法为 **加权比特翻转（Weighted Bit-Flipping）**，利用 NPU 的矩阵乘法单元加速校验子计算与投票过程。

假设 $U$ 为待译码的比特序列矩阵（形状 $[M, N]$），$H$ 为 LDPC 校验矩阵。单次迭代包含以下三个步骤：

#### 步骤 1：伴校征计算 (Syndrome Calculation)
首先验证当前码字是否满足校验方程。通过矩阵乘法计算伴校征向量 $S$：

$$
S = (U \times H^T) \mod 2
$$

* **代码对应**：第一次 Matmul (`mm`) + 模2运算。
* **物理含义**：$S_j = 1$ 表示第 $j$ 个校验方程未被满足。若 $\sum S = 0$，则译码成功，提前终止。

#### 步骤 2：违规投票 (Voting / Correlation)
计算每个比特位参与了多少个“未满足的校验方程”。这通过将伴校征 $S$ 反向投影回变量节点实现：

$$
V_{otes} = S \times H
$$

* **代码对应**：第二次 Matmul (`mm2`)，其中 TensorB 进行了转置处理。
* **物理含义**：$V_{otes}[i]$ 的值越大，表示第 $i$ 个比特导致校验错误的概率越高。

#### 步骤 3：比特翻转决策 (Bit Flipping)
在 Vector 单元执行判决逻辑：
1. **寻找最大票数**：计算当前行的最大违规数 $V_{max} = \max(V_{otes})$。
2. **翻转比特**：对于所有满足 $V_{otes}[i] == V_{max}$ 且 $V_{max} > T_{hresh}$ 的比特位置，对其值进行翻转 ($0 \leftrightarrow 1$)。

---

## 2. 工程目录结构
```text
LDPC_Encoder
├── cmake/                      # 编译工程配置文件
编译工程文件
├── CMakeLists.txt              // 编译工程文件
LDPC编码矩阵以及校验矩阵
├── matrix_G_padded_256x512.bin    # LDPC编码矩阵
└── matrix_H_transposed_padded_512x256.bin               # LDPC校验矩阵
算子实现
├── LDPCDec.cpp          # 算子 Kernel 核心实现 (Ascend C)
├── LDPCDec.cpp   # 算子 Tiling 策略实现
封装以及调用测试
├── pybind11.cpp                # Python C++ 接口封装 (Pybind11)
├── test_ldpc_pybind.py    # Python 端测试脚本
└── run_pybind.sh               # 自动化编译与运行脚本
```

### 说明  
- 这次把20次迭代直接放到kernel中去完成了，在host侧调用算子就调用一次就行了。
- 如果要迭代10次就去kernel中改。
- 算子文件：LDPCDec_loopInKernel.cpp
- 可以直接运行 bash run_pybind.sh 