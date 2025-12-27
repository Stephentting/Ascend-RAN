# Ascend Custom Zero-Forcing (ZF) Equalization Operator

## 1. 算子简介
本样例实现了一个基于 **Ascend C** 的 ZF（Zero-Forcing，迫零）信道均衡算子。

该算子用于 OFDM 通信接收机中，通过简单的复数除法运算来抵消信道衰落对信号的影响。算子利用 NPU 的 Vector 计算单元并行处理复数运算，实现了高吞吐率的信道均衡。

### 1.1 数学原理
ZF 均衡的核心思想是直接通过信道估计值 $H$ 的逆来恢复发送信号 $X$。

假设接收信号模型为（忽略噪声项）：
$$Y = H \cdot X$$
其中 $Y$ 是接收信号，$H$ 是信道频率响应，$X$ 是发送信号。

ZF 均衡器的估计输出 $\hat{X}$ 计算公式为：
$$\hat{X} = \frac{Y}{H}$$

由于输入均为复数，算子内部执行了 **复数除法** 运算。
设 $Y = Y_r + jY_i$，$H = H_r + jH_i$，则：

$$
\hat{X} = \frac{Y_r + jY_i}{H_r + jH_i} = \frac{(Y_r + jY_i)(H_r - jH_i)}{H_r^2 + H_i^2}
$$

**展开后实部与虚部计算如下：**

1.  **分母（功率归一化项）**：
    $$|H|^2 = H_r^2 + H_i^2 + \epsilon$$
    > **注**：代码中加入了 $\epsilon = 10^{-6}$ 用于防止除以零的数值不稳定性。

2.  **实部输出 ($\hat{X}_{real}$)**：
    $$\hat{X}_{real} = \frac{H_r Y_r + H_i Y_i}{|H|^2}$$

3.  **虚部输出 ($\hat{X}_{imag}$)**：
    $$\hat{X}_{imag} = \frac{H_r Y_i - H_i Y_r}{|H|^2}$$

---

## 2. 工程目录结构
```text
ZF_Equalizer
├── cmake/                      # 编译工程配置文件
├── CMakeLists.txt              # 编译工程文件
算子实现
├── zf_equalization.cpp         # 算子 Kernel 核心实现 (Ascend C)
封装以及调用测试
├── pybind11.cpp                # Python C++ 接口封装 (Pybind11)
├── zf_test.py           # Python 端测试脚本 (含正确性验证)
└── run_pybind.sh               # 自动化编译与运行脚本
```
### 说明  
- 可以直接运行 bash run_pybind.sh 
- 验证结果显示因为精度问题会有最大约0.02的误差，这相对于64qam的最小欧式距离0.308来说有很大的容错空间。