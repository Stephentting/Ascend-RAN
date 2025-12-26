# Ascend Custom LDPC Encoder Operator

## 1. 算子简介
本样例实现了一个基于 **Ascend C** 的 LDPC 编码自定义算子。
算子采用固定维度 `[M, K, N] = [512, 512, 1024]` 进行演示（具体计算逻辑基于下述矩阵形状）。

### 1.1 数学原理
LDPC 编码算子的核心逻辑为矩阵乘法后进行模 2 运算（GF(2) 域）：

$$
C = (A \times B) \mod 2
$$

### 1.2 输入输出说明
| 符号 | 含义 | 形状 (Shape) | 备注 |
| :--- | :--- | :--- | :--- |
| **A** | 信息比特流 (Input) | `[256, 256]` | 每 256 bit 为一个 batch |
| **B** | 生成矩阵 (Generator) | `[256, 512]` | LDPC 编码矩阵 |
| **C** | 编码结果 (Output) | `[256, 512]` | 最终编码输出 |

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
├── LDPCEnc_custom.cpp          # 算子 Kernel 核心实现 (Ascend C)
├── LDPCEnc_custom_tiling.cpp   # 算子 Tiling 策略实现
封装以及调用测试
├── pybind11.cpp                # Python C++ 接口封装 (Pybind11)
├── test_ldpc_encode_1192.py    # Python 端测试脚本
└── run_pybind.sh               # 自动化编译与运行脚本
```
### 说明  
- 可以直接运行 bash run_pybind.sh 