#!/usr/bin/env python3
# coding=utf-8
import numpy as np
import os

# ==================== 1. 系统参数配置 ====================
N_SUBCARRIERS = 256
BATCH_SIZE = 1192

# 硬件子载波结构定义
GUARD_LEFT = list(range(0, 7))
GUARD_RIGHT = list(range(250, 256))
DC_CARRIERS = list(range(125, 132))
ZERO_CARRIERS = set(GUARD_LEFT + GUARD_RIGHT + DC_CARRIERS)

# 导频位置计算 (128 为中心)
PILOT_OFFSETS = [117, 103, 89, 75, 57, 39, 25, 11]
PILOT_INDICES = sorted([128 - x for x in PILOT_OFFSETS] + [128 + x for x in PILOT_OFFSETS])
N_PILOTS = len(PILOT_INDICES)

# 用户指定的 QPSK 导频符号序列 (按顺序对应 16 个导频位置)
QPSK_PILOT_VALUES = np.array([
    0.7071 + 0.7071j,  # Pos 11 (对应原表 0)
    0.7071 - 0.7071j,  # Pos 25 (对应原表 16)
    -0.7071 + 0.7071j, # Pos 39
    -0.7071 - 0.7071j, # Pos 53
    0.7071 + 0.7071j,  # Pos 71
    -0.7071 - 0.7071j, # Pos 89
    0.7071 - 0.7071j,  # Pos 103
    -0.7071 + 0.7071j, # Pos 117
    -0.7071 - 0.7071j, # Pos 139
    0.7071 + 0.7071j,  # Pos 153
    -0.7071 + 0.7071j, # Pos 167
    0.7071 - 0.7071j,  # Pos 185
    0.7071 + 0.7071j,  # Pos 203
    -0.7071 + 0.7071j, # Pos 217
    0.7071 - 0.7071j,  # Pos 231
    -0.7071 - 0.7071j  # Pos 245
], dtype=np.complex64)

# ==================== 2. 工具函数 ====================

def generate_64qam_constellation():
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    constellation = np.array([complex(i, q) for i in levels for q in levels])
    return constellation / np.sqrt(np.mean(np.abs(constellation)**2))

def build_compact_matrix():
    """构建针对非等间距导频的插值估计矩阵 M"""
    matrix = np.zeros((N_PILOTS, N_SUBCARRIERS), dtype=np.complex64)
    
    for k in range(N_SUBCARRIERS):
        if k in ZERO_CARRIERS: continue
        
        # 寻找相邻导频进行线性插值
        idx = np.searchsorted(PILOT_INDICES, k)
        
        if idx == 0: # 左边界外推
            p_idx = 0
            matrix[p_idx, k] = 1.0 / QPSK_PILOT_VALUES[p_idx]
        elif idx == N_PILOTS: # 右边界外推
            p_idx = N_PILOTS - 1
            matrix[p_idx, k] = 1.0 / QPSK_PILOT_VALUES[p_idx]
        else: # 线性插值
            l_p, r_p = PILOT_INDICES[idx-1], PILOT_INDICES[idx]
            weight_r = (k - l_p) / (r_p - l_p)
            weight_l = 1.0 - weight_r
            matrix[idx-1, k] = weight_l * (1.0 / QPSK_PILOT_VALUES[idx-1])
            matrix[idx, k] = weight_r * (1.0 / QPSK_PILOT_VALUES[idx])
    return matrix

def complex_to_real_block(m_complex):
    """[M, N] complex -> [2M, 2N] real (Standard for NPU Matmul)"""
    M, N = m_complex.shape
    res = np.zeros((2*M, 2*N), dtype=np.float32)
    R, I = np.real(m_complex), np.imag(m_complex)
    res[0:M, 0:N], res[0:M, N:2*N] = R, -I
    res[M:2*M, 0:N], res[M:2*M, N:2*N] = I, R
    return res.astype(np.float16)

# ==================== 3. 主生成流程 ====================

def run_gen():
    print(f"正在生成适配硬件结构的 OFDM 数据...")
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # 1. 生成发送信号
    tx_signal = np.zeros((BATCH_SIZE, N_SUBCARRIERS), dtype=np.complex64)
    data_pos = [k for k in range(N_SUBCARRIERS) if k not in ZERO_CARRIERS and k not in PILOT_INDICES]
    
    qam64 = generate_64qam_constellation()
    for b in range(BATCH_SIZE):
        tx_signal[b, PILOT_INDICES] = QPSK_PILOT_VALUES
        tx_signal[b, data_pos] = qam64[np.random.randint(0, 64, len(data_pos))]

    # 2. 模拟多径信道 (此处简化为随机信道)
    h_real = np.random.randn(BATCH_SIZE, N_SUBCARRIERS)
    h_imag = np.random.randn(BATCH_SIZE, N_SUBCARRIERS)
    channel = (h_real + 1j*h_imag).astype(np.complex64)
    
    rx_signal = tx_signal * channel + (np.random.randn(BATCH_SIZE, N_SUBCARRIERS)*0.01).astype(np.complex64)

    # 3. 提取导频并转为实数格式 [32, 32]
    pilots_rx = rx_signal[:, PILOT_INDICES]
    pilots_real = np.concatenate([np.real(pilots_rx), np.imag(pilots_rx)], axis=1).astype(np.float16)

    # 4. 构建紧凑矩阵并转为实数块 [32, 512]
    m_complex = build_compact_matrix()
    m_real = complex_to_real_block(m_complex)

    # 5. 计算 Golden 参考结果
    golden_real = (pilots_real.astype(np.float32) @ m_real.astype(np.float32))

    # 6. 保存文件
    pilots_real.tofile("input/x1_gm.bin")
    m_real.tofile("input/x2_gm.bin")
    golden_real.tofile("output/golden.bin")

    print(f"成功保存文件:")
    print(f" - 输入数据: input/x1_gm.bin [32, 32]")
    print(f" - 权重矩阵: input/x2_gm.bin [32, 512]")
    print(f" - 参考结果: output/golden.bin [32, 512]")

if __name__ == "__main__":
    run_gen()