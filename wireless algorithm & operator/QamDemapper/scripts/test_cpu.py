import torch
import numpy as np
import time

def test_qam_demod_cpu(num_symbols=262240, iterations=100):
    print(f"🖥️  正在初始化 CPU 数据 (规模: {num_symbols} 符号)...")
    
    # 模拟输入数据 (Float32)
    i_data = torch.randn(num_symbols, dtype=torch.float32)
    q_data = torch.randn(num_symbols, dtype=torch.float32)
    
    # 64-QAM 电平定义 (归一化)
    scale = 6.4807406984
    levels = torch.tensor([-7, -5, -3, -1, 1, 3, 5, 7], dtype=torch.float32) / scale
    
    def run_cpu_logic():
        # 1. 计算距离矩阵 (利用广播机制模拟并行计算)
        # i_data[:, None] 形状为 (N, 1), levels 为 (8,)
        # dists 形状为 (N, 8)
        dist_i = torch.abs(i_data[:, None] - levels)
        dist_q = torch.abs(q_data[:, None] - levels)
        
        # 2. 找到最小距离的索引
        idx_i = torch.argmin(dist_i, dim=1).to(torch.uint8)
        idx_q = torch.argmin(dist_q, dim=1).to(torch.uint8)
        
        # 3. 拆分比特 (模拟 Binary Mapping: 6 bits per symbol)
        # 注意：这是 CPU 向量化实现，比逐个循环快得多
        out = torch.empty((num_symbols, 6), dtype=torch.uint8)
        out[:, 0] = (idx_i >> 2) & 1
        out[:, 1] = (idx_i >> 1) & 1
        out[:, 2] = idx_i & 1
        out[:, 3] = (idx_q >> 2) & 1
        out[:, 4] = (idx_q >> 1) & 1
        out[:, 5] = idx_q & 1
        return out

    # 🔥 预热
    print("🔥 正在预热 CPU...")
    for _ in range(10):
        _ = run_cpu_logic()

    # ⏱️ 性能测试
    print(f"⏱️  开始进行 CPU 性能测试 ({iterations} 次迭代)...")
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        _ = run_cpu_logic()
        
    end_time = time.perf_counter()
    
    # 结果计算
    avg_time_ms = ((end_time - start_time) / iterations) * 1000
    throughput = (num_symbols / 1e6) / (avg_time_ms / 1000)
    
    print("\n" + "="*40)
    print("QAM Demapper CPU 性能报告 (PyTorch-CPU)")
    print(f"输入规模: {num_symbols} 符号")
    print(f"平均单次耗时: {avg_time_ms:.4f} ms")
    print(f"吞吐量: {throughput:.4f} MSymbols/s")
    print("="*40)

if __name__ == "__main__":
    # 使用你 NPU 测试时相同的规模
    test_qam_demod_cpu(num_symbols=262240, iterations=100)