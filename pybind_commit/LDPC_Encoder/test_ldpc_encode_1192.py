import torch
import torch_npu
import numpy as np
import ldpc_encode_custom
import os
import time

def test_aggregation_12():
    device = "npu:0"
    M, K, N = 256, 256, 512
    num_chunks = 12

    # 1. 加载数据
    x1_single = np.fromfile("../input/x1_gm.bin", dtype=np.int8)
    x1_all = np.tile(x1_single, num_chunks).reshape(num_chunks * M, K)
    x2_g = np.fromfile("../input/x2_gm.bin", dtype=np.int8).reshape(K, N)
    
    golden_single = np.fromfile("../output/golden.bin", dtype=np.int16).reshape(M, N)
    golden_all = np.tile(golden_single, (num_chunks, 1))

    # 2. 搬运到 NPU
    bits_in = torch.from_numpy(x1_all).to(device).contiguous()
    h_matrix = torch.from_numpy(x2_g).to(device).contiguous()

    # --- 性能测试部分 ---
    
    warmup_iters = 10    # 预热次数
    test_iters = 100     # 正式测试次数

    print(f"🔥 正在预热 ({warmup_iters} 次)...")
    for _ in range(warmup_iters):
        _ = ldpc_encode_custom.run_ldpc_encode(bits_in, h_matrix)
    
    # 强制同步确保预热完成
    torch.npu.synchronize()

    print(f"⏱️ 正在进行性能测试 ({test_iters} 次)...")
    
    # 使用 NPU Event 进行高精度计时
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    start_event.record()
    for _ in range(test_iters):
        output_npu = ldpc_encode_custom.run_ldpc_encode(bits_in, h_matrix)
    end_event.record()

    # 等待所有任务完成
    torch.npu.synchronize()

    # 计算总耗时 (单位：毫秒 ms)
    total_latency_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_latency_ms / test_iters

    print("\n" + "="*30)
    print(f"性能分析报告 (Device: {torch.npu.get_device_name(0)})")
    print(f"单次算子平均耗时: {avg_latency_ms:.4f} ms")
    print(f"每秒推理次数 (TPS): {1000/avg_latency_ms:.2f}")
    print("="*30 + "\n")

    # --- 结果比对部分 ---
    print(f"🚀 正在验证结果一致性...")
    res = output_npu.cpu().numpy()
    error_mask = (res != golden_all)
    total_errors = np.sum(error_mask)
    
    print(f"NPU 输出形状: {res.shape}")
    print(f"总错误点数: {total_errors} / {res.size}")

    if total_errors == 0:
        print("✅ [Success] 全链路 12 组聚合验证通过！")
    else:
        first_err = np.where(error_mask.flatten())[0][0]
        r, c = divmod(first_err, N)
        print(f"❌ 首次错误发生在: 第 {r//256} 组, 行 {r%256}, 列 {c}")

if __name__ == "__main__":
    test_aggregation_12()