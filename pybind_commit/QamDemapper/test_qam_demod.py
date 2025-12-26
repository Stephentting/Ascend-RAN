import torch
import torch_npu
import numpy as np
import qamdemapper_custom  # 确保编译生成的so在路径下
import time

def test_qam_demod_performance():
    device = "npu:0"
    
    # 根据 main.cpp 的配置：1192 * 220 = 262240 个符号
    TOTAL_ELEMENTS = 1192 * 220
    BITS_PER_SYMBOL = 6
    
    print(f"📦 初始化数据尺寸: {TOTAL_ELEMENTS} 符号...")

    # 1. 构造随机输入数据 (模拟 I/Q 路信号)
    # 假设信号在 [-7, 7] 范围内波动
    input_I_cpu = np.random.uniform(-8, 8, TOTAL_ELEMENTS).astype(np.float32)
    input_Q_cpu = np.random.uniform(-8, 8, TOTAL_ELEMENTS).astype(np.float32)

    # 2. 搬运到 NPU
    input_I_npu = torch.from_numpy(input_I_cpu).to(device).contiguous()
    input_Q_npu = torch.from_numpy(input_Q_cpu).to(device).contiguous()

    # --- 性能测试 ---
    warmup_iters = 20
    test_iters = 100

    print(f"🔥 正在预热 ({warmup_iters} 次)...")
    for _ in range(warmup_iters):
        _ = qamdemapper_custom.run_qam_demod(input_I_npu, input_Q_npu)
    torch.npu.synchronize()

    print(f"⏱️ 正在进行性能测试 ({test_iters} 次)...")
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    start_event.record()
    for _ in range(test_iters):
        output_npu = qamdemapper_custom.run_qam_demod(input_I_npu, input_Q_npu)
    end_event.record()

    torch.npu.synchronize()

    # 计算耗时
    avg_latency_ms = start_event.elapsed_time(end_event) / test_iters
    
    print("\n" + "="*40)
    print(f"QAM Demapper 性能报告 (310B1)")
    print(f"输入规模: {TOTAL_ELEMENTS} 符号")
    print(f"输出规模: {output_npu.numel()} bits")
    print(f"平均单次耗时: {avg_latency_ms:.4f} ms")
    print(f"吞吐量: {(TOTAL_ELEMENTS / avg_latency_ms / 1000):.2f} MSymbols/s")
    print("="*40 + "\n")

    # 3. 结果基本检查
    res = output_npu.cpu().numpy()
    print(f"输出数据样例 (前12 bit): {res[:12]}")
    print(f"输出 Shape: {res.shape}")
    
    if res.shape[0] == TOTAL_ELEMENTS * BITS_PER_SYMBOL:
        print("✅ [Success] 输出维度校验通过！")
    else:
        print("❌ [Error] 输出维度不匹配！")

if __name__ == "__main__":
    test_qam_demod_performance()