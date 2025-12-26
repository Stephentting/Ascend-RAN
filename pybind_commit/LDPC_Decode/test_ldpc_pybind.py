import torch
import torch_npu
import numpy as np
import time
import os
import ldpc_custom

def load_bin_with_tiling(path, dtype, shape):
    """
    读取二进制文件，如果文件尺寸不足，则循环复制数据以匹配目标 shape
    """
    if not os.path.exists(path):
        return None
    
    # 1. 读入原始一维数据
    raw_data = np.fromfile(path, dtype=dtype)
    expected_size = np.prod(shape)
    actual_size = raw_data.size
    
    # 2. 检查并循环填充
    if actual_size < expected_size:
        # 计算需要复制的倍数
        repeats = int(np.ceil(expected_size / actual_size))
        # np.tile 会在内存中按顺序复制数据
        raw_data = np.tile(raw_data, repeats)[:expected_size]
        print(f"💡 [Info] 文件 {os.path.basename(path)} 大小不足，已通过循环填充扩展至 {shape} 规模")
    elif actual_size > expected_size:
        raw_data = raw_data[:expected_size]
        
    # 3. 变形并转为 Tensor
    return torch.from_numpy(raw_data.copy()).reshape(shape).to("npu:0")

def test_ldpc_logic():
    device = "npu:0"
    # 参数定义
    M, K, N = 256, 512, 256 
    num_chunks = 12
    total_rows = M * num_chunks  # 3072
    
    print(f"🔍 开始 LDPC 正确性与性能测试 (OrangePi AI Pro - 310B1)...")

    # --- 1. 导入数据 (应用循环填充逻辑) ---
    # 预期输入形状: (3072, 512)
    # 预期 H 矩阵形状: (512, 256)
    input_bits = load_bin_with_tiling("../input/x1_gm.bin", np.int8, (total_rows, K))
    h_matrix = load_bin_with_tiling("../input/x2_gm.bin", np.int8, (K, N))
    golden_bits = load_bin_with_tiling("../output/golden.bin", np.int8, (M, K)) # Golden 通常只有一组

    if input_bits is None or h_matrix is None:
        print("❌ 错误: 关键输入文件缺失，请检查路径！")
        return

    # --- 2. 正确性校验 (验证第一组数据) ---
    print("🧪 正在执行正确性校验...")
    # 克隆一份用于验证，避免原位修改导致性能测试数据改变
    verify_bits = input_bits.clone()
    
    # 启动 NPU 算子，执行 20 次迭代
    ldpc_custom.run_ldpc_decode(verify_bits, h_matrix)
    
    if golden_bits is not None:
        # 提取第一组 256 行结果进行比对
        npu_res_first = verify_bits[:256, :].cpu().numpy()
        golden_np = golden_bits.cpu().numpy()
        
        error_count = np.sum(npu_res_first != golden_np)
        if error_count == 0:
            print("✅ [Success] 数据正确性比对通过！")
        else:
            print(f"❌ [Fail] 校验未通过！错误点数: {error_count}/{256*512}")
    
    # --- 3. 性能压测 (12 组大帧聚合) ---
    print(f"🔥 正在进行吞吐量测试 (30 次迭代)...")
    torch.npu.synchronize()
    start_time = time.perf_counter()

    for _ in range(30):
        # 原位修改模式，模拟真实通信流
        _ = ldpc_custom.run_ldpc_decode(input_bits, h_matrix)
        
        # 模拟视频数据提取逻辑
        video_data = input_bits.view(12, 256, 512)[:, :, :252]

    torch.npu.synchronize()
    end_time = time.perf_counter()

    # --- 统计结果 ---
    avg_ms = ((end_time - start_time) * 1000) / 30
    total_payload_bits = total_rows * K
    mbps = (total_payload_bits / (avg_ms / 1000.0)) / 1e6

    print("-" * 50)
    print(f"📊 性能结果 (大帧聚合模式):")
    print(f"平均单帧耗时: {avg_ms:.2f} ms")
    print(f"有效吞吐量:   {mbps:.2f} Mbps")
    print("-" * 50)

if __name__ == "__main__":
    test_ldpc_logic()