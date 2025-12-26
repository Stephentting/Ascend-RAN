import numpy as np
import os

def generate_qam64_test_data_gray():
    """生成标准 Gray 码映射的 QAM64 调制测试数据"""
    
    os.makedirs('./input', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    
    # 配置参数
    batch_size = 1192
    symbols_per_batch = 220
    bits_per_symbol = 6
    total_symbols = batch_size * symbols_per_batch
    total_bits = total_symbols * bits_per_symbol
    
    np.random.seed(42)
    
    print("生成标准 Gray 码 QAM64 调制测试数据...")
    
    # 生成随机比特流
    input_bits = np.random.randint(0, 2, (batch_size, symbols_per_batch, bits_per_symbol), dtype=np.uint8)
    
    # --- 核心修改点：重新排列映射表以匹配标准 Gray 码 ---
    # 索引 (二进制值) -> 电平值的映射关系：
    # 0(000):-7, 1(001):-5, 2(010):-1, 3(011):-3, 4(100):7, 5(101):5, 6(110):1, 7(111):3
    gray_lut = np.array([-7, -5, -1, -3, 7, 5, 1, 3], dtype=np.float32)
    norm_factor = 1.0 / np.sqrt(42.0)
    
    # 生成期望输出
    output_real = np.zeros((batch_size, symbols_per_batch), dtype=np.float32)
    output_imag = np.zeros((batch_size, symbols_per_batch), dtype=np.float32)
    
    for b in range(batch_size):
        for s in range(symbols_per_batch):
            # 提取 6 个比特
            b5, b4, b3, b2, b1, b0 = input_bits[b, s]
            
            # 计算索引（保持与算子一致的位权逻辑）
            i_index = (b5 << 2) | (b4 << 1) | b3
            q_index = (b2 << 2) | (b1 << 1) | b0
            
            # 使用修正后的 Gray LUT 映射
            output_real[b, s] = norm_factor * gray_lut[i_index]
            output_imag[b, s] = norm_factor * gray_lut[q_index]
    
    # 保存文件
    input_bits.flatten().astype(np.uint8).tofile('./input/input_bits.bin')
    # NPU 算子通常输出 half (float16)
    output_real.flatten().astype(np.float16).tofile('./output/golden_symbols_real.bin')
    output_imag.flatten().astype(np.float16).tofile('./output/golden_symbols_imag.bin')
    
    print("✓ Gray 码数据生成完成")
    print(f"验证点：比特 [0,1,0] 现在映射到电平 {gray_lut[2]*norm_factor:.4f} (预期 -1/sqrt(42))")

if __name__ == "__main__":
    generate_qam64_test_data_gray()