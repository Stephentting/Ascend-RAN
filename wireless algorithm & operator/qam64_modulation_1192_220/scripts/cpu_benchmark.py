"""
快速CPU测试 - simple_cpu_test_qam64.py
QAM64调制CPU性能测试
"""
import numpy as np
import time

def test_cpu_qam64_32batch():
    """快速测试32 batch的QAM64调制CPU性能"""
    batch_size = 1192
    num_symbols_per_batch = 220
    bits_per_symbol = 6
    total_symbols = batch_size * num_symbols_per_batch
    total_bits = total_symbols * bits_per_symbol
    
    # QAM64格雷编码映射
    gray_map = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float32)
    normalization_factor = 1.0 / np.sqrt(42.0)  # 1/√42
    
    # 生成随机比特流 [batch_size, num_symbols_per_batch, 6]
    input_bits = np.random.randint(0, 2, (batch_size, num_symbols_per_batch, bits_per_symbol), dtype=np.uint8)
    
    # 预热
    for _ in range(50):
        output_real = np.zeros((batch_size, num_symbols_per_batch), dtype=np.float32)
        output_imag = np.zeros((batch_size, num_symbols_per_batch), dtype=np.float32)
        
        for b in range(batch_size):
            for s in range(num_symbols_per_batch):
                b5, b4, b3, b2, b1, b0 = input_bits[b, s]
                i_index = (b5 << 2) | (b4 << 1) | b3
                q_index = (b2 << 2) | (b1 << 1) | b0
                output_real[b, s] = normalization_factor * gray_map[i_index]
                output_imag[b, s] = normalization_factor * gray_map[q_index]
    
    # 测试性能
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        
        output_real = np.zeros((batch_size, num_symbols_per_batch), dtype=np.float32)
        output_imag = np.zeros((batch_size, num_symbols_per_batch), dtype=np.float32)
        
        # QAM64调制核心算法
        for b in range(batch_size):
            for s in range(num_symbols_per_batch):
                b5, b4, b3, b2, b1, b0 = input_bits[b, s]
                i_index = (b5 << 2) | (b4 << 1) | b3
                q_index = (b2 << 2) | (b1 << 1) | b0
                output_real[b, s] = normalization_factor * gray_map[i_index]
                output_imag[b, s] = normalization_factor * gray_map[q_index]
        
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # 转换为微秒
    
    times = np.array(times)
    
    print("QAM64调制CPU性能测试")
    print("=" * 50)
    print(f"配置: {batch_size}批次 × {num_symbols_per_batch}符号 = {total_symbols}符号")
    print(f"数据量: {total_bits}比特 -> {total_symbols}个QAM64符号")
    print()
    print("性能统计:")
    print(f"  平均时间: {np.mean(times):.2f} us")
    print(f"  最小时间: {np.min(times):.2f} us") 
    print(f"  最大时间: {np.max(times):.2f} us")
    print(f"  标准差: {np.std(times):.2f} us")
    print()
    
    # 计算吞吐量
    avg_time_seconds = np.mean(times) / 1e6
    symbols_per_second = total_symbols / avg_time_seconds
    bits_per_second = total_bits / avg_time_seconds
    
    print("吞吐量:")
    print(f"  {symbols_per_second/1e6:.2f} MSymbols/s (符号/秒)")
    print(f"  {bits_per_second/1e6:.2f} Mbps (比特/秒)")
    print(f"  {symbols_per_second/avg_time_seconds/1e6:.2f} MSymbols/s/us (每微秒符号数)")

def test_optimized_qam64():
    """优化版本的QAM64调制测试（使用向量化操作）"""
    batch_size = 1024
    num_symbols = 240
    bits_per_symbol = 6
    
    # 生成测试数据
    input_bits = np.random.randint(0, 2, (batch_size, num_symbols, bits_per_symbol), dtype=np.uint8)
    
    # 优化版本：使用向量化操作
    def qam64_modulate_vectorized(bits):
        """向量化QAM64调制"""
        gray_map = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float32)
        norm = 1.0 / np.sqrt(42.0)
        
        # 提取I、Q路索引
        i_indices = (bits[:,:,0] << 2) | (bits[:,:,1] << 1) | bits[:,:,2]
        q_indices = (bits[:,:,3] << 2) | (bits[:,:,4] << 1) | bits[:,:,5]
        
        # 映射到星座点
        real = norm * gray_map[i_indices]
        imag = norm * gray_map[q_indices]
        
        return real, imag
    
    # 预热
    for _ in range(50):
        real, imag = qam64_modulate_vectorized(input_bits)
    
    # 性能测试
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        real, imag = qam64_modulate_vectorized(input_bits)
        end = time.perf_counter()
        times.append((end - start) * 1e6)
    
    times = np.array(times)
    
    print("\n优化版本QAM64调制（向量化）")
    print("=" * 50)
    print(f"平均时间: {np.mean(times):.2f} us")
    print(f"吞吐量: {(batch_size * num_symbols) / (np.mean(times) / 1e6) / 1e6:.2f} MSymbols/s")

if __name__ == "__main__":
    print("QAM64调制CPU性能测试工具")
    print("=" * 60)
    
    # 测试基础版本
    test_cpu_qam64_32batch()
    
    # 测试优化版本
    test_optimized_qam64()
    
    print("\n" + "=" * 60)
    print("测试完成!")