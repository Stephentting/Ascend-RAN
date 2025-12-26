"""
快速CPU测试 - simple_cpu_test.py
"""
import numpy as np
import time

def test_cpu_zf_32batch():
    """快速测试32 batch的CPU性能"""
    batch_size = 32
    num_subcarriers = 256
    
    # 生成数据
    h_real = np.random.randn(batch_size, num_subcarriers).astype(np.float32)
    h_imag = np.random.randn(batch_size, num_subcarriers).astype(np.float32)
    y_real = np.random.randn(batch_size, num_subcarriers).astype(np.float32)
    y_imag = np.random.randn(batch_size, num_subcarriers).astype(np.float32)
    
    # 预热
    for _ in range(50):
        H = h_real + 1j * h_imag
        y = y_real + 1j * y_imag
        x_hat = (np.conj(H) * y) / (np.abs(H)**2 + 1e-6)
    
    # 测试
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        
        H = h_real + 1j * h_imag
        y = y_real + 1j * y_imag
        x_hat = (np.conj(H) * y) / (np.abs(H)**2 + 1e-6)
        
        end = time.perf_counter()
        times.append((end - start) * 1e6)
    
    times = np.array(times)
    print(f"CPU性能测试 - Batch={batch_size}, Subcarriers={num_subcarriers}")
    print(f"  平均时间: {np.mean(times):.2f} us")
    print(f"  最小时间: {np.min(times):.2f} us")
    print(f"  最大时间: {np.max(times):.2f} us")
    print(f"  吞吐量: {(batch_size * num_subcarriers) / (np.mean(times) / 1e6) / 1e6:.2f} MSamples/s")

if __name__ == "__main__":
    test_cpu_zf_32batch()