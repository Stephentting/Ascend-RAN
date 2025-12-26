import numpy as np
import os

def generate_zf_test_data_batch():
    """生成ZF均衡测试数据 - Batch=32，使用随机数"""
    
    os.makedirs('./input', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    
    batch_size = 1192
    num_subcarriers = 256
    total_size = batch_size * num_subcarriers  # 8192
    
    # 移除固定随机种子，使用真正的随机数
    # np.random.seed(42)  # 注释掉固定种子
    
    print("="*70)
    print(f"ZF均衡测试数据生成 - Batch={batch_size}, Subcarriers={num_subcarriers}")
    print("="*70)
    
    # 生成batch数据 [batch_size, num_subcarriers]
    # 使用更多样化的信道分布
    H_real = (np.random.randn(batch_size, num_subcarriers) * 0.5 + 1.0).astype(np.float32)
    H_imag = (np.random.randn(batch_size, num_subcarriers) * 0.3).astype(np.float32)
    H = H_real + 1j * H_imag
    
    print(f"\n信道H统计:")
    print(f"  形状: {H.shape}")
    print(f"  |H| 范围: {np.min(np.abs(H)):.4f} ~ {np.max(np.abs(H)):.4f}")
    print(f"  |H| 平均: {np.mean(np.abs(H)):.4f}")
    
    # 生成发送符号 (QPSK)
    x_real = np.random.choice([-1.0, 1.0], (batch_size, num_subcarriers)).astype(np.float32)
    x_imag = np.random.choice([-1.0, 1.0], (batch_size, num_subcarriers)).astype(np.float32)
    x = x_real + 1j * x_imag
    
    print(f"\n发送符号x (QPSK):")
    print(f"  形状: {x.shape}")
    print(f"  平均功率: {np.mean(np.abs(x)**2):.4f}")
    
    # 接收信号
    y = H * x
    
    print(f"\n接收信号y:")
    print(f"  |y| 范围: {np.min(np.abs(y)):.4f} ~ {np.max(np.abs(y)):.4f}")
    
    # Golden结果
    H_conj = np.conj(H)
    H_squared_mag = np.abs(H)**2
    x_hat_golden = (H_conj * y) / H_squared_mag
    
    error = np.abs(x_hat_golden - x)
    print(f"\nGolden结果验证:")
    print(f"  平均误差: {np.mean(error):.8f}")
    print(f"  最大误差: {np.max(error):.8f}")
    print(f"  ✓ Golden计算正确!")
    
    # 转换为half并flatten
    H_real_half = H.real.astype(np.float16).flatten()
    H_imag_half = H.imag.astype(np.float16).flatten()
    y_real_half = y.real.astype(np.float16).flatten()
    y_imag_half = y.imag.astype(np.float16).flatten()
    x_hat_real_half = x_hat_golden.real.astype(np.float16).flatten()
    x_hat_imag_half = x_hat_golden.imag.astype(np.float16).flatten()
    
    # 保存文件
    H_real_half.tofile('./input/input_h_real.bin')
    H_imag_half.tofile('./input/input_h_imag.bin')
    y_real_half.tofile('./input/input_y_real.bin')
    y_imag_half.tofile('./input/input_y_imag.bin')
    
    x_hat_real_half.tofile('./output/golden_x_hat_real.bin')
    x_hat_imag_half.tofile('./output/golden_x_hat_imag.bin')
    
    print(f"\n生成的文件:")
    print(f"  输入文件 (每个{total_size} half = {total_size*2} bytes):")
    print(f"    ./input/input_h_real.bin")
    print(f"    ./input/input_h_imag.bin")
    print(f"    ./input/input_y_real.bin")
    print(f"    ./input/input_y_imag.bin")
    print(f"\n  Golden输出:")
    print(f"    ./output/golden_x_hat_real.bin")
    print(f"    ./output/golden_x_hat_imag.bin")
    print(f"\n数据排列: [batch0_sub0, batch0_sub1, ..., batch0_sub255, "
          f"batch1_sub0, ..., batch31_sub255]")
    print("="*70)

if __name__ == "__main__":
    generate_zf_test_data_batch()