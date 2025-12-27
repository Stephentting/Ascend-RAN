import numpy as np
import os

def generate_qam_hard_demapper_test_data():
    """Generate 64-QAM hard demodulation test data - Binary mapping (NO Gray code)"""
    
    # 检测当前目录，自动适配路径
    if os.path.basename(os.getcwd()) == 'scripts':
        # 从scripts目录运行
        input_dir = '../input'
        output_dir = '../output'
    else:
        # 从项目根目录运行
        input_dir = 'input'
        output_dir = 'output'
    
    # 确保目录存在
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    num_symbols = 1192 * 220
    noise_var = 1.0
    bits_per_symbol = 6
    
    np.random.seed(42)
    
    print("="*70)
    print(f"64-QAM Hard Demodulation - Binary Mapping (NO Gray Code)")
    print(f"Symbols={num_symbols}")
    print("="*70)
    
    # 8个电平值（归一化）
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float32)
    norm_factor = np.sqrt(42.0)
    levels = levels / norm_factor
    
    print(f"\nNormalized levels: {levels}")
    print(f"Normalization: sqrt(42) = {norm_factor:.6f}")
    print(f"\n⚠️  Using BINARY mapping (index = binary code)")
    print(f"   Index 0->000, 1->001, 2->010, ..., 7->111")
    
    # 生成64QAM星座图
    constellation = []
    for i in levels:
        for q in levels:
            constellation.append(complex(i, q))
    constellation = np.array(constellation)
    
    # 生成随机符号
    symbol_indices = np.random.randint(0, 64, num_symbols)
    tx_symbols = constellation[symbol_indices]
    
    # 添加噪声
    noise = (np.random.normal(0, np.sqrt(noise_var/2), num_symbols) + 
             1j * np.random.normal(0, np.sqrt(noise_var/2), num_symbols)).astype(np.complex64)
    rx_symbols = tx_symbols + noise
    
    snr_db = 10 * np.log10(np.mean(np.abs(tx_symbols)**2) / noise_var)
    print(f"\nSNR: {snr_db:.2f} dB")
    
    # ========== 硬解调（普通二进制） ==========
    def hard_demod_binary(symbols, levels):
        num_symbols = len(symbols)
        i_comp = symbols.real
        q_comp = symbols.imag
        
        # 找最近电平的索引（索引就是二进制码）
        i_dist = np.abs(levels[:, np.newaxis] - i_comp)
        q_dist = np.abs(levels[:, np.newaxis] - q_comp)
        i_idx = np.argmin(i_dist, axis=0)  # 0-7
        q_idx = np.argmin(q_dist, axis=0)  # 0-7
        
        # 直接从索引提取比特
        bits = np.zeros(num_symbols * 6, dtype=np.uint8)
        bits[0::6] = (i_idx >> 2) & 1
        bits[1::6] = (i_idx >> 1) & 1
        bits[2::6] = i_idx & 1
        bits[3::6] = (q_idx >> 2) & 1
        bits[4::6] = (q_idx >> 1) & 1
        bits[5::6] = q_idx & 1
        
        return bits
    
    golden_output = hard_demod_binary(rx_symbols, levels)
    
    print(f"\nGolden output: {len(golden_output)} bytes")
    print(f"First 4 symbols:")
    for i in range(4):
        b = golden_output[i*6:(i+1)*6]
        i_val = b[0]*4 + b[1]*2 + b[2]
        q_val = b[3]*4 + b[4]*2 + b[5]
        print(f"  Sym{i}: I={b[:3]}({i_val}) Q={b[3:]}({q_val})")
    
    # 保存文件
    input_i_file = os.path.join(input_dir, 'input_input_I.bin')
    input_q_file = os.path.join(input_dir, 'input_input_Q.bin')
    golden_file = os.path.join(output_dir, 'golden_output.bin')
    
    rx_symbols.real.astype(np.float32).tofile(input_i_file)
    rx_symbols.imag.astype(np.float32).tofile(input_q_file)
    golden_output.tofile(golden_file)
    
    print(f"\n✅ Files saved:")
    print(f"   {input_i_file} - {num_symbols*4} bytes")
    print(f"   {input_q_file} - {num_symbols*4} bytes")
    print(f"   {golden_file} - {len(golden_output)} bytes")
    print("="*70)


if __name__ == "__main__":
    generate_qam_hard_demapper_test_data()
