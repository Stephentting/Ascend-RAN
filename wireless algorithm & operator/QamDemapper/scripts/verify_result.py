import numpy as np
import sys
import os

def verify_qam_hard_demapper_result(output_file, golden_file):
    """验证64-QAM硬解调结果 - Binary mapping (NO Gray code)"""
    
    num_symbols = 1192 * 220
    bits_per_symbol = 6
    total_bits = num_symbols * bits_per_symbol
    
    print("="*70)
    print(f"64-QAM硬解调结果验证 - Binary Mapping (NO Gray Code)")
    print(f"符号数={num_symbols}, 每符号比特数={bits_per_symbol}")
    print("="*70)
    
    # 检查文件存在
    if not os.path.exists(output_file):
        print(f"\n✗ 错误: 输出文件不存在: {output_file}")
        print(f"  请先运行: ./ascendc_kernels_bbit 或 ./run.sh")
        return 1
    
    if not os.path.exists(golden_file):
        print(f"\n✗ 错误: 参考文件不存在: {golden_file}")
        print(f"  请先运行: python scripts/gen_data.py")
        return 1
    
    # 读取数据
    output = np.fromfile(output_file, dtype=np.uint8)
    golden = np.fromfile(golden_file, dtype=np.uint8)
    
    print(f"\n文件路径:")
    print(f"  输出文件: {output_file}")
    print(f"  参考文件: {golden_file}")
    
    print(f"\n文件大小:")
    print(f"  输出: {len(output)} bytes")
    print(f"  参考: {len(golden)} bytes")
    print(f"  期望: {total_bits} bytes")
    
    # 检查长度
    if len(output) != total_bits:
        print(f"\n✗ 错误: 输出数据长度不匹配!")
        print(f"  期望: {total_bits}, 实际: {len(output)}")
        return 1
    
    if len(golden) != total_bits:
        print(f"\n✗ 错误: 参考数据长度不匹配!")
        print(f"  期望: {total_bits}, 实际: {len(golden)}")
        return 1
    
    # Reshape为符号格式 (num_symbols, 6)
    output_reshaped = output.reshape(num_symbols, bits_per_symbol)
    golden_reshaped = golden.reshape(num_symbols, bits_per_symbol)
    
    # ========== 逐比特分析 ==========
    bit_errors = (output_reshaped != golden_reshaped).astype(int)
    
    print(f"\n每个比特位置的错误统计:")
    print(f"{'比特位':<8} {'错误数':<10} {'错误率(%)':<12} {'状态':<6}")
    print("-"*40)
    
    bit_names = ['I_b2', 'I_b1', 'I_b0', 'Q_b2', 'Q_b1', 'Q_b0']
    for bit in range(bits_per_symbol):
        bit_error_count = np.sum(bit_errors[:, bit])
        bit_error_rate = (bit_error_count / num_symbols) * 100
        status = "✓" if bit_error_count == 0 else "✗"
        print(f"{bit_names[bit]:<8} {bit_error_count:<10} {bit_error_rate:<12.4f} {status}")
    
    # ========== 符号级分析 ==========
    symbol_errors = np.sum(bit_errors, axis=1)  # 每个符号的错误比特数
    symbols_with_errors = np.sum(symbol_errors > 0)
    symbol_error_rate = (symbols_with_errors / num_symbols) * 100
    
    print(f"\n符号级错误统计:")
    print(f"  总符号数: {num_symbols}")
    print(f"  错误符号数: {symbols_with_errors}")
    print(f"  符号错误率: {symbol_error_rate:.4f}%")
    
    # 错误比特数分布
    print(f"\n每符号错误比特数分布:")
    print(f"{'错误比特数':<12} {'符号数':<10} {'百分比(%)':<12}")
    print("-"*40)
    
    for err_bits in range(7):
        count = np.sum(symbol_errors == err_bits)
        percentage = (count / num_symbols) * 100
        if count > 0 or err_bits == 0:
            print(f"{err_bits:<12} {count:<10} {percentage:<12.4f}")
    
    # ========== 总体BER统计 ==========
    total_bit_errors = np.sum(bit_errors)
    ber = total_bit_errors / total_bits
    
    print(f"\n总体误码率(BER)统计:")
    print(f"  总比特数: {total_bits}")
    print(f"  错误比特数: {total_bit_errors}")
    print(f"  BER: {ber:.6f} ({ber*100:.4f}%)")
    
    # ========== 样本对比 ==========
    print(f"\n样本对比 (前5个符号):")
    print(f"{'符号':<6} {'位置':<8} {'输出':<6} {'参考':<6} {'误差':<6} {'状态':<6}")
    print("-"*45)
    
    for symbol in range(min(5, num_symbols)):
        for bit in range(bits_per_symbol):
            output_val = output_reshaped[symbol, bit]
            golden_val = golden_reshaped[symbol, bit]
            error = int(output_val != golden_val)
            status = "✓" if error == 0 else "✗"
            
            print(f"{symbol:<6} {bit_names[bit]:<8} {output_val:<6} {golden_val:<6} {error:<6} {status}")
        
        if symbol < 4:
            print("-"*45)
    
    # ========== I/Q分量对比 ==========
    print(f"\nI/Q分量解码对比 (前5个符号):")
    print(f"{'符号':<6} {'分量':<6} {'输出':<12} {'参考':<12} {'匹配':<6}")
    print("-"*50)
    
    for symbol in range(min(5, num_symbols)):
        # I分量 (bits 0,1,2)
        output_i_bits = output_reshaped[symbol, :3]
        golden_i_bits = golden_reshaped[symbol, :3]
        output_i_val = output_i_bits[0]*4 + output_i_bits[1]*2 + output_i_bits[2]
        golden_i_val = golden_i_bits[0]*4 + golden_i_bits[1]*2 + golden_i_bits[2]
        i_match = "✓" if output_i_val == golden_i_val else "✗"
        
        output_i_str = f"{output_i_bits}({output_i_val})"
        golden_i_str = f"{golden_i_bits}({golden_i_val})"
        print(f"{symbol:<6} {'I':<6} {output_i_str:<12} {golden_i_str:<12} {i_match}")
        
        # Q分量 (bits 3,4,5)
        output_q_bits = output_reshaped[symbol, 3:]
        golden_q_bits = golden_reshaped[symbol, 3:]
        output_q_val = output_q_bits[0]*4 + output_q_bits[1]*2 + output_q_bits[2]
        golden_q_val = golden_q_bits[0]*4 + golden_q_bits[1]*2 + golden_q_bits[2]
        q_match = "✓" if output_q_val == golden_q_val else "✗"
        
        output_q_str = f"{output_q_bits}({output_q_val})"
        golden_q_str = f"{golden_q_bits}({golden_q_val})"
        print(f"{symbol:<6} {'Q':<6} {output_q_str:<12} {golden_q_str:<12} {q_match}")
        
        if symbol < 4:
            print("-"*50)
    
    # ========== 错误符号详情 ==========
    if total_bit_errors > 0:
        error_symbols = np.where(symbol_errors > 0)[0]
        
        print(f"\n错误符号详细信息 (最多显示10个):")
        print(f"{'符号ID':<8} {'错误比特数':<12} {'输出':<20} {'参考':<20}")
        print("-"*70)
        
        for idx in error_symbols[:73]:
            err_count = symbol_errors[idx]
            output_str = f"{output_reshaped[idx]}"
            golden_str = f"{golden_reshaped[idx]}"
            print(f"{idx:<8} {err_count:<12} {output_str:<20} {golden_str:<20}")
        
        if len(error_symbols) > 10:
            print(f"... 还有 {len(error_symbols)-10} 个错误符号未显示")
    
    # ========== 判断通过标准 ==========
    print("\n" + "="*70)
    
    if total_bit_errors == 0:
        print("✓ 测试完美通过! 所有比特完全匹配")
        print(f"  BER = 0.000000 (0/{total_bits})")
        print(f"  符号错误率 = 0.00%")
        print("="*70)
        return 0
    elif ber < 0.01:
        print("✓ 测试通过 - BER在可接受范围内")
        print(f"  BER = {ber:.6f} < 0.01 (1%)")
        print(f"  错误比特数 = {total_bit_errors}/{total_bits}")
        print("="*70)
        return 0
    elif ber < 0.05:
        print("⚠ 测试通过但误码率偏高")
        print(f"  BER = {ber:.6f} < 0.05 (5%)")
        print(f"  错误比特数 = {total_bit_errors}/{total_bits}")
        print(f"  建议检查算法实现")
        print("="*70)
        return 0
    else:
        print("✗ 测试失败! 误码率过高")
        print(f"  BER = {ber:.6f} >= 0.05 (5%)")
        print(f"  错误比特数 = {total_bit_errors}/{total_bits}")
        print(f"  请检查算法实现")
        print("="*70)
        return 1


if __name__ == "__main__":
    # 支持从不同目录运行
    # 从 QamDemapperKernalInvoationNeo/ 运行: python scripts/verify_result.py
    # 从 scripts/ 运行: python verify_result.py
    
    # 检测当前目录
    if os.path.basename(os.getcwd()) == 'scripts':
        # 从scripts目录运行
        output_path = '../output/output.bin'
        golden_path = '../output/golden_output.bin'
    else:
        # 从项目根目录运行
        output_path = 'output/output.bin'
        golden_path = 'output/golden_output.bin'
    
    # 命令行参数覆盖默认路径
    if len(sys.argv) == 3:
        output_path = sys.argv[1]
        golden_path = sys.argv[2]
    elif len(sys.argv) != 1:
        print("用法:")
        print("  python scripts/verify_result.py")
        print("  或")
        print("  python scripts/verify_result.py output/output.bin output/golden_output.bin")
        sys.exit(1)
    
    sys.exit(verify_qam_hard_demapper_result(output_path, golden_path))
