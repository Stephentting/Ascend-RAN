import numpy as np
import sys
import os

def verify_qam64_result_batch(output_real_file, output_imag_file, 
                             golden_real_file, golden_imag_file):
    """验证QAM64调制结果 - Batch版本"""
    
    batch_size = 1192
    num_symbols_per_batch = 220
    total_symbols = batch_size * num_symbols_per_batch
    
    print("="*70)
    print(f"QAM64调制结果验证 - Batch={batch_size}")
    print("="*70)
    
    # 检查文件是否存在
    for file in [output_real_file, output_imag_file, golden_real_file, golden_imag_file]:
        if not os.path.exists(file):
            print(f"✗ 错误: 文件不存在: {file}")
            return 1
    
    # 读取数据
    try:
        output_real = np.fromfile(output_real_file, dtype=np.float16)
        output_imag = np.fromfile(output_imag_file, dtype=np.float16)
        golden_real = np.fromfile(golden_real_file, dtype=np.float16)
        golden_imag = np.fromfile(golden_imag_file, dtype=np.float16)
    except Exception as e:
        print(f"✗ 错误: 读取文件失败: {e}")
        return 1
    
    if len(output_real) != total_symbols:
        print(f"✗ 错误: 数据长度不匹配!")
        print(f"  期望: {total_symbols}, 实际: {len(output_real)}")
        return 1
    
    # Reshape为batch格式
    output_real = output_real.reshape(batch_size, num_symbols_per_batch)
    output_imag = output_imag.reshape(batch_size, num_symbols_per_batch)
    golden_real = golden_real.reshape(batch_size, num_symbols_per_batch)
    golden_imag = golden_imag.reshape(batch_size, num_symbols_per_batch)
    
    # 组合成复数
    output_complex = output_real + 1j * output_imag
    golden_complex = golden_real + 1j * golden_imag
    
    # 误差分析
    abs_error = np.abs(output_complex - golden_complex)
    mean_error = np.mean(abs_error)
    max_error = np.max(abs_error)
    rms_error = np.sqrt(np.mean(abs_error**2))
    
    # 每个batch的统计
    print(f"\n每个Batch的误差统计:")
    print(f"{'Batch':<6} {'平均误差':<12} {'最大误差':<12} {'状态':<10}")
    print("-"*50)
    
    batch_pass_count = 0
    for b in range(batch_size):
        batch_mean = np.mean(abs_error[b])
        batch_max = np.max(abs_error[b])
        status = "✓" if batch_mean < 0.01 and batch_max < 0.1 else "✗"
        if status == "✓":
            batch_pass_count += 1
        print(f"{b:<6} {batch_mean:<12.6f} {batch_max:<12.6f} {status}")
    
    # 总体统计
    print(f"\n总体统计:")
    print(f"  总符号数: {total_symbols}")
    print(f"  平均误差: {mean_error:.6f}")
    print(f"  最大误差: {max_error:.6f}")
    print(f"  RMS误差:  {rms_error:.6f}")
    print(f"  通过batch数: {batch_pass_count}/{batch_size}")
    
    # 显示几个样本对比
    print(f"\n样本对比 (Batch 0, 前5个符号):")
    print(f"{'Symbol':<6} {'Output':>22} {'Golden':>22} {'Error':>10}")
    print("-"*70)
    for i in range(5):
        out_str = f"{output_real[0,i]:>7.4f}{output_imag[0,i]:+7.4f}j"
        gold_str = f"{golden_real[0,i]:>7.4f}{golden_imag[0,i]:+7.4f}j"
        err = abs_error[0,i]
        print(f"{i:<6} {out_str:>22} {gold_str:>22} {err:>10.6f}")
    
    # 星座点验证
    print(f"\n星座点验证:")
    unique_output = len(np.unique(output_complex, axis=None))
    unique_golden = len(np.unique(golden_complex, axis=None))
    print(f"  输出唯一星座点数: {unique_output}")
    print(f"  期望唯一星座点数: {unique_golden}")
    
    # 功率验证
    output_power = np.mean(np.abs(output_complex)**2)
    golden_power = np.mean(np.abs(golden_complex)**2)
    power_error = abs(output_power - golden_power)
    print(f"  输出平均功率: {output_power:.6f}")
    print(f"  期望平均功率: {golden_power:.6f}")
    print(f"  功率误差: {power_error:.6f}")
    
    # 判断通过
    print("\n" + "="*70)
    if mean_error < 0.01 and max_error < 0.1 and power_error < 0.01:
        print("✅ 测试通过! QAM64调制算子工作正常")
        print(f"  ✓ 平均误差 {mean_error:.6f} < 0.01")
        print(f"  ✓ 最大误差 {max_error:.6f} < 0.1") 
        print(f"  ✓ 功率误差 {power_error:.6f} < 0.01")
        print(f"  ✓ {batch_pass_count}/{batch_size} 个batch通过验证")
        print("="*70)
        return 0
    else:
        print("❌ 测试失败!")
        if mean_error >= 0.01:
            print(f"  ✗ 平均误差 {mean_error:.6f} >= 0.01")
        if max_error >= 0.1:
            print(f"  ✗ 最大误差 {max_error:.6f} >= 0.1")
        if power_error >= 0.01:
            print(f"  ✗ 功率误差 {power_error:.6f} >= 0.01")
        print("="*70)
        return 1

def main():
    if len(sys.argv) != 5:
        print("用法: python3 verify_qam64.py <输出实部> <输出虚部> <期望实部> <期望虚部>")
        print("示例: python3 verify_qam64.py output/actual_real.bin output/actual_imag.bin output/golden_real.bin output/golden_imag.bin")
        sys.exit(1)
    
    output_real_file, output_imag_file, golden_real_file, golden_imag_file = sys.argv[1:5]
    
    result = verify_qam64_result_batch(output_real_file, output_imag_file, 
                                     golden_real_file, golden_imag_file)
    sys.exit(result)

if __name__ == "__main__":
    main()