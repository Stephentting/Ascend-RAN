#!/usr/bin/python3
# coding=utf-8
"""
C++调用模式的ZF均衡结果验证
输出格式与pybind测试保持一致
"""

import numpy as np
import sys

def verify_zf_result_batch(output_real_file, output_imag_file,
                          golden_real_file, golden_imag_file):
    """验证ZF均衡结果 - Batch版本"""
    
    batch_size = 1192  # 修改为32以匹配pybind测试
    num_subcarriers = 256
    total_size = batch_size * num_subcarriers
    
    print("\n" + "="*70)
    print(f"[C++ Mode] ZF Equalization Result Verification")
    print(f"Batch={batch_size}, Subcarriers={num_subcarriers}")
    print("="*70)
    
    # 读取数据
    try:
        output_real = np.fromfile(output_real_file, dtype=np.float16)
        output_imag = np.fromfile(output_imag_file, dtype=np.float16)
        golden_real = np.fromfile(golden_real_file, dtype=np.float16)
        golden_imag = np.fromfile(golden_imag_file, dtype=np.float16)
    except Exception as e:
        print(f"✗ Error reading files: {e}")
        return 1
    
    if len(output_real) != total_size:
        print(f"✗ 错误: 数据长度不匹配!")
        print(f"  期望: {total_size}, 实际: {len(output_real)}")
        return 1
    
    # Reshape为batch格式
    output_real = output_real.reshape(batch_size, num_subcarriers)
    output_imag = output_imag.reshape(batch_size, num_subcarriers)
    golden_real = golden_real.reshape(batch_size, num_subcarriers)
    golden_imag = golden_imag.reshape(batch_size, num_subcarriers)
    
    # 计算误差 (与pybind测试一致)
    diff_real = np.abs(output_real - golden_real)
    diff_imag = np.abs(output_imag - golden_imag)
    
    print(f"\n[Debug] Checking results...")
    print(f"  output_real shape: {output_real.shape}, dtype: {output_real.dtype}")
    print(f"  golden_real shape: {golden_real.shape}, dtype: {golden_real.dtype}")
    
    print(f"\n[Debug] Error Statistics (Real part):")
    print(f"  Mean error: {diff_real.mean():.6f}")
    print(f"  Max error: {diff_real.max():.6f}")
    print(f"  Min error: {diff_real.min():.6f}")
    
    print(f"\n[Debug] Error Statistics (Imag part):")
    print(f"  Mean error: {diff_imag.mean():.6f}")
    print(f"  Max error: {diff_imag.max():.6f}")
    print(f"  Min error: {diff_imag.min():.6f}")
    
    print(f"\n[Debug] Sample comparison (first 5 elements):")
    print(f"  NPU Real: {output_real.flatten()[:5]}")
    print(f"  CPU Real: {golden_real.flatten()[:5]}")
    print(f"  NPU Imag: {output_imag.flatten()[:5]}")
    print(f"  CPU Imag: {golden_imag.flatten()[:5]}")
    
    # 组合成复数计算总体误差
    output_complex = output_real + 1j * output_imag
    golden_complex = golden_real + 1j * golden_imag
    abs_error = np.abs(output_complex - golden_complex)
    mean_error = np.mean(abs_error)
    max_error = np.max(abs_error)
    rms_error = np.sqrt(np.mean(abs_error**2))
    
    # 每个batch的统计 (显示前10个)
    print(f"\n每个Batch的误差统计 (前10个):")
    print(f"{'Batch':<6} {'平均误差':<12} {'最大误差':<12} {'状态':<10}")
    print("-"*50)
    
    for b in range(min(10, batch_size)):
        batch_mean = np.mean(abs_error[b])
        batch_max = np.max(abs_error[b])
        status = "✓" if batch_mean < 0.01 else "✗"
        print(f"{b:<6} {batch_mean:<12.6f} {batch_max:<12.6f} {status}")
    
    # 总体统计
    print(f"\n总体统计:")
    print(f"  总样本数: {total_size}")
    print(f"  平均误差: {mean_error:.6f}")
    print(f"  最大误差: {max_error:.6f}")
    print(f"  RMS误差:  {rms_error:.6f}")
    
    # 显示样本对比 (Batch 0, 前5个)
    print(f"\n样本对比 (Batch 0, 前5个子载波):")
    print(f"{'Sub':<4} {'Output':>22} {'Golden':>22} {'Error':>10}")
    print("-"*70)
    for i in range(5):
        out_str = f"{output_real[0,i]:>7.4f}{output_imag[0,i]:+7.4f}j"
        gold_str = f"{golden_real[0,i]:>7.4f}{golden_imag[0,i]:+7.4f}j"
        err = abs_error[0,i]
        print(f"{i:<4} {out_str:>22} {gold_str:>22} {err:>10.6f}")
    
    # 判断通过 (使用与pybind一致的阈值)
    print("\n" + "="*70)
    print("验证结果:")
    print("="*70)
    
    # 分别判断实部和虚部
    real_pass = diff_real.mean() < 0.01 and diff_real.max() < 0.1
    imag_pass = diff_imag.mean() < 0.01 and diff_imag.max() < 0.1
    
    if real_pass and imag_pass:
        print("✓ 测试通过! ZF均衡算子工作正常")
        print(f"  实部平均误差 {diff_real.mean():.6f} < 0.01")
        print(f"  虚部平均误差 {diff_imag.mean():.6f} < 0.01")
    else:
        print("✗ 测试失败!")
        if not real_pass:
            print(f"  实部平均误差 = {diff_real.mean():.6f} (期望 < 0.01)")
            print(f"  实部最大误差 = {diff_real.max():.6f} (期望 < 0.1)")
        if not imag_pass:
            print(f"  虚部平均误差 = {diff_imag.mean():.6f} (期望 < 0.01)")
            print(f"  虚部最大误差 = {diff_imag.max():.6f} (期望 < 0.1)")
    
    print("="*70 + "\n")
    
    return 0 if (real_pass and imag_pass) else 1

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("用法: python3 verify_zf_result_cpp.py <output_real> <output_imag> <golden_real> <golden_imag>")
        print("示例: python3 verify_zf_result_cpp.py output/output_x_hat_real.bin output/output_x_hat_imag.bin output/golden_x_hat_real.bin output/golden_x_hat_imag.bin")
        sys.exit(1)
    
    sys.exit(verify_zf_result_batch(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))