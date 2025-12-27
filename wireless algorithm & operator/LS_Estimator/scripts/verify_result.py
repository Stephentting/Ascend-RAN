#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
# ===============================================================================
import sys
import numpy as np

# 容差设置
relative_tol = 1e-2   # 1%相对容差
absolute_tol = 1e-5   # 绝对容差  
error_tol = 2e-2      # 2%元素错误率容差

def real_to_complex_result(real_result):
    """将实数结果转换回复数"""
    M, two_N = real_result.shape
    N = two_N // 2
    real_part = real_result[:, 0:N]
    imag_part = real_result[:, N:2*N]
    return real_part + 1j * imag_part

def verify_result(output_file, golden_file):
    """验证算子输出结果"""
    print("="*60)
    print("OFDM信道估计结果验证")
    print("="*60)
    
    # 读取文件
    try:
        output = np.fromfile(output_file, dtype=np.float32)
        golden = np.fromfile(golden_file, dtype=np.float32)
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        return False
    
    # 验证尺寸
    expected_size = 1192 * 512
    if output.size != expected_size or golden.size != expected_size:
        print(f"❌ 数据大小错误:")
        print(f"   输出: {output.size}, 期望: {expected_size}")
        print(f"   参考: {golden.size}, 期望: {expected_size}")
        return False
    
    # Reshape数据
    output = output.reshape((1192, 512))
    golden = golden.reshape((1192, 512))
    print(f"✅ 数据读取成功: {output.shape}")

    
    
    # 转换回复数域分析
    output_complex = real_to_complex_result(output)
    golden_complex = real_to_complex_result(golden)
    
    # 计算误差统计
    complex_diff = output_complex - golden_complex
    abs_error = np.abs(complex_diff)
    
    mse = np.mean(abs_error ** 2)
    mae = np.mean(abs_error) 
    max_error = np.max(abs_error)
    
    # 相对误差
    signal_magnitude = np.mean(np.abs(golden_complex))
    relative_mae = mae / signal_magnitude if signal_magnitude > 0 else mae
    
    print(f"\n📊 复数域误差分析:")
    print(f"   MSE:         {mse:.2e}")
    print(f"   MAE:         {mae:.2e}")
    print(f"   最大误差:    {max_error:.2e}")
    print(f"   信号幅度:    {signal_magnitude:.2e}")
    print(f"   相对MAE:     {relative_mae:.3%}")
    
    # 实数域逐元素分析
    output_flat = output.reshape(-1)
    golden_flat = golden.reshape(-1)
    
    # 检查异常值
    if np.any(np.isnan(output_flat)) or np.any(np.isinf(output_flat)):
        print("❌ 输出包含NaN或Inf值!")
        return False
    
    # 逐元素比较
    close_mask = np.isclose(output_flat, golden_flat,
                           rtol=relative_tol, atol=absolute_tol,
                           equal_nan=True)
    
    different_count = np.sum(~close_mask)
    error_ratio = different_count / len(golden_flat)
    
    print(f"\n📊 实数域误差分析:")
    print(f"   不同元素数: {different_count:,}/{len(golden_flat):,}")
    print(f"   错误率:     {error_ratio:.3%}")
    print(f"   容差:       {error_tol:.1%}")
    
    # 显示前几个错误元素
    if different_count > 0:
        diff_indices = np.where(~close_mask)[0]
        print(f"\n🔍 前5个不同元素:")
        for i in range(min(5, len(diff_indices))):
            idx = diff_indices[i]
            batch = idx // 512
            elem = idx % 512
            part = "实部" if elem < 256 else "虚部"
            subcarrier = elem % 256
            
            expected = golden_flat[idx]
            actual = output_flat[idx]
            diff_val = abs(actual - expected)
            
            print(f"   [{batch:2d},{part},{subcarrier:3d}]: 期望={expected:8.5f}, 实际={actual:8.5f}, 差值={diff_val:.5f}")
    
    # 判断是否通过
    complex_pass = relative_mae <= 0.05      # 5%相对误差
    real_pass = error_ratio <= error_tol     # 错误率容差
    magnitude_pass = 0.001 <= signal_magnitude <= 100  # 合理信号范围
    
    overall_pass = complex_pass and real_pass and magnitude_pass
    
    print(f"\n📋 测试结果:")
    print(f"   复数域:     {'✅ 通过' if complex_pass else '❌ 失败'} (相对MAE ≤ 5%)")
    print(f"   实数域:     {'✅ 通过' if real_pass else '❌ 失败'} (错误率 ≤ {error_tol:.1%})")
    print(f"   信号幅度:   {'✅ 通过' if magnitude_pass else '❌ 失败'} (合理范围)")
    
    if overall_pass:
        print(f"\n🎉 总体结果: ✅ 测试通过!")
        print(f"   OFDM信道估计算子工作正常")
        print(f"   精度: {relative_mae:.3%} (相对误差)")
    else:
        print(f"\n❌ 总体结果: ❌ 测试失败!")
        if not complex_pass:
            print(f"   - 复数域误差过大: {relative_mae:.3%} > 5%")
        if not real_pass:
            print(f"   - 实数域错误率过高: {error_ratio:.3%} > {error_tol:.1%}")
        if not magnitude_pass:
            print(f"   - 信号幅度异常: {signal_magnitude:.2e}")
    
    return overall_pass

if __name__ == '__main__':
    try:
        if len(sys.argv) != 3:
            print("用法: python verify_result.py <输出文件> <参考文件>")
            sys.exit(1)
            
        success = verify_result(sys.argv[1], sys.argv[2])
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)