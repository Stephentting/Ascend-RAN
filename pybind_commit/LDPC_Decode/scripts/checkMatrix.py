#!/usr/bin/python3
# coding=utf-8
# ===============================================================================
# LDPC Matrix Verification and Decoding Simulation Script
# ===============================================================================

import numpy as np

def verify_ldpc_system():
    # 设定维度参数
    M = 256  # 帧数
    N = 576  # 码长
    K = 288  # 信息位长
    MAX_ITER = 50  # 最大迭代次数 (与 C++ 保持一致)

    print(f"Checking LDPC System with M={M}, N={N}, K={K}...")

    # =========================================================================
    # 1. 读取矩阵
    # =========================================================================
    
    # x2_gm: 译码校验矩阵 H^T (576 x 288)
    try:
        x2_gm = np.fromfile("matrix_A_576_288_int8.bin", dtype=np.uint8).reshape([N, K])
        print("Loaded Decoder Matrix (H^T).")
    except FileNotFoundError:
        print("Error: matrix_A_576_288_int8.bin not found.")
        print("Using random matrix (WILL FAIL VALIDATION)...")
        x2_gm = np.random.randint(0, 2, [N, K]).astype(np.uint8)

    # G_matrix: 编码矩阵 G (288 x 576)
    try:
        G_matrix = np.fromfile("WIMAX_288_576_G_RAW.bin", dtype=np.uint8).reshape([K, N])
        print("Loaded Encoder Matrix (G).")
    except FileNotFoundError:
        print("Error: WIMAX_288_576_G_RAW.bin not found.")
        print("Using random matrix (WILL FAIL VALIDATION)...")
        G_matrix = np.random.randint(0, 2, [K, N]).astype(np.uint8)

    # =========================================================================
    # 2. 验证矩阵正交性 (G * H^T == 0 mod 2)
    # =========================================================================
    print("-" * 60)
    print("Step 1: Verifying Matrix Orthogonality (G * H^T = 0)")
    
    # 矩阵乘法: (288, 576) x (576, 288) -> (288, 288)
    check_matrix = np.matmul(G_matrix, x2_gm) % 2
    non_zero_count = np.count_nonzero(check_matrix)
    
    if non_zero_count == 0:
        print("[PASS] Matrices are compatible. G * H^T is all zeros.")
    else:
        print(f"[FAIL] Matrices are incompatible! Found {non_zero_count} non-zero elements.")
        print("       Encoding data will NOT satisfy Parity Check constraints.")
        # 如果矩阵不对，后面的测试意义不大，但为了调试流程继续执行
    
    print("-" * 60)

    # =========================================================================
    # 3. 编码 (Encoding) & 注入错误 (Error Injection)
    # =========================================================================
    
    # 生成随机信息比特
    msgs = np.random.randint(0, 2, [M, K]).astype(np.uint8)
    
    # 编码得到“完美”码字 (Golden)
    codewords = np.matmul(msgs, G_matrix) % 2
    
    # 注入错误
    error_rate = 0.005 # 2% 误码率
    error_mask = np.random.choice(
        [0, 1], 
        size=codewords.shape, 
        p=[1 - error_rate, error_rate]
    ).astype(np.uint8)
    
    # 带噪数据 (作为译码器输入)
    x1_gm = np.bitwise_xor(codewords.astype(np.uint8), error_mask)
    initial_errors = np.sum(error_mask)
    
    print(f"Step 2: Data Generated.")
    print(f"        Total initial bit errors injected: {initial_errors}")
    print("-" * 60)

    # =========================================================================
    # 4. Python 模拟 C++ 算子译码 (Bit Flipping)
    # =========================================================================
    print("Step 3: Running Python Bit-Flipping Decoding...")
    
    # 复制一份输入数据用于处理
    current_data = x1_gm.copy()
    success = False
    
    for i in range(MAX_ITER):
        # A. 计算伴随式 (Syndrome) [M, K]
        #    S = x * H^T
        syndrome = np.matmul(current_data, x2_gm) % 2
        
        # B. 检查校验和
        #    如果某行的 Syndrome 全为 0，说明该帧校验通过
        row_syndrome_sum = np.sum(syndrome, axis=1) # [M]
        total_syndrome_sum = np.sum(row_syndrome_sum)
        
        print(f"  Iter {i:02d}: Total Syndrome Sum = {total_syndrome_sum}")
        
        if total_syndrome_sum == 0:
            print(f"  >>> Decoding Converged (Success) at iteration {i}!")
            success = True
            break
            
        # C. 生成行掩码 (Valid Row Mask)
        #    只有校验失败 (Sum > 0) 的行才允许翻转
        #    维度扩展为 [M, 1] 以便广播
        valid_row_mask = (row_syndrome_sum > 0).astype(np.uint8).reshape(-1, 1)
        
        # D. 计算投票 (Votes) [M, N]
        #    Votes = Syndrome * H (即 x2_gm.T)
        #    注意：这里是整数矩阵乘法，不取模
        votes = np.matmul(syndrome, x2_gm.T)
        
        # E. 翻转决策
        #    1. 找每行的最大票数
        max_votes = np.max(votes, axis=1, keepdims=True)
        
        #    2. 基础翻转逻辑: 票数等于最大值 且 最大值 > 0
        #       (对应 C++ 中的 Compare EQ 和 max > 0 判断)
        do_flip = np.logical_and(votes == max_votes, max_votes > 0).astype(np.uint8)
        
        #    3. 应用行掩码: 已经校验通过的行不翻转
        final_flip_mask = do_flip * valid_row_mask
        
        # F. 更新数据 (原地修改)
        #    current_data = current_data XOR flip_mask
        current_data = np.bitwise_xor(current_data, final_flip_mask)

    print("-" * 60)

    # =========================================================================
    # 5. 结果验证 (Verification)
    # =========================================================================
    print("Step 4: Final Verification")
    
    # 比较 译码后数据(current_data) 与 原始无误数据(codewords)
    # 使用 int8 比较
    decoded_result = current_data.astype(np.int8)
    golden_truth = codewords.astype(np.int8)
    
    diff_mask = (decoded_result != golden_truth)
    diff_count = np.sum(diff_mask)
    total_bits = M * N
    
    if diff_count == 0:
        print("[PASS] Decoding perfectly matches the original codewords!")
        print(f"       Recovered {initial_errors} errors completely.")
    else:
        print(f"[FAIL] Decoding mismatch.")
        print(f"       Remaining Bit Errors: {diff_count} / {total_bits}")
        print(f"       Error Rate: {diff_count/total_bits:.6f}")
        
        # 打印一下哪些帧失败了
        failed_frames = np.where(np.sum(diff_mask, axis=1) > 0)[0]
        print(f"       Failed Frames Indices: {failed_frames[:20]} ... (Total {len(failed_frames)})")
        
        # 检查是否因为矩阵本身有问题导致无法译码
        if non_zero_count > 0:
            print("       [Hint] Matrix check failed in Step 1, decoding failure is expected.")

    print("=" * 60)

if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    verify_ldpc_system()