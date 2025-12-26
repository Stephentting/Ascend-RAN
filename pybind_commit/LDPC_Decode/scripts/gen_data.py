# import numpy as np
# import os

# def python_simulate_ldpc(received_bits, H_transposed, perfect_codewords, max_iter=1):
#     """
#     模拟C++算子的行为进行LDPC译码 (Bit-Flipping算法)
    
#     Args:
#         received_bits: [M, N], 接收到的带噪数据
#         H_transposed: [N, K], 校验矩阵 H^T (即 x2_gm)
#         max_iter: 迭代次数，与 main.cpp 中保持一致 (20)
#     Returns:
#         decoded_bits: [M, N], 译码后的结果
#     """
#     print(f"Starting Python Simulation for {max_iter} iterations...")
    
#     # 复制一份数据，避免修改原始输入
#     current_bits = received_bits.astype(np.int32).copy()
    
#     # H_transposed 是 [N, K]，我们需要 H [K, N] 用于计算 Vote
#     # 对应 C++ 中 mm2 的输入 B (SetTensorB(..., true))
#     H_forward = H_transposed.T 
    
#     M, N = current_bits.shape
    
#     for i in range(max_iter):
#         # =====================================================================
#         # Step 1: 第一次矩阵乘法 (计算校验子 Syndrome)
#         # =====================================================================
#         # C++: Matmul(A, B) -> int32 -> Cast to int16 -> AND 1 -> Cast to int8
#         # 逻辑上等同于矩阵乘法后模 2
        
#         # [M, N] x [N, K] -> [M, K]
#         syndromes_raw = np.matmul(current_bits, H_transposed.astype(np.int32))
#         syndromes = syndromes_raw % 2
        
#         # =====================================================================
#         # Step 2: 检查 Mask (校验子求和)
#         # =====================================================================
#         # C++: ReduceSum on Syndromes per row
#         row_check_sum = np.sum(syndromes, axis=1)
        
#         # 统计完全正确的行数
#         converged_count = np.sum(row_check_sum == 0)
#         # print(f"  Iter {i}: Converged Rows = {converged_count}/{M}")
        
#         if converged_count == M:
#             print(f"  Simulation converged at iteration {i}")
#             break

#         # =====================================================================
#         # Step 3: 第二次矩阵乘法 (计算投票 Votes)
#         # =====================================================================
#         # C++: Matmul(Syndrome, H) -> Votes
#         # [M, K] x [K, N] -> [M, N]
#         votes = np.matmul(syndromes, H_forward.astype(np.int32))
        
#         # =====================================================================
#         # Step 4: 比特翻转 (LdpcFlipping Kernel Logic)
#         # =====================================================================
#         # 这一步我们必须严格模拟 C++ 核函数的逻辑
        
#         print(f"--- Iteration {i} Flipping Details (First 32 Rows) ---")
        
#         # 遍历每一行 (对应核函数中的 Process -> Compute)
#         for r in range(M):
#             # 如果该行校验和为0，C++中会 continue (跳过)
#             if row_check_sum[r] == 0:
#                 continue
            
#             row_votes = votes[r] # [N]
            
#             # 1. 找最大值 (ReduceMax)
#             max_val = np.max(row_votes)
            
#             # 2. 生成 Mask (Compare EQ)
#             # 找出所有票数等于最大值的比特位置
#             flip_mask = (row_votes == max_val)
            
#             # 3. 阈值判断 (对应 C++ 中的 fillValue逻辑)
#             # C++ 代码: float fillValue = (maxVal > 0.0f) ? 1.0f : 0.0f;
#             if max_val > 0:
#                 # =========================================================
#                 # 【新增】打印前32行的翻转索引
#                 # =========================================================
#                 if r >= 32 and r < 64:
#                     # np.where 返回的是 tuple，取第一个元素即为索引数组
#                     flipped_indices = np.where(flip_mask)[0]
#                     print(f"  [Iter {i}][Row {r:03d}] MaxVote={max_val} | Flip Indices: {flipped_indices}")
#                 # =========================================================

#                 # 执行翻转 XOR
#                 # flip_mask 是 boolean，转换成 0/1 进行异或
#                 current_bits[r] = np.bitwise_xor(current_bits[r], flip_mask.astype(np.int32))
                
#         print("iteration", i, "completed.")
#         VALID_LEN = 504
#         diff_count = np.sum(current_bits != perfect_codewords)
#         print(f"Total remaining bit errors after Python simulation: {diff_count}")
#     return current_bits.astype(np.int8)

# def gen_golden_data():
#     # 设定维度参数
#     M = 256  # 帧数
#     N = 512  # 码长
#     K = 256  # 信息位长

#     print(f"Generating data with M={M}, N={N}, K={K}...")

#     # =========================================================================
#     # 1. 读取矩阵
#     # =========================================================================
#     try:
#         x2_gm = np.fromfile("matrix_H_transposed_padded_512x256.bin", dtype=np.uint8).reshape([N, K])
#     except FileNotFoundError:
#         print("Error: matrix_H_transposed_padded_512x256.bin not found.")
#         print("Warning: Generating random dummy matrix...")
#         x2_gm = np.random.randint(0, 2, [N, K]).astype(np.uint8)

#     try:
#         G_matrix = np.fromfile("matrix_G_padded_256x512.bin", dtype=np.uint8).reshape([K, N])
#     except FileNotFoundError:
#         print("Error: matrix_G_padded_256x512.bin not found.")
#         print("Warning: Generating random dummy matrix...")
#         G_matrix = np.random.randint(0, 2, [K, N]).astype(np.uint8)

#     # =========================================================================
#     # 2. 编码 (生成标准答案)
#     # =========================================================================
#     msgs = np.random.randint(0, 2, [M, K]).astype(np.uint8)
#     # 理想的无误码数据
#     perfect_codewords = np.matmul(msgs, G_matrix) % 2
    
#     # =========================================================================
#     # 3. 注入错误 (生成输入数据)
#     # =========================================================================
#     error_rate = 0.014 # 0.5% 误码率
#     error_mask = np.random.choice(
#         [0, 1], 
#         size=perfect_codewords.shape, 
#         p=[1 - error_rate, error_rate]
#     ).astype(np.uint8)
    
#     # x1_gm: 输入给 NPU 的带噪数据
#     x1_gm = np.bitwise_xor(perfect_codewords.astype(np.uint8), error_mask)
    
#     print(f"Data generated. Introduced approx {np.sum(error_mask)} bit errors.")

#     # =========================================================================
#     # 4. 生成 Golden Data (使用 Python 仿真你的算子逻辑)
#     # =========================================================================
    
#     print("Running Python simulation to generate Golden Data...")
#     # 注意：这里我们使用仿真结果作为 Golden，而不是 perfect_codewords。
#     # 目的：验证 NPU 算子代码实现是否与 Python 逻辑一致。
#     # 如果你想看算法是否能解出正确答案，可以对比 simulated_output 和 perfect_codewords。
    
#     simulated_output = python_simulate_ldpc(x1_gm, x2_gm, perfect_codewords, max_iter=20)
#     # golden = simulated_output
#     golden = perfect_codewords
#     # =========================================================================
#     # 打印对比信息
#     # =========================================================================
#     np.set_printoptions(threshold=np.inf, linewidth=np.inf)
#     print("\n" + "="*80)
#     print("Debug Output")
#     print("="*80)
    
#     # 对比 Python 仿真结果与完美结果 (评估算法能力)
#     diff_count = np.sum(simulated_output != perfect_codewords)
#     print(f"Total remaining bit errors after Python simulation: {diff_count}")
#     if diff_count == 0:
#         print(">> Python logic decoded ALL errors successfully!")
#     else:
#         print(">> Python logic failed to decode all errors. (Algorithm limitation or iterations)")

#     print("="*80 + "\n")
#     np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)

#     # =========================================================================
#     # 5. 保存文件
#     # =========================================================================
#     os.system("mkdir -p input")
#     os.system("mkdir -p output")
    
#     x1_gm.tofile("./input/x1_gm.bin")
#     x2_gm.tofile("./input/x2_gm.bin")
    
#     # 保存仿真结果作为 Golden
#     golden.tofile("./output/golden.bin")
    
#     print("Done. Files saved to ./input/ and ./output/")

# if __name__ == "__main__":
#     gen_golden_data()

#!/usr/bin/python3
# coding=utf-8
#!/usr/bin/python3
# coding=utf-8
import numpy as np
import os

# 定义有效数据长度常量
VALID_CODE_LEN = 504
VALID_INFO_LEN = 252

def python_simulate_ldpc(received_bits, H_transposed, perfect_codewords, max_iter=1):
    print(f"Starting Python Simulation for {max_iter} iterations...")
    
    current_bits = received_bits.astype(np.int32).copy()
    H_forward = H_transposed.T 
    
    M, N = current_bits.shape
    
    # --- 打印初始误码数 ---
    initial_errors = np.sum(current_bits[:, :VALID_CODE_LEN] != perfect_codewords[:, :VALID_CODE_LEN])
    print(f"Initial Valid Errors (Before Decoding): {initial_errors}")

    for i in range(max_iter):
        # 1. 校验子计算
        syndromes_raw = np.matmul(current_bits, H_transposed.astype(np.int32))
        syndromes = syndromes_raw % 2
        
        # 2. 检查收敛
        row_check_sum = np.sum(syndromes, axis=1)
        converged_count = np.sum(row_check_sum == 0)
        
        if converged_count == M:
            print(f"  >> Simulation converged at iteration {i} (All Parity Checks Passed)")
            break

        # 3. 投票
        votes = np.matmul(syndromes, H_forward.astype(np.int32))
        
        # 4. 翻转
        for r in range(M):
            if row_check_sum[r] == 0:
                continue
            
            row_votes = votes[r] 
            max_val = np.max(row_votes)
            
            # Padding 区域票数为0，不会翻转，保留原样
            if max_val > 0:
                flip_mask = (row_votes == max_val)
                current_bits[r] = np.bitwise_xor(current_bits[r], flip_mask.astype(np.int32))
        
        # =================================================================
        # 【新增】打印每一轮翻转后的错误数
        # =================================================================
        # 注意：必须使用切片 [:VALID_CODE_LEN]，忽略 Padding 区域的噪声
        current_errors = np.sum(current_bits[:, :VALID_CODE_LEN] != perfect_codewords[:, :VALID_CODE_LEN])
        print(f"  [Iteration {i}] Post-Flip Valid Errors: {current_errors}")
        
    return current_bits.astype(np.int8)

def gen_golden_data():
    # 昇腾对齐后的维度
    M = 256  # Batch Size
    N = 512  # Padded Code Length
    K = 256  # Padded Check Nodes

    print(f"Generating realistic noisy data (Padding Included)...")

    # 1. 读取矩阵
    try:
        # 请确保路径正确
        x2_gm = np.fromfile("./matrix_H_transposed_padded_512x256.bin", dtype=np.uint8).reshape([N, K])
        G_matrix = np.fromfile("./matrix_G_padded_256x512.bin", dtype=np.uint8).reshape([K, N])
    except FileNotFoundError:
        print("Error: Matrix files not found. Please run gen_matrices_only.py first.")
        return

    # 2. 编码
    msgs = np.random.randint(0, 2, [M, K]).astype(np.uint8)
    msgs[:, VALID_INFO_LEN:] = 0 # 信息位 Padding 置 0
    
    perfect_codewords = np.matmul(msgs, G_matrix) % 2
    
    # 3. 注入错误 (全量注入，Padding 区域也带噪)
    error_rate = 0.015 
    error_mask = np.random.choice(
        [0, 1], 
        size=perfect_codewords.shape, 
        p=[1 - error_rate, error_rate]
    ).astype(np.uint8)
    
    # 这里我们允许 Padding 区域带噪声，模拟真实信道
    x1_gm = np.bitwise_xor(perfect_codewords.astype(np.uint8), error_mask)
    
    print(f"Data generated.")

    # 4. 仿真验证
    print("\n" + "="*40)
    print("Running Python simulation...")
    print("="*40)
    simulated_output = python_simulate_ldpc(x1_gm, x2_gm, perfect_codewords, max_iter=20)
    
    # 5. 结果对比
    diff_count = np.sum(simulated_output[:, :VALID_CODE_LEN] != perfect_codewords[:, :VALID_CODE_LEN])
    
    print("\n" + "="*80)
    if diff_count == 0:
        print(">> SUCCESS! Decoded all VALID errors.")
    else:
        print(f">> FAILED. Remaining valid errors: {diff_count}")
    print("="*80 + "\n")

    golden = simulated_output
    # 6. 保存文件
    os.system("mkdir -p input output")
    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    # perfect_codewords.tofile("./output/golden.bin")
    golden.tofile("./output/golden.bin")
    print("Files saved.")

if __name__ == "__main__":
    gen_golden_data()