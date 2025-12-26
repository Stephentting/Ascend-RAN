import numpy as np
import os
import sys

def read_alist(alist_file):
    """读取 Alist 文件"""
    print(f"1. Reading {alist_file}...")
    try:
        with open(alist_file, 'r') as f:
            content = f.read().split()
    except FileNotFoundError:
        print(f"Error: {alist_file} not found.")
        sys.exit(1)

    iterator = iter(content)
    try:
        n_cols = int(next(iterator)) # 504
        n_rows = int(next(iterator)) # 252
        max_col_weight = int(next(iterator))
        max_row_weight = int(next(iterator))

        for _ in range(n_cols + n_rows + n_cols * max_col_weight): next(iterator)

        H = np.zeros((n_rows, n_cols), dtype=np.uint8)
        for r in range(n_rows):
            for _ in range(max_row_weight):
                val = int(next(iterator))
                if val > 0:
                    H[r, val - 1] = 1
        return H
    except StopIteration:
        print("Error: Alist file incomplete.")
        sys.exit(1)

def calculate_G(H_orig):
    """计算 G 矩阵"""
    print("2. Calculating Generator Matrix G...")
    H = H_orig.copy()
    M, N = H.shape
    K = N - M
    col_perm = np.arange(N)
    pivot_row = 0

    for col in range(N - M, N):
        if pivot_row >= M: break
        if H[pivot_row, col] == 0:
            swap_rows = np.where(H[pivot_row+1:, col] == 1)[0]
            if len(swap_rows) > 0:
                H[[pivot_row, pivot_row + 1 + swap_rows[0]]] = H[[pivot_row + 1 + swap_rows[0], pivot_row]]
            else:
                swap_cols = np.where(H[pivot_row, :N-M] == 1)[0]
                if len(swap_cols) > 0:
                    src = swap_cols[0]
                    H[:, [col, src]] = H[:, [src, col]]
                    col_perm[[col, src]] = col_perm[[src, col]]
                else:
                    continue
        rows_to_xor = np.where(H[:, col] == 1)[0]
        for r in rows_to_xor:
            if r != pivot_row:
                H[r] ^= H[pivot_row]
        pivot_row += 1

    P_T = H[:, :K]
    G_sys = np.hstack((np.eye(K, dtype=np.uint8), P_T.T))
    
    G_final = np.zeros_like(G_sys)
    for i in range(N):
        G_final[:, col_perm[i]] = G_sys[:, i]
        
    return G_final

def main():
    # ================= 维度配置 =================
    # 原始参数
    ORIG_M = 252  # 校验行数
    ORIG_N = 504  # 码长
    ORIG_K = ORIG_N - ORIG_M # 252 (信息位长度)

    # 目标参数 (NPU 32字节对齐)
    TARGET_M = 256 # 校验行补齐 (252->256)
    TARGET_N = 512 # 码长补齐 (504->512)
    # 信息位也要补齐，因为 G 的行数等于信息位长度
    TARGET_K = 256 # 信息位补齐 (252->256)
    
    INPUT_ALIST = "PEGReg252x504.alist"
    OUTPUT_DIR = "./matrices_padded"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 生成原始矩阵
    H_orig = read_alist(INPUT_ALIST)
    G_orig = calculate_G(H_orig) # (252, 504)

    # =======================================================
    # 3. 对 G 进行 Padding (252x504 -> 256x512)
    # =======================================================
    print(f"3. Padding Matrices to ({TARGET_M}x{TARGET_N})...")
    
    # G 的行数是信息位长度 (252)，列数是码长 (504)
    # Pad Rows: 252 -> 256 (补 4 行 0)
    # Pad Cols: 504 -> 512 (补 8 列 0)
    pad_rows_g = TARGET_K - ORIG_K
    pad_cols_g = TARGET_N - ORIG_N
    
    G_padded = np.pad(G_orig, ((0, pad_rows_g), (0, pad_cols_g)), 'constant')
    print(f"   G Padded Shape: {G_padded.shape}")

    # =======================================================
    # 4. 对 H 进行 Padding (252x504 -> 256x512)
    # =======================================================
    # H 的行数是校验方程数 (252)，列数是码长 (504)
    pad_rows_h = TARGET_M - ORIG_M
    pad_cols_h = TARGET_N - ORIG_N
    
    H_padded = np.pad(H_orig, ((0, pad_rows_h), (0, pad_cols_h)), 'constant')
    print(f"   H Padded Shape: {H_padded.shape}")
    
    # 转置 H (用于译码器 B 矩阵)
    H_transposed_padded = H_padded.T # (512, 256)

    # =======================================================
    # 5. 保存文件
    # =======================================================
    print("4. Saving binary files...")

    # 保存 G (256 x 512)
    G_padded.tofile(os.path.join(OUTPUT_DIR, "matrix_G_padded_256x512.bin"))
    print(f"   Saved G: {OUTPUT_DIR}/matrix_G_padded_256x512.bin")

    # 保存 H (256 x 512)
    H_padded.tofile(os.path.join(OUTPUT_DIR, "matrix_H_padded_256x512.bin"))
    print(f"   Saved H: {OUTPUT_DIR}/matrix_H_padded_256x512.bin")

    # 保存 H^T (512 x 256) -> 这是 x2_gm
    H_transposed_padded.tofile(os.path.join(OUTPUT_DIR, "matrix_H_transposed_padded_512x256.bin"))
    print(f"   Saved H^T: {OUTPUT_DIR}/matrix_H_transposed_padded_512x256.bin")
    
    # 验证逻辑
    print("\nVerification Info:")
    print("When using G for encoding (u * G):")
    print(f"  Input 'u' must be [Batch, {TARGET_K}]. (Last {pad_rows_g} cols must be 0)")
    print(f"  Output 'c' will be [Batch, {TARGET_N}]. (Last {pad_cols_g} cols will be 0)")

if __name__ == "__main__":
    main()