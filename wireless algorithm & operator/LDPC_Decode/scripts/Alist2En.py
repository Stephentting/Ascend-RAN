import numpy as np
import sys

def gf2_rank(rows):
    """计算GF(2)下的矩阵秩"""
    rows = np.array(rows, dtype=np.uint8)
    n_rows, n_cols = rows.shape
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        if rows[pivot_row, col] == 0:
            nonzero = np.where(rows[pivot_row+1:, col] == 1)[0]
            if len(nonzero) == 0:
                continue
            rows[[pivot_row, pivot_row + 1 + nonzero[0]]] = rows[[pivot_row + 1 + nonzero[0], pivot_row]]
        
        # 消元
        others = np.where(rows[:, col] == 1)[0]
        others = others[others != pivot_row]
        if len(others) > 0:
            rows[others] ^= rows[pivot_row]
        pivot_row += 1
    return pivot_row

def get_generator_matrix(alist_file, output_file):
    print(f"读取 H 矩阵: {alist_file}")
    
    # --- 1. 读取 Alist ---
    with open(alist_file, 'r') as f:
        content = f.read().split()
    iterator = iter(content)
    n_cols = int(next(iterator)) # 504
    n_rows = int(next(iterator)) # 252
    max_col_weight = int(next(iterator))
    max_row_weight = int(next(iterator))
    
    # 跳过权重信息
    for _ in range(n_cols + n_rows): next(iterator)
    # 跳过列连接
    for _ in range(n_cols * max_col_weight): next(iterator)
    
    # 构建 H 矩阵
    H = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for r in range(n_rows):
        for _ in range(max_row_weight):
            val = int(next(iterator))
            if val > 0:
                H[r, val-1] = 1
                
    print("H 矩阵构建完成。正在进行高斯消元以获取系统形式...")

    # --- 2. 高斯消元变换为 [P^T | I] ---
    # 我们希望 H = [P^T | I]
    # 注意：PEG矩阵通常不是系统码，右边的子矩阵可能不可逆。
    # 这里我们尝试通过列交换找到可逆的子矩阵。
    
    H_work = H.copy()
    M = n_rows
    N = n_cols
    K = N - M # 252
    
    # 记录列交换的顺序，因为我们需要知道哪些是信息位，哪些是校验位
    col_permutation = np.arange(N)
    
    # 我们尝试把后 M 列变成单位矩阵
    # 也就是对 H 进行行变换 + 列交换，使其右边 M x M 为 I
    
    pivot_row = 0
    for col in range(N - M, N): # 目标是对齐右边的 M 列
        # 在当前列及之后的列中寻找主元
        # 注意：这里我们实际上是在对整个矩阵做 RREF，但在寻找 pivot 时
        # 我们优先保留右侧区域作为单位矩阵
        
        # 这里的逻辑稍微简化：我们将 H 变换为 [ P^T | I ]
        # 对 H 的右半部分 (N-M 到 N) 进行消元
        
        target_col_idx = col
        # 当前处理的行是 pivot_row
        
        # 1. 找主元
        if H_work[pivot_row, target_col_idx] == 0:
            # 往下找行
            found_row = np.where(H_work[pivot_row+1:, target_col_idx] == 1)[0]
            if len(found_row) > 0:
                # 交换行
                swap_r = pivot_row + 1 + found_row[0]
                H_work[[pivot_row, swap_r]] = H_work[[swap_r, pivot_row]]
            else:
                # 如果这列在下面全是0，我们需要从左边的列借一个过来（列交换）
                # 这说明原本选定的校验位之间线性相关
                print(f"警告: 列 {target_col_idx} 无法找到主元，尝试从信息位区域交换列...")
                
                found_col = -1
                # 在左边 (0 到 N-M-1) 或者 当前列右边寻找可交换的列
                # 为了保持系统形式，我们尽量只动左边的
                for candidate_c in range(0, N):
                    if candidate_c == target_col_idx: continue
                    if candidate_c >= N-M and candidate_c < target_col_idx: continue # 已经处理过的单位阵部分不动
                    
                    if H_work[pivot_row, candidate_c] == 1:
                        found_col = candidate_c
                        break
                
                if found_col != -1:
                    # 交换列 (H矩阵和记录索引的数组都要换)
                    H_work[:, [target_col_idx, found_col]] = H_work[:, [found_col, target_col_idx]]
                    col_permutation[[target_col_idx, found_col]] = col_permutation[[found_col, target_col_idx]]
                    print(f"  -> 交换了列 {target_col_idx} 和 {found_col}")
                else:
                    print("错误: 矩阵秩不足，无法构造生成矩阵！")
                    return

        # 2. 消元 (使该列其他行为 0)
        idx = np.where(H_work[:, target_col_idx] == 1)[0]
        for r_idx in idx:
            if r_idx != pivot_row:
                H_work[r_idx] ^= H_work[pivot_row]
        
        pivot_row += 1
        if pivot_row >= M: break

    # 现在 H_work 的形式 应该是 [ P^T | I ] (可能列序被打乱了)
    # 提取 P^T (即前 K 列)
    # 因为我们交换了列，所以现在的 H_work 对应的是打乱顺序后的码字 c'
    # c' = [u | p]
    # H_work = [P^T | I]
    # G_sys = [I | P]
    
    P_T = H_work[:, 0:K] # M x K
    P = P_T.T            # K x M
    
    I_K = np.eye(K, dtype=np.uint8)
    
    # 构造系统生成矩阵 G_sys = [I_K | P]
    G_sys = np.hstack((I_K, P)) # 尺寸 K x N (252 x 504)
    
    print(f"生成矩阵 G 构造完成。尺寸: {G_sys.shape}")
    print("注意：由于 PEG 矩阵非系统码，为了使其可逆进行了列交换。")
    print("这意味着：u * G 得到的码字，其比特顺序是经过重新排列的。")
    print("为了还原原始顺序，我们需要把列换回去。")
    
    # 还原列顺序
    # 我们构造出来的 G 产生的是 c_permuted = u * G_sys
    # c_original 的第 col_permutation[i] 位 = c_permuted 的第 i 位
    # 所以我们需要调整 G 的列顺序，使得 u * G_final 直接得到原始顺序的码字
    
    G_final = np.zeros_like(G_sys)
    # 把 G_sys 的第 i 列，放到 col_permutation[i] 的位置上
    for i in range(N):
        original_idx = col_permutation[i]
        G_final[:, original_idx] = G_sys[:, i]

    # --- 3. 保存 ---
    G_final.tofile(output_file)
    print(f"保存成功: {output_file} ({G_final.nbytes} 字节)")
    
    # 验证 check: G * H^T = 0
    print("正在验证 G * H^T == 0 ...")
    check = np.dot(G_final, H.T) % 2
    if np.all(check == 0):
        print("验证通过！矩阵完美正交。")
    else:
        print("验证失败！生成的矩阵有问题。")

if __name__ == "__main__":
    get_generator_matrix("PEGReg252x504.alist", "G_matrix_252x504.bin")