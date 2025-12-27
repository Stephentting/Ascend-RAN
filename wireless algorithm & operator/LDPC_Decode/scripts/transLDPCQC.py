import numpy as np
import sys
import os

def parse_qc_file(filepath):
    """
    解析QC-LDPC文件，提取基矩阵参数和移位值。
    """
    with open(filepath, 'r') as f:
        content = f.read().split()

    # 1. 读取头部信息 (Columns, Rows, Expansion_Factor)
    # 根据你的文件内容: 20 12 512
    n_blk_cols = int(content[0])
    n_blk_rows = int(content[1])
    expansion_factor = int(content[2]) # Z

    print(f"检测到参数: Block Rows={n_blk_rows}, Block Cols={n_blk_cols}, Z={expansion_factor}")
    print(f"预期矩阵大小: {n_blk_rows * expansion_factor} x {n_blk_cols * expansion_factor}")

    # 2. 提取移位值 (Shift Values)
    # header 占了3个位置，我们需要读取 n_blk_rows * n_blk_cols 个数据
    num_entries = n_blk_rows * n_blk_cols
    shift_values = []
    
    current_idx = 3
    found_count = 0
    
    while found_count < num_entries and current_idx < len(content):
        val = content[current_idx]
        # 跳过可能的非数字字符（虽然split()通常已经处理好了）
        try:
            shift_val = int(val)
            shift_values.append(shift_val)
            found_count += 1
        except ValueError:
            pass # 忽略非整数字符
        current_idx += 1

    if len(shift_values) != num_entries:
        raise ValueError(f"文件格式错误：预期找到 {num_entries} 个移位值，但只找到 {len(shift_values)} 个。")

    # Reshape成 grid [Rows, Cols]
    shifts_grid = np.array(shift_values).reshape(n_blk_rows, n_blk_cols)
    
    return n_blk_rows, n_blk_cols, expansion_factor, shifts_grid

def generate_dense_h(n_rows, n_cols, z, shifts_grid):
    """
    根据移位值生成全尺寸的二进制校验矩阵 H。
    """
    # 初始化全零矩阵，使用 uint8 (0/1) 以节省空间
    # 尺寸 = (行块数 * Z) x (列块数 * Z)
    total_rows = n_rows * z
    total_cols = n_cols * z
    H = np.zeros((total_rows, total_cols), dtype=np.uint8)

    print("正在生成全尺寸 H 矩阵...")
    
    # 遍历基矩阵的每一个块
    for r in range(n_rows):
        for c in range(n_cols):
            shift = shifts_grid[r, c]
            
            # -1 表示全零块，跳过
            if shift >= 0:
                # 生成单位矩阵的循环移位
                # 快速算法：直接计算索引
                
                # 当前块在 H 中的起始行和列
                row_start = r * z
                col_start = c * z
                
                # 构造单位阵的 row 索引 [0, 1, ..., z-1]
                eye_rows = np.arange(z)
                # 构造循环移位后的 col 索引: (row_index + shift) % z
                # 注意：标准的 QC-LDPC 定义通常是向右循环移位
                eye_cols = (eye_rows + shift) % z
                
                # 将对应的位置置为 1
                H[row_start + eye_rows, col_start + eye_cols] = 1

    return H

def main():
    input_file = 'AR4JA_4096_8192.qc'
    output_bin = 'H_matrix_4096_8192_uint8.bin'
    
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    # 1. 解析
    m, n, z, shifts = parse_qc_file(input_file)
    
    # 2. 生成矩阵
    H_matrix = generate_dense_h(m, n, z, shifts)
    
    # 3. 统计信息
    ones_count = np.sum(H_matrix)
    sparsity = ones_count / H_matrix.size
    print(f"矩阵生成完毕。")
    print(f"非零元素个数: {ones_count}")
    print(f"稀疏度: {sparsity:.6f}")
    
    # 4. 保存为二进制文件
    # tofile 会将数组以 C-Order (行优先) 写入磁盘
    H_matrix.tofile(output_bin)
    
    print(f"\n成功保存到: {output_bin}")
    print(f"文件大小: {os.path.getsize(output_bin) / 1024 / 1024:.2f} MB")
    print("-" * 30)
    print("Host 读取提示 (C++):")
    print(f"Rows: {H_matrix.shape[0]}, Cols: {H_matrix.shape[1]}, Type: uint8_t")
    print(f"Buffer Size: {H_matrix.size} bytes")

if __name__ == "__main__":
    main()