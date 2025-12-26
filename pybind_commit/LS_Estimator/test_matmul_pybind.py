#!/usr/bin/python3
# coding=utf-8
import torch
import torch_npu
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
import sys
import os
import time

# 将当前目录加入路径以便导入编译好的 pybind 模块
sys.path.append(os.getcwd())
try:
    import matmul_LS_custom
except ImportError:
    print("错误: 找不到 matmul_LS_custom 模块，请确保已完成编译并在当前目录下。")
    sys.exit(1)

class TestLSEstimatorCustom(TestCase):
    def test_ls_estimator_performance_and_precision(self):
        # 1. 参数配置
        BATCH_SIZE = 1192
        K_DIM = 32
        N_DIM = 512
        WARMUP_ITERS = 20    # 预热次数
        TEST_ITERS = 100     # 正式统计耗时次数

        # 2. 读取数据
        x1_path = "../input/x1_gm.bin"
        x2_path = "../input/x2_gm.bin"
        golden_path = "../output/golden.bin"

        if not (os.path.exists(x1_path) and os.path.exists(x2_path) and os.path.exists(golden_path)):
            print("错误: 找不到输入数据文件。")
            return

        x1_np = np.fromfile(x1_path, dtype=np.float16).reshape(BATCH_SIZE, K_DIM)
        x2_np = np.fromfile(x2_path, dtype=np.float16).reshape(K_DIM, N_DIM)
        golden_np = np.fromfile(golden_path, dtype=np.float32).reshape(BATCH_SIZE, N_DIM)

        # 搬运到 NPU
        a = torch.from_numpy(x1_np).npu()
        b = torch.from_numpy(x2_np).npu()
        golden = torch.from_numpy(golden_np).npu()

        print(f"--- 开始性能与精度测试 (Batch={BATCH_SIZE}) ---")

        # 3. 预热 (Warmup)
        # 目的：初始化上下文、加载算子二进制文件到指令缓存
        print(f"正在预热 {WARMUP_ITERS} 次...")
        for _ in range(WARMUP_ITERS):
            _ = matmul_LS_custom.run_ls_estimator(a, b)
        torch.npu.synchronize() # 等待预热任务在 NPU 上全部完成

        # 4. 性能测试 (Timing)
        print(f"正在运行性能测试 {TEST_ITERS} 次...")
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)

        start_event.record()
        for _ in range(TEST_ITERS):
            output = matmul_LS_custom.run_ls_estimator(a, b)
        end_event.record()
        
        torch.npu.synchronize() # 确保所有计算完成
        # 计算毫秒耗时并求平均
        elapsed_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_time_ms / TEST_ITERS

        # 5. 精度验证与详细数据打印
        print("\n" + "="*30)
        print("📊 [精度统计 Precision Stats]")
        
        # 计算误差
        diff = torch.abs(output - golden)
        max_err = torch.max(diff).item()
        mean_err = torch.mean(diff).item()
        
        print(f"最大绝对误差 (Max Error): {max_err:.6f}")
        print(f"平均绝对误差 (Mean Error): {mean_err:.6f}")
        
        # 打印多组数据对比 (头部、中间、尾部)
        def print_sample(name, tensor):
            t = tensor.cpu().numpy()
            print(f"{name} 样例:")
            print(f"  前5个: {t[0, :5]}")
            print(f"  中间5个: {t[BATCH_SIZE//2, N_DIM//2 : N_DIM//2+5]}")
            print(f"  末尾5个: {t[-1, -5:]}")

        print_sample("NPU 输出", output)
        print_sample("Reference 输出", golden)
    
        print("\n⏱️ [性能统计 Performance Stats]")
        print(f"平均运行耗时: {avg_time_ms:.4f} ms")
        print(f"单次理论吞吐量 (假设): { (BATCH_SIZE * K_DIM * N_DIM * 2) / (avg_time_ms / 1000) / 1e12 :.2f} TFLOPS")
        print("="*30 + "\n")

        # 5. 深度精度验证 (跳过空子载波)
        print("\n" + "="*30)
        print("🔍 [有效数据提取验证]")

        # 找到 Reference 中第一个非零元素的索引
        # golden 的形状是 [1192, 512]
        nonzero_indices = torch.nonzero(golden)

        if nonzero_indices.shape[0] == 0:
            print("❌ 警告: 参考输出 (Golden) 全是 0！请检查输入数据或生成逻辑。")
        else:
            # 取第一个非零点的位置
            first_idx = nonzero_indices[0]
            r, c = first_idx[0].item(), first_idx[1].item()
            
            # 确保切片不越界
            c_start = max(0, c)
            c_end = min(N_DIM, c + 8)

            print(f"检测到有效数据起始位置: Batch={r}, Column={c}")
            print(f"NPU 对应片段: {output[r, c_start:c_end].cpu().numpy()}")
            print(f"Ref 对应片段: {golden[r, c_start:c_end].cpu().numpy()}")

            # 计算该片段的误差
            segment_err = torch.abs(output[r, c_start:c_end] - golden[r, c_start:c_end]).max()
            print(f"该片段最大误差: {segment_err.item():.6f}")

        # 6. 全局统计
        max_err = torch.max(torch.abs(output - golden)).item()
        print(f"\n全局最大绝对误差: {max_err:.6f}")
        
        # 统计非零占比，确认数据分布
        npu_nz_count = torch.count_nonzero(output).item()
        total_elements = output.numel()
        print(f"数据活跃度 (非零占比): {npu_nz_count / total_elements * 100:.2f}%")

        # 断言精度是否合格
        self.assertRtolEqual(output, golden, prec=1e-3)

if __name__ == "__main__":
    if not torch.npu.is_available():
        print("错误: NPU 环境不可用")
        sys.exit(1)
    run_tests()