#!/usr/bin/env python3
# test_deliberate_error.py - 故意制造不同的golden结果来验证
import numpy as np

def create_different_golden():
    """
    故意创建一个不同的golden结果，看NPU输出是否真的不同
    """
    print("🔬 创建故意不同的参考结果...")
    
    # 读取原始golden
    original_golden = np.fromfile("output/golden.bin", dtype=np.float32)
    
    # 制造明显的差异
    modified_golden = original_golden + 0.1  # 加上0.1的偏移
    
    # 保存修改后的golden
    modified_golden.tofile("output/golden_modified.bin")
    
    print("✅ 已创建修改版golden结果 (全部+0.1)")
    return True

def test_with_modified_golden():
    """
    用修改后的golden测试
    """
    create_different_golden()
    
    print("\n🧪 测试NPU输出 vs 修改后的golden:")
    import subprocess
    result = subprocess.run([
        "python", "verify_result.py", 
        "output/output.bin", "output/golden_modified.bin"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    
    if "0.000%" in result.stdout:
        print("🚨 警告：即使golden被修改，误差仍然是0% - 可能有问题!")
        return False
    else:
        print("✅ 正常：修改golden后出现了预期的误差")
        return True

if __name__ == "__main__":
    test_with_modified_golden()