import torch
import torch_npu
import numpy as np
import qam64_mod_custom # 假设这是你的 C++ 绑定模块

def verify_qam64_logic():
    device = "npu:0"
    k = 1 / np.sqrt(42.0)
    
    # 1. 定义标准映射字典 (基于上述内核逻辑)
    # Key: 3-bit 十进制值, Value: 对应电平
    standard_map = {
        0: -7*k, 1: -5*k, 2: -1*k, 3: -3*k,
        4:  7*k, 5:  5*k, 6:  1*k, 7:  3*k
    }

    # 2. 构造测试数据：覆盖 000000 到 111111 (共 64 个符号)
    test_bits = []
    for i in range(64):
        # 将 i 转为 6 位比特
        bits = [int(b) for b in format(i, '06b')]
        test_bits.extend(bits)
    
    input_bits_np = np.array(test_bits, dtype=np.uint8)
    input_bits_npu = torch.from_numpy(input_bits_np).to(device)

    # 3. 运行 NPU 算子
    print("🚀 Running NPU QAM64 Operator...")
    real_npu, imag_npu = qam64_mod_custom.run_qam_mod(input_bits_npu)
    
    res_real = real_npu.cpu().float().numpy()
    res_imag = imag_npu.cpu().float().numpy()

    # 4. 逐个符号验证
    print("\n🔍 Starting Verification...")
    errors = 0
    for i in range(64):
        # 提取当前符号对应的 I/Q 比特
        bits = test_bits[i*6 : (i+1)*6]
        i_bits_val = bits[0]*4 + bits[1]*2 + bits[2]
        q_bits_val = bits[3]*4 + bits[4]*2 + bits[5]
        
        expected_real = standard_map[i_bits_val]
        expected_imag = standard_map[q_bits_val]
        
        # 验证实部和虚部
        match_r = np.isclose(res_real[i], expected_real, atol=1e-3)
        match_i = np.isclose(res_imag[i], expected_imag, atol=1e-3)
        
        if not (match_r and match_i):
            errors += 1
            print(f"❌ Error at Symbol {i} (Bits:{bits}):")
            print(f"   Expected: {expected_real:.4f} + {expected_imag:.4f}j")
            print(f"   Got     : {res_real[i]:.4f} + {res_imag[i]:.4f}j")

    if errors == 0:
        print("\n✅ SUCCESS: NPU mapping matches standard Gray logic!")
    else:
        print(f"\n❌ FAILED: Found {errors} mapping errors.")

if __name__ == "__main__":
    verify_qam64_logic()