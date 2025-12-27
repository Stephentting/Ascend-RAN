/**
 * @file qam_mod_pybind.cpp
 * @brief QAM64 调制算子 Python 接口
 */
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"

// 引入自动生成的启动头文件
#include "aclrtlaunch_qam64_modulation.h"

// 声明外部 C 接口（对应算子工程中的实现）
extern "C" void qam64_modulation_do(uint32_t block_dim, void *stream,
                                   uint8_t *input_bits, 
                                   uint8_t *output_real, 
                                   uint8_t *output_imag);

/**
 * @brief 执行 QAM64 调制
 * @param input_bits 输入比特张量 [TOTAL_BITS], 类型 torch.uint8
 * @return std::tuple<at::Tensor, at::Tensor> (实部, 虚部) [TOTAL_SYMBOLS], 类型 torch.float16
 */
std::tuple<at::Tensor, at::Tensor> run_qam64_modulation(const at::Tensor& input_bits) {
    // 1. 获取当前 NPU 流
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    
    // 2. 维度计算
    int64_t total_bits = input_bits.numel();
    int64_t total_symbols = total_bits / 6; // QAM64 每 6 bits 映射 1 个符号

    // 3. 准备输出张量 (float16/half)
    auto options = at::TensorOptions().dtype(at::kHalf).device(input_bits.device());
    at::Tensor output_real = at::empty({total_symbols}, options);
    at::Tensor output_imag = at::empty({total_symbols}, options);

    // 4. 设置配置
    uint32_t blockDim = 8; // 算子中硬编码或根据 Tiling 获取的核数

    // 5. 启动核函数
    // 注意：input_bits.storage().data() 获取底层指针
    ACLRT_LAUNCH_KERNEL(qam64_modulation)(
        blockDim, 
        acl_stream,
        const_cast<void*>(input_bits.storage().data()),
        const_cast<void*>(output_real.storage().data()),
        const_cast<void*>(output_imag.storage().data())
    );

    return std::make_tuple(output_real, output_imag);
}

PYBIND11_MODULE(qam64_mod_custom, m) {
    m.doc() = "QAM64 Modulation operator for Ascend NPU";
    m.def("run_qam_mod", &run_qam64_modulation, "Run QAM64 Modulation Kernel",
          pybind11::arg("input_bits"));
}