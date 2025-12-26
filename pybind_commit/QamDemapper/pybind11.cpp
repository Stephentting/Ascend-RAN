/**
 * @brief QAM 解调算子的 Python 接口
 */
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

// 引入 CANN 自动生成的头文件，注意文件名规律：aclrtlaunch_<核函数名>.h
#include "aclrtlaunch_qam_demapper.h" 

// 声明你的核函数包装接口 (根据 ZF 的方式，我们可以直接传递 void*)
extern "C" void qam_demapper_do(uint32_t blockDim, void* stream, 
                                 float* input_I, float* input_Q, 
                                 uint8_t* output);



at::Tensor run_qam_demod(const at::Tensor& input_I, const at::Tensor& input_Q) {
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    
    int64_t num_elements = input_I.numel();
    auto options = at::TensorOptions().dtype(at::kByte).device(input_I.device());
    at::Tensor output = at::empty({num_elements * 6}, options);

    uint32_t blockDim = 8;

    // 使用 CANN 提供的宏进行启动
    // 这种方式会自动寻找 symbols，极大减少 undefined symbol 错误
    ACLRT_LAUNCH_KERNEL(qam_demapper)(
        blockDim, 
        acl_stream,
        const_cast<void*>(input_I.storage().data()),
        const_cast<void*>(input_Q.storage().data()),
        const_cast<void*>(output.storage().data())
    );

    return output;
}

// 定义 Pybind11 模块
PYBIND11_MODULE(qamdemapper_custom, m) {
    m.doc() = "QAM Demapper operator for Ascend NPU";
    m.def("run_qam_demod", &run_qam_demod, "Run QAM Demapper Kernel",
          pybind11::arg("input_I"), pybind11::arg("input_Q"));
}