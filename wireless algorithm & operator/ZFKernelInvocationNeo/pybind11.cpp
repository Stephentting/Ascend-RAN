/**
 * @file zf_equalization_pybind11.cpp
 * 
 * ZF均衡算子的Pybind11封装
 * 支持复数输入(实部+虚部分离的格式)
 */
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "aclrtlaunch_zf_equalization.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace ofdm_zf {

/**
 * @brief ZF均衡算子的Python接口封装
 * 
 * @param h_real 信道估计实部 [batch_size, num_subcarriers]
 * @param h_imag 信道估计虚部 [batch_size, num_subcarriers]
 * @param y_real 接收信号实部 [batch_size, num_subcarriers]
 * @param y_imag 接收信号虚部 [batch_size, num_subcarriers]
 * @return std::vector<at::Tensor> 返回均衡后的信号 [x_hat_real, x_hat_imag]
 */
std::vector<at::Tensor> run_zf_equalization(const at::Tensor &h_real, 
                                             const at::Tensor &h_imag,
                                             const at::Tensor &y_real, 
                                             const at::Tensor &y_imag)
{
    // 获取当前NPU流
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    
    // 分配输出内存(与输入形状相同)
    at::Tensor x_hat_real = at::empty_like(h_real);
    at::Tensor x_hat_imag = at::empty_like(h_imag);
    
    // 设置块维度(使用8个AI Core)
    uint32_t blockDim = 8;
    
    // 调用ZF均衡核函数
    // 注意:你的算子设计为固定处理 32 batch * 256 subcarriers = 8192个元素
    ACLRT_LAUNCH_KERNEL(zf_equalization)(
        blockDim, 
        acl_stream,
        const_cast<void *>(h_real.storage().data()),
        const_cast<void *>(h_imag.storage().data()),
        const_cast<void *>(y_real.storage().data()),
        const_cast<void *>(y_imag.storage().data()),
        const_cast<void *>(x_hat_real.storage().data()),
        const_cast<void *>(x_hat_imag.storage().data())
    );
    
    // 返回实部和虚部
    return {x_hat_real, x_hat_imag};
}

} // namespace ofdm_zf

// 定义Pybind11模块
PYBIND11_MODULE(zf_equalization, m)
{
    m.doc() = "ZF Equalization operator for OFDM system on Ascend NPU";
    
    // 绑定ZF均衡函数
    m.def("run_zf_equalization", 
          &ofdm_zf::run_zf_equalization, 
          "ZF equalization for OFDM received signals",
          pybind11::arg("h_real"),
          pybind11::arg("h_imag"),
          pybind11::arg("y_real"),
          pybind11::arg("y_imag"));
}