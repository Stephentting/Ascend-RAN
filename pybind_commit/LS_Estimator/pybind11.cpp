/**
 * @file pybind11.cpp
 * 用于将 OFDM LS 信道估计算子封装为 PyTorch 接口
*/
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cstring>

#include "acl/acl.h"
#include "aclrtlaunch_matmul_custom.h" // 对应你的算子名
#include "kernel_tiling/kernel_tiling.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tiling/platform/platform_ascendc.h"

// 链接你在 matmul_custom_tiling.cpp 中定义的函数
extern "C" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf);

namespace my_ls_estimator {

at::Tensor run_ls_estimator_custom(const at::Tensor &a, const at::Tensor &b)
{
    // 1. 获取当前 NPU Stream
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    
    // 2. 获取维度信息 [M, K] x [K, N]
    // a: [Batch, 32], b: [32, 512]
    auto M = a.sizes()[0];
    auto K = a.sizes()[1];
    auto N = b.sizes()[1];

    // 3. 创建输出 Tensor [M, N] -> [1192, 512] float32
    auto c = at::empty({M, N}, a.options().dtype(at::kFloat));

    // 4. 计算并准备 Workspace (系统 + 用户)
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    size_t system_workspace_size = static_cast<size_t>(ascendc_platform->GetLibApiWorkSpaceSize());
    auto workspace_tensor = at::empty({(long)system_workspace_size}, 
                                     at::TensorOptions().dtype(at::kByte).device(a.options().device()));

    // 5. Tiling 准备 (包含 TCubeTiling 和末尾的 localMemSize)
    size_t tilingFileSize = sizeof(TCubeTiling) + sizeof(uint64_t);
    uint8_t *tilingHost;
    uint8_t *tilingDevice;

    // 分配 Host 和 Device 侧 Tiling 内存
    aclrtMallocHost((void **)(&tilingHost), tilingFileSize);
    aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 调用你之前的 GenerateTiling 函数生成 8 核切分参数
    // 注意：确保 SOC_VERSION 宏在编译时已定义，如 "Ascend310B1"
    const char* SOC_VERSION = "Ascend310B1";
    GenerateTiling(SOC_VERSION, tilingHost);

    // 将 Tiling 数据同步到 Device
    aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // 6. 获取 blockDim (从 Tiling 结果中动态获取 usedCoreNum = 8)
    uint32_t blockDim = reinterpret_cast<TCubeTiling *>(tilingHost)->usedCoreNum;

    // 7. 启动 Kernel
    // 参数顺序：(blockDim, stream, A_ptr, B_ptr, C_ptr, Workspace_ptr, Tiling_ptr)
    ACLRT_LAUNCH_KERNEL(matmul_custom)
    (blockDim, acl_stream, 
     const_cast<void *>(a.data_ptr()), 
     const_cast<void *>(b.data_ptr()),
     const_cast<void *>(c.data_ptr()),
     const_cast<void *>(workspace_tensor.data_ptr()), 
     tilingDevice);

    // 8. 清理 Tiling 内存 (Device 侧建议异步清理或使用特定的管理逻辑，此处为同步示例)
    // 注意：在实际高频调用中，建议将 Tiling 缓存以避免频繁 Malloc
    aclrtFreeHost(tilingHost);
    // aclrtFree(tilingDevice); // 若要极致性能，tilingDevice 可在 Python 层维护周期

    return c;
}

} // namespace my_ls_estimator

// 注册 Pybind11 模块
PYBIND11_MODULE(matmul_LS_custom, m)
{
    m.doc() = "AscendC LS Estimator Pybind11 plugin";
    m.def("run_ls_estimator", &my_ls_estimator::run_ls_estimator_custom, "Run LS estimation on NPU");
}