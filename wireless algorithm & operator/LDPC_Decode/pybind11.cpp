#include <torch/extension.h>
#include "acl/acl.h"
#include "aclrtlaunch_matmul_custom.h"
#include "kernel_tiling/kernel_tiling.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tiling/platform/platform_ascendc.h"

// 声明外部生成 Tiling 的函数
extern "C" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf);

namespace ldpc_decoder {

// 静态资源：只在第一次调用时初始化
static void* g_tilingDevice = nullptr;
static void* g_workspaceDevice = nullptr;
// 预分配中间变量，避免在循环中重复申请内存
static void* g_c1Device = nullptr;
static void* g_cDevice = nullptr;
static void* g_maskDevice = nullptr;

void ensure_resources_init() {
    if (!g_tilingDevice) {
        // 1. Tiling 初始化
        aclrtMalloc(&g_tilingDevice, 4096, ACL_MEM_MALLOC_HUGE_FIRST);
        uint8_t tilingHost[4096];
        // 修复报错：使用 :: 显式指定全局命名空间
        ::GenerateTiling("Ascend310B1", tilingHost); 
        aclrtMemcpy(g_tilingDevice, 4096, tilingHost, 4096, ACL_MEMCPY_HOST_TO_DEVICE);
        
        // 2. Workspace 初始化
        auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        size_t total_ws = static_cast<size_t>(ascendc_platform->GetLibApiWorkSpaceSize()) + (256 * 256);
        aclrtMalloc(&g_workspaceDevice, total_ws, ACL_MEM_MALLOC_HUGE_FIRST);

        // 3. 中间变量内存预分配 (一次性分配)
        aclrtMalloc(&g_maskDevice, 256, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&g_c1Device, 256 * 256, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&g_cDevice, 256 * 512 * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    }
}

at::Tensor run_ldpc_decode_1192(at::Tensor &bits, const at::Tensor &h_matrix) {
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    ensure_resources_init();

    uint8_t* base_bits_ptr = (uint8_t*)bits.data_ptr();
    void* h_ptr = h_matrix.data_ptr();

    const int num_chunks = 12; 
    const int rows_per_chunk = 256;
    const int row_len = 512;

    // --- 极致性能模式 2.0：Loop Sinking ---
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        void* current_bits_ptr = (void*)(base_bits_ptr + chunk * rows_per_chunk * row_len);

        // 【修改点】：直接发射一次 Kernel，传入 max_iter 参数
        // 移除了 Host 侧的 for (i < max_iter) 循环
        ACLRT_LAUNCH_KERNEL(matmul_custom)(
            8, acl_stream,
            current_bits_ptr, h_ptr, g_c1Device, g_maskDevice, g_cDevice,
            g_workspaceDevice, g_tilingDevice// 传入迭代次数
        );
    }

    // aclrtSynchronizeStream(acl_stream); 
    return bits;
}
} // namespace ldpc_decoder

PYBIND11_MODULE(ldpc_custom, m) {
    m.def("run_ldpc_decode", &ldpc_decoder::run_ldpc_decode_1192);
}