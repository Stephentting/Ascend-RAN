#include <torch/extension.h>
#include "acl/acl.h"
#include "aclrtlaunch_LDPCEnc_custom.h"
#include "kernel_tiling/kernel_tiling.h" 
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <iostream>

extern "C" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf);

namespace ldpc_encoder {

static void* g_tilingDevice = nullptr;
static void* g_workspaceDevice = nullptr;

at::Tensor run_ldpc_encode_1192(at::Tensor &bits_in, const at::Tensor &h_matrix) {
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    
    // --- 关键修正 1：统一 SoC 版本为 P3 ---
    const char* socVersion = "Ascend310B1"; 

    // 1. 静态资源初始化
    if (!g_tilingDevice) {
        size_t tilingFileSize = sizeof(TCubeTiling) + sizeof(uint64_t);
        uint8_t tilingHost[tilingFileSize];
        GenerateTiling(socVersion, tilingHost);
        
        aclrtMalloc(&g_tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(g_tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
        size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
        if (systemWorkspaceSize > 0) {
            aclrtMalloc(&g_workspaceDevice, systemWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }
    }

    // --- 关键修正 2：确保输入输出内存完全连续 ---
    auto a_contig = bits_in.contiguous();
    auto b_contig = h_matrix.contiguous();
    // 输出维度：12 组 * 256 行 = 3072 行，512 列
    auto output = at::empty({3072, 512}, bits_in.options().dtype(at::kShort));

    int8_t* in_ptr_base  = (int8_t*)a_contig.data_ptr();
    int16_t* out_ptr_base = (int16_t*)output.data_ptr();
    void* h_ptr          = b_contig.data_ptr();

    // 核心数同步 (通常 P3 上 Matmul 建议用 8)
    uint32_t blockDim = 8; 

    // --- 关键修正 3：修正指针算术偏移 ---
    for (int chunk = 0; chunk < 12; ++chunk) {
        // A 矩阵 (int8)：每组偏移 256 * 256 字节
        void* cur_in  = (void*)(in_ptr_base + chunk * 256 * 256);
        // C 矩阵 (int16)：每组偏移 256 * 512 个元素
        // C++ 的 int16_t* 指针加法会自动处理 *2 的字节跨度
        void* cur_out = (void*)(out_ptr_base + chunk * 256 * 512);

        ACLRT_LAUNCH_KERNEL(LDPCEnc_custom)(
            blockDim, acl_stream,
            cur_in, h_ptr, cur_out, 
            g_workspaceDevice, g_tilingDevice
        );
        // std::cout << "cur_in:" <<std::endl;
        // for (int i = 0; i < 100; ++i) {
        //     std::cout << (int)((int8_t*)cur_in)[i] << " ";
        // }
        // std::cout << std::endl;
    }

    aclrtSynchronizeStream(acl_stream);
    return output;
}

} // namespace ldpc_encoder

PYBIND11_MODULE(ldpc_encode_custom, m) {
    m.def("run_ldpc_encode", &ldpc_encoder::run_ldpc_encode_1192);
}