/**
 * @file matmul_custom_tiling.cpp
 */
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;

extern "C" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf) {
    // 保持维度一致
    constexpr int32_t M = 1192; 
    constexpr int32_t N = 512;
    constexpr int32_t K = 32;
    
    // ★ 修改点：为了开启8核，计算 SINGLECORE_M = 1192 / 8 = 149
    // 这样 M 方向会被切分为 8 份，分配给 8 个核
    constexpr int32_t SINGLECORE_M = 149; 
    constexpr int32_t SINGLECORE_N = 512; // N 轴不切分，保持单核处理全宽
    
    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);
    
    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    
    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    
    if (ascendcPlatform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND310B) {
        tilingApi.SetSingleShape(SINGLECORE_M, SINGLECORE_N, -1);
        
        // 计算总块数：(1192 / 149) * (512 / 512) = 8 * 1 = 8
        int32_t mBlockNum = M / SINGLECORE_M;
        int32_t nBlockNum = N / SINGLECORE_N;
        int32_t totalBlocks = mBlockNum * nBlockNum;
        
        tilingApi.SetDim(totalBlocks); // 设置总核数
        // printf("Tiling Config: M=%d, SINGLECORE_M=%d, Cores=%d\n", M, SINGLECORE_M, totalBlocks);
    } else {
        tilingApi.SetDim(ascendcPlatform->GetCoreNumAiv());
    }
    
    tilingApi.SetBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);
    tilingApi.GetTiling(tilingData);
    
    uint32_t tcubeTilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);
    
    uint64_t localMemSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    *reinterpret_cast<uint64_t *>(tilingBuf + tcubeTilingSize) = localMemSize;
}