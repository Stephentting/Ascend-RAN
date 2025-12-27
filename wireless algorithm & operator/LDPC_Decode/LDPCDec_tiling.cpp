/**
 * @file matmul_custom_tiling.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;
using namespace std;
#define TILING_OFFSET_2 2048
/**
  * @brief  Generate matmul tiling.
  * @param  socVersion: Platform socversion.
  * @param  tilingBuf data buffer.
  */
// void GenerateTiling(const char *socVersion, uint8_t *tilingBuf)
// {
//     constexpr int32_t M = 256;
//     constexpr int32_t N = 256;
//     constexpr int32_t K = 512;

//     TPosition leftPosition = TPosition::GM;
//     CubeFormat leftFormat = CubeFormat::ND;
//     DataType leftDtype = DataType::DT_INT8;
//     bool isTransA = false;

//     TPosition rightPosition = TPosition::GM;
//     CubeFormat rightFormat = CubeFormat::ND;
//     DataType rightDtype = DataType::DT_INT8;
//     bool isTransB = false;

//     TPosition resultPosition = TPosition::VECCALC;
//     CubeFormat resultFormat = CubeFormat::ND;
//     DataType resultDtype = DataType::DT_INT32;

//     bool isBias = false;

//     constexpr int32_t SINGLECORE_M = 32;    //开启4个核
//     constexpr int32_t SINGLECORE_N = 256;   //  816/8 = 102

//     optiling::TCubeTiling tilingData;
//     auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
//     MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

//     tilingApi.SetDim(ascendcPlatform->GetCoreNumAic()); // Set the number of cores that participate in multi-core computaion is 48.
//     std::cout<<"Core的个数为："<<ascendcPlatform->GetCoreNumAic()<<std::endl;//310B这个GetCoreNumAiv返回1，操！
    
//     tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
//     tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
//     tilingApi.SetCType(resultPosition, resultFormat, resultDtype);

//     tilingApi.SetOrgShape(M, N, K);
//     tilingApi.SetShape(M, N, K);
//     if (ascendcPlatform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND310B) {
//         tilingApi.SetSingleShape(SINGLECORE_M, SINGLECORE_N, -1);  // Set the fixed singleCoreM=512, singleCoreN=512.
//         int32_t mBlockNum = M / SINGLECORE_M;
//         int32_t nBlockNum = N / SINGLECORE_N;
//         tilingApi.SetDim(mBlockNum * nBlockNum);
//         std::cout<<"Core的个数修改为："<<mBlockNum * nBlockNum<<std::endl;
//     }
//     tilingApi.SetBias(isBias);
//     tilingApi.SetBufferSpace(-1, -1, -1);

//     int64_t res = tilingApi.GetTiling(tilingData); // Get matmul tiling data.
//     if (res == -1) {
//         std::cout << "gen tiling failed" << std::endl;
//     }

//     // 2. 生成 Tiling 2 (Half x Half -> Float)
//     // =========================================================
//     MultiCoreMatmulTiling api2(*ascendcPlatform);
//     optiling::TCubeTiling data2;
    


//     // MM2 参数: M=256, N=512, K=256 (假设是 MM1结果 x H^T)
//     // 关键点：输入A在VECCALC(UB)，输入B在GM，都是Half
//     api2.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false); 
//     api2.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, true); 
//     api2.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32); 
//     api2.SetOrgShape(256, 512, 256);
//     api2.SetShape(256, 512, 256);

//      int32_t coreNum = ascendcPlatform->GetCoreNumAic();
//     if(coreNum == 0) coreNum = 1; // 310B 模拟器保护
//     api2.SetDim(coreNum); 
//     printf("CoreNum = %d\n", coreNum);
//     // 【缺失的部分 2】：设置切分策略 (针对 310B)
//     if (ascendcPlatform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND310B) {
//         // MM2 的单核目标：M=32, N=512 (因为 MM2 的 N 是 512)
//         // 注意：这里的 N 必须对应 api2.SetShape 中的 N
//         constexpr int32_t SINGLECORE_M = 32;
//         constexpr int32_t SINGLECORE_N_MM2 = 512; 
        
//         api2.SetSingleShape(SINGLECORE_M, SINGLECORE_N_MM2, -1); 
        
//         // 重新计算参与的 Block 数
//         int32_t mBlockNum = 256 / SINGLECORE_M;      // 8
//         int32_t nBlockNum = 512 / SINGLECORE_N_MM2;  // 1
//         api2.SetDim(mBlockNum * nBlockNum);
//         printf("针对310B，CoreNum = %d\n", mBlockNum * nBlockNum);
//     }

//     api2.SetBufferSpace(-1, -1, -1);

//     if (api2.GetTiling(data2) == -1) {
//         std::cerr << "Gen Tiling2 failed" << std::endl;
//     }

//     uint32_t tcubeTilingSize = sizeof(optiling::TCubeTiling);
//     tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);
//     printf("tilingData.GetDataSize():%ld\n",tilingData.GetDataSize());
//     printf("sizeof(optiling::TCubeTiling):%ld\n",sizeof(optiling::TCubeTiling));
//     uint64_t ubSize;
//     ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
//     *reinterpret_cast<uint64_t *>(tilingBuf + tcubeTilingSize) = ubSize;

//     data2.SaveToBuffer(tilingBuf + TILING_OFFSET_2, tcubeTilingSize );
//     printf("data2.GetDataSize():%ld\n",data2.GetDataSize());
//     *reinterpret_cast<uint64_t *>(tilingBuf + TILING_OFFSET_2 + sizeof(optiling::TCubeTiling)) = ubSize;
//     return;
// }

extern "C" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf)
{
    // =========================================================
    // 1. 全局配置 & 获取 UB 大小
    // =========================================================
    constexpr int32_t M = 256;
    constexpr int32_t N = 256;
    constexpr int32_t K = 512;
    
    // 【关键】：强制两个矩阵乘法都在 M 维度切分为 32 行
    constexpr int32_t SINGLECORE_M = 32; 
    
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // =========================================================
    // 2. 生成 Tiling 1 (Bits * H^T -> Syndrome)
    // Shape: [256, 512] * [512, 256] -> [256, 256]
    // =========================================================
    optiling::TCubeTiling tilingData;
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    tilingApi.SetCType(TPosition::VECCALC, CubeFormat::ND, DataType::DT_INT32);

    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    
    // MM1 强制切分: 32 x 256
    tilingApi.SetSingleShape(SINGLECORE_M, 256, -1);
    
    int32_t mBlockNum = M / SINGLECORE_M; // 8
    int32_t nBlockNum = N / 256;          // 1
    tilingApi.SetDim(mBlockNum * nBlockNum); // SetDim = 8
    
    tilingApi.SetBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);

    if (tilingApi.GetTiling(tilingData) == -1) {
        std::cout << "Gen Tiling1 failed" << std::endl;
    }

    // =========================================================
    // 3. 生成 Tiling 2 (Syndrome * H -> Votes)
    // Shape: [256, 256] * [256, 512] -> [256, 512]
    // =========================================================
    MultiCoreMatmulTiling api2(*ascendcPlatform);
    optiling::TCubeTiling data2;

    api2.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, false);
    api2.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, true);
    api2.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32);

    // 注意：这里的 OrgShape 是 MM2 的逻辑形状
    api2.SetOrgShape(256, 512, 256);
    api2.SetShape(256, 512, 256);

    // 【核心修复】：移除 if(310B) 判断，无条件强制设置 SingleShape
    // 必须保证 M 维度切分与 MM1 完全一致 (32)，才能对齐 BlockIdx
    api2.SetSingleShape(SINGLECORE_M, 512, -1);
    
    int32_t mBlockNum2 = M / SINGLECORE_M; // 8
    int32_t nBlockNum2 = 512 / 512;        // 1
    api2.SetDim(mBlockNum2 * nBlockNum2);  // SetDim = 8

    api2.SetBufferSpace(-1, -1, -1);

    if (api2.GetTiling(data2) == -1) {
        std::cerr << "Gen Tiling2 failed" << std::endl;
    }

    // =========================================================
    // 4. 保存结果
    // =========================================================
    uint32_t tcubeTilingSize = sizeof(optiling::TCubeTiling);
    
    // 保存 Tiling 1
    tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);
    *reinterpret_cast<uint64_t *>(tilingBuf + tcubeTilingSize) = ubSize;
    // 保存 Tiling 2 (偏移 2048)
    data2.SaveToBuffer(tilingBuf + 2048, tcubeTilingSize);
    *reinterpret_cast<uint64_t *>(tilingBuf + 2048 + tcubeTilingSize) = ubSize;
}