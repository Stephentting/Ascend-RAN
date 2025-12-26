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

/**
  * @brief  Generate matmul tiling.
  * @param  socVersion: Platform socversion.
  * @param  tilingBuf data buffer.
  */
 
extern "C" void GenerateTiling(const char *socVersion, uint8_t *tilingBuf)
{
    constexpr int32_t M = 256;
    constexpr int32_t N = 512;
    constexpr int32_t K = 256;
    
    TPosition leftPosition = TPosition::GM;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_INT8;
    bool isTransA = false;

    TPosition rightPosition = TPosition::GM;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_INT8;
    bool isTransB = false;

    TPosition resultPosition = TPosition::GM;
    CubeFormat resultFormat = CubeFormat::ND;
    DataType resultDtype = DataType::DT_INT32;

    bool isBias = false;

    constexpr int32_t SINGLECORE_M = 32;
    constexpr int32_t SINGLECORE_N = 512;

    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetDim(ascendcPlatform->GetCoreNumAiv()); // Set the number of cores that participate in multi-core computaion is 8.
    std::cout <<  "CoreNum_in_tiling:" << ascendcPlatform->GetCoreNumAiv() << endl;
    tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    tilingApi.SetCType(resultPosition, resultFormat, resultDtype);
    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    if (ascendcPlatform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND310B) {
        tilingApi.SetSingleShape(SINGLECORE_M, SINGLECORE_N, -1);  // Set the fixed singleCoreM=512, singleCoreN=512.
        int32_t mBlockNum = M / SINGLECORE_M;
        int32_t nBlockNum = N / SINGLECORE_N;
        tilingApi.SetDim(mBlockNum * nBlockNum);
    }
    tilingApi.SetBias(isBias);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData); // Get matmul tiling data.
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
    }
    uint32_t tcubeTilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);

    uint64_t localMemSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    *reinterpret_cast<uint64_t *>(tilingBuf + tcubeTilingSize) = localMemSize;
    return;
}
