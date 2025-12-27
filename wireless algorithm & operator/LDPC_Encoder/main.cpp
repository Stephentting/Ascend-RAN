/**
 * @file main.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h" 
#include "tiling/platform/platform_ascendc.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_LDPCEnc_custom.h"
#else
#include "tikicpulib.h"
extern "C" void matmul_custom(uint8_t *a, uint8_t *b, uint8_t *c, uint8_t *workspace, uint8_t *tiling);
extern "C" void mod2_custom(GM_ADDR x, GM_ADDR z, Mod2CustomTilingData tiling_gm);
#endif
extern void GenerateTiling(const char *socVersion, uint8_t *tilingBuf);

int32_t main(int32_t argc, char *argv[])
{   
    // const char *socVersion;
    // if (strcmp(SOC_VERSION, "ASCEND310B1") == 0)
    //     socVersion = "ASCEND310B";
    // else
    const uint32_t M = 256;
    const uint32_t N = 512;
    const uint32_t K = 256;
    const char *socVersion = SOC_VERSION;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    size_t aFileSize = M * K * sizeof(int8_t);  // uint16_t represent half
    size_t bFileSize = K * N * sizeof(int8_t); // uint16_t represent half
    size_t cFileSize = M * N * sizeof(int16_t);
    size_t tilingFileSize = sizeof(TCubeTiling) + sizeof(uint64_t);
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    uint8_t *tilingBuf = (uint8_t *)malloc(tilingFileSize);
    GenerateTiling(socVersion, tilingBuf);


#ifdef CUSTOM_ASCEND310P
    uint32_t blockDim = reinterpret_cast<TCubeTiling *>(tilingBuf)->usedCoreNum;
#else
    uint32_t blockDim = reinterpret_cast<TCubeTiling *>(tilingBuf)->usedCoreNum;
#endif

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *a = (uint8_t *)AscendC::GmAlloc(aFileSize);
    uint8_t *b = (uint8_t *)AscendC::GmAlloc(bFileSize);
    uint8_t *c = (uint8_t *)AscendC::GmAlloc(cFileSize);
    // uint8_t *EncodeRes = (uint8_t *)AscendC::GmAlloc(cFileSize); //模2的结果长度和模2前一样
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingFileSize);


    ReadFile("./input/x1_gm.bin", aFileSize, a, aFileSize);
    ReadFile("./input/x2_gm.bin", bFileSize, b, bFileSize);
    memcpy_s(tiling, tilingFileSize, tilingBuf, tilingFileSize);

    ICPU_RUN_KF(matmul_custom, blockDim, a, b, c, workspace, tiling);
    // ICPU_RUN_KF(mod2_custom, blockDim, c, EncodeRes, *mod2Tiling);

    WriteFile("./output/output.bin", c, cFileSize);

    AscendC::GmFree((void *)a);
    AscendC::GmFree((void *)b);
    AscendC::GmFree((void *)c);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *aHost;
    uint8_t *aDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&aHost), aFileSize));
    CHECK_ACL(aclrtMalloc((void **)&aDevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x1_gm.bin", aFileSize, aHost, aFileSize);
    CHECK_ACL(aclrtMemcpy(aDevice, aFileSize, aHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bHost;
    uint8_t *bDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&bHost), bFileSize));
    CHECK_ACL(aclrtMalloc((void **)&bDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x2_gm.bin", bFileSize, bHost, bFileSize);
    CHECK_ACL(aclrtMemcpy(bDevice, bFileSize, bHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *tilingHost;
    uint8_t *tilingDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingFileSize));
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, tilingBuf, tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *cHost;
    uint8_t *cDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&cHost), cFileSize));
    CHECK_ACL(aclrtMalloc((void **)&cDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACLRT_LAUNCH_KERNEL(LDPCEnc_custom)
    (blockDim, stream, aDevice, bDevice, cDevice, workspaceDevice, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(cHost, cFileSize, cDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    //调用mod2算子
    // uint8_t blockDimMod2 = 8;
    // uint8_t *mod2TilingDevice;
    // CHECK_ACL(aclrtMalloc((void **)&mod2TilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // CHECK_ACL(aclrtMemcpy(mod2TilingDevice, mod2TilingSize, mod2TilingBuf, mod2TilingSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // uint8_t *encodeResDevice;
    // CHECK_ACL(aclrtMalloc((void **)&encodeResDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // ACLRT_LAUNCH_KERNEL(mod2_custom)(blockDimMod2, stream, cDevice, encodeResDevice, (Mod2CustomTilingData *)mod2TilingDevice);
    
    // // 4. 等待完成
    // CHECK_ACL(aclrtSynchronizeStream(stream));
    // CHECK_ACL(aclrtMemcpy(cHost, cFileSize, encodeResDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", cHost, cFileSize);
    
    // CHECK_ACL(aclrtFree(mod2TilingDevice));
    // CHECK_ACL(aclrtFree(encodeResDevice));
    CHECK_ACL(aclrtFree(aDevice));
    CHECK_ACL(aclrtFreeHost(aHost));
    CHECK_ACL(aclrtFree(bDevice));
    CHECK_ACL(aclrtFreeHost(bHost));
    CHECK_ACL(aclrtFree(workspaceDevice));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(tilingHost));
    CHECK_ACL(aclrtFree(cDevice));
    CHECK_ACL(aclrtFreeHost(cHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    free(tilingBuf);
    return 0;
}