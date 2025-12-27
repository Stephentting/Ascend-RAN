/**
 * @file main.cpp
 * ZF均衡算子主程序 - Batch=10
 */
#include "data_utils.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include <chrono>
extern void zf_equalization_do(uint32_t blockDim, void *stream,
                              uint8_t *h_real, uint8_t *h_imag,
                              uint8_t *y_real, uint8_t *y_imag,
                              uint8_t *x_hat_real, uint8_t *x_hat_imag);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void zf_equalization(GM_ADDR h_real, GM_ADDR h_imag,
                                                      GM_ADDR y_real, GM_ADDR y_imag,
                                                      GM_ADDR x_hat_real, GM_ADDR x_hat_imag);
#endif

int32_t main(int32_t argc, char *argv[])
{
    constexpr int32_t BATCH_SIZE = 1192;          // 32个batch
    constexpr int32_t NUM_SUBCARRIERS = 256;
    constexpr int32_t TOTAL_ELEMENTS = BATCH_SIZE * NUM_SUBCARRIERS;  // 8192
    
    uint32_t blockDim = 8;  // 使用8个AI Core
    size_t channelByteSize = TOTAL_ELEMENTS * sizeof(uint16_t);  // 16384 bytes

#ifdef ASCENDC_CPU_DEBUG
    // CPU调试模式
    uint8_t *h_real = (uint8_t *)AscendC::GmAlloc(channelByteSize);
    uint8_t *h_imag = (uint8_t *)AscendC::GmAlloc(channelByteSize);
    uint8_t *y_real = (uint8_t *)AscendC::GmAlloc(channelByteSize);
    uint8_t *y_imag = (uint8_t *)AscendC::GmAlloc(channelByteSize);
    uint8_t *x_hat_real = (uint8_t *)AscendC::GmAlloc(channelByteSize);
    uint8_t *x_hat_imag = (uint8_t *)AscendC::GmAlloc(channelByteSize);
    
    ReadFile("./input/input_h_real.bin", channelByteSize, h_real, channelByteSize);
    ReadFile("./input/input_h_imag.bin", channelByteSize, h_imag, channelByteSize);
    ReadFile("./input/input_y_real.bin", channelByteSize, y_real, channelByteSize);
    ReadFile("./input/input_y_imag.bin", channelByteSize, y_imag, channelByteSize);
    
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(zf_equalization, blockDim, h_real, h_imag, y_real, y_imag, 
                x_hat_real, x_hat_imag);
    
    WriteFile("./output/output_x_hat_real.bin", x_hat_real, channelByteSize);
    WriteFile("./output/output_x_hat_imag.bin", x_hat_imag, channelByteSize);
    
    AscendC::GmFree((void *)h_real);
    AscendC::GmFree((void *)h_imag);
    AscendC::GmFree((void *)y_real);
    AscendC::GmFree((void *)y_imag);
    AscendC::GmFree((void *)x_hat_real);
    AscendC::GmFree((void *)x_hat_imag);
#else
    // NPU运行模式
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    
    // Host端内存分配
    uint8_t *hRealHost, *hImagHost, *yRealHost, *yImagHost;
    uint8_t *xHatRealHost, *xHatImagHost;
    
    CHECK_ACL(aclrtMallocHost((void **)(&hRealHost), channelByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&hImagHost), channelByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yRealHost), channelByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yImagHost), channelByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&xHatRealHost), channelByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&xHatImagHost), channelByteSize));
    
    // Device端内存分配
    uint8_t *hRealDevice, *hImagDevice, *yRealDevice, *yImagDevice;
    uint8_t *xHatRealDevice, *xHatImagDevice;
    
    CHECK_ACL(aclrtMalloc((void **)&hRealDevice, channelByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&hImagDevice, channelByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yRealDevice, channelByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yImagDevice, channelByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&xHatRealDevice, channelByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&xHatImagDevice, channelByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    
    // 读取输入数据
    ReadFile("./input/input_h_real.bin", channelByteSize, hRealHost, channelByteSize);
    ReadFile("./input/input_h_imag.bin", channelByteSize, hImagHost, channelByteSize);
    ReadFile("./input/input_y_real.bin", channelByteSize, yRealHost, channelByteSize);
    ReadFile("./input/input_y_imag.bin", channelByteSize, yImagHost, channelByteSize);
    
    // Host到Device拷贝
    CHECK_ACL(aclrtMemcpy(hRealDevice, channelByteSize, hRealHost, channelByteSize, 
                         ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(hImagDevice, channelByteSize, hImagHost, channelByteSize, 
                         ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yRealDevice, channelByteSize, yRealHost, channelByteSize, 
                         ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yImagDevice, channelByteSize, yImagHost, channelByteSize, 
                         ACL_MEMCPY_HOST_TO_DEVICE));
    
    // // 1. 预热 (Warm-up)
    // // 目的：激活 Device，填充指令 Cache，避免首次运行的抖动
    // constexpr int32_t WARMUP_COUNT = 10;
    // printf("Starting warm-up (%d runs)...\n", WARMUP_COUNT);
    // for (int i = 0; i < WARMUP_COUNT; ++i) {
    //     zf_equalization_do(blockDim, stream, hRealDevice, hImagDevice, 
    //                       yRealDevice, yImagDevice, xHatRealDevice, xHatImagDevice);
    // }
    // // 预热结束后同步一次，确保计时开始前设备是空闲的
    // CHECK_ACL(aclrtSynchronizeStream(stream));

    // // 2. 循环计时测试 (Loop Test)
    // constexpr int32_t TEST_LOOP = 100; // 循环次数，建议设大一点，如 100 或 1000
    // printf("Starting performance test (%d runs)...\n", TEST_LOOP);

    // // 开始计时
    // auto start_time = std::chrono::high_resolution_clock::now();

    // for (int i = 0; i < TEST_LOOP; ++i) {
    //     zf_equalization_do(blockDim, stream, hRealDevice, hImagDevice, 
    //                       yRealDevice, yImagDevice, xHatRealDevice, xHatImagDevice);
    // }

    zf_equalization_do(blockDim, stream, hRealDevice, hImagDevice, 
                          yRealDevice, yImagDevice, xHatRealDevice, xHatImagDevice);

    // 关键：必须等待 Stream 中所有任务执行完毕才能停止计时
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // 停止计时
    // auto end_time = std::chrono::high_resolution_clock::now();

    // // 3. 计算结果
    // // 计算总耗时 (毫秒)
    // double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    // // 计算平均耗时
    // double avg_time_ms = total_time_ms / TEST_LOOP;

    // printf("------------------------------------------------\n");
    // printf("[Performance Result]\n");
    // printf("Total Time  : %.4f ms\n", total_time_ms);
    // printf("Loop Count  : %d\n", TEST_LOOP);
    // printf("Avg Time/Op : %.4f ms\n", avg_time_ms);
    // printf("------------------------------------------------\n");
    
    CHECK_ACL(aclrtSynchronizeStream(stream));
    
    // Device到Host拷贝
    CHECK_ACL(aclrtMemcpy(xHatRealHost, channelByteSize, xHatRealDevice, channelByteSize, 
                         ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(xHatImagHost, channelByteSize, xHatImagDevice, channelByteSize, 
                         ACL_MEMCPY_DEVICE_TO_HOST));
    
    // 写出结果
    WriteFile("./output/output_x_hat_real.bin", xHatRealHost, channelByteSize);
    WriteFile("./output/output_x_hat_imag.bin", xHatImagHost, channelByteSize);
    
    // 清理
    CHECK_ACL(aclrtFree(hRealDevice));
    CHECK_ACL(aclrtFree(hImagDevice));
    CHECK_ACL(aclrtFree(yRealDevice));
    CHECK_ACL(aclrtFree(yImagDevice));
    CHECK_ACL(aclrtFree(xHatRealDevice));
    CHECK_ACL(aclrtFree(xHatImagDevice));
    
    CHECK_ACL(aclrtFreeHost(hRealHost));
    CHECK_ACL(aclrtFreeHost(hImagHost));
    CHECK_ACL(aclrtFreeHost(yRealHost));
    CHECK_ACL(aclrtFreeHost(yImagHost));
    CHECK_ACL(aclrtFreeHost(xHatRealHost));
    CHECK_ACL(aclrtFreeHost(xHatImagHost));
    
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}