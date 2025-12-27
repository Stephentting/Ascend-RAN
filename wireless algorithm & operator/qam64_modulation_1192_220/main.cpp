/**
 * @file main.cpp
 * QAM64调制算子主程序
 */

#include "data_utils.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void qam64_modulation_do(uint32_t block_dim, void *stream,
                               uint8_t *input_bits, 
                               uint8_t *output_real, 
                               uint8_t *output_imag);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void qam64_modulation(GM_ADDR input_bits, 
                                                       GM_ADDR output_real, 
                                                       GM_ADDR output_imag);
#endif

int main() {
    constexpr int32_t BATCH_SIZE = 1192;
    constexpr int32_t SYMBOLS_PER_BATCH = 220;
    constexpr int32_t BITS_PER_SYMBOL = 6;
    
    constexpr int32_t TOTAL_SYMBOLS = BATCH_SIZE * SYMBOLS_PER_BATCH;
    constexpr int32_t TOTAL_BITS = TOTAL_SYMBOLS * BITS_PER_SYMBOL;
    
    constexpr uint32_t BLOCK_DIM = 8;  // 使用8个AI Core
    
    size_t bits_size = TOTAL_BITS * sizeof(uint8_t);
    size_t symbols_size = TOTAL_SYMBOLS * sizeof(uint16_t);  // half精度

#ifdef ASCENDC_CPU_DEBUG
    // CPU调试模式
    uint8_t *input_bits = (uint8_t *)AscendC::GmAlloc(bits_size);
    uint8_t *output_real = (uint8_t *)AscendC::GmAlloc(symbols_size);
    uint8_t *output_imag = (uint8_t *)AscendC::GmAlloc(symbols_size);
    
    // 读取输入数据
    ReadFile("./input/input_bits.bin", bits_size, input_bits, bits_size);
    
    // 执行算子
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(qam64_modulation, BLOCK_DIM, input_bits, output_real, output_imag);
    
    // 保存输出
    WriteFile("./output/output_symbols_real.bin", output_real, symbols_size);
    WriteFile("./output/output_symbols_imag.bin", output_imag, symbols_size);
    
    // 释放内存
    AscendC::GmFree((void *)input_bits);
    AscendC::GmFree((void *)output_real);
    AscendC::GmFree((void *)output_imag);
#else
    // NPU运行模式
    CHECK_ACL(aclInit(nullptr));
    int device_id = 0;
    CHECK_ACL(aclrtSetDevice(device_id));
    
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    
    // Host内存分配
    uint8_t *input_bits_host, *output_real_host, *output_imag_host;
    CHECK_ACL(aclrtMallocHost((void **)&input_bits_host, bits_size));
    CHECK_ACL(aclrtMallocHost((void **)&output_real_host, symbols_size));
    CHECK_ACL(aclrtMallocHost((void **)&output_imag_host, symbols_size));
    
    // Device内存分配
    uint8_t *input_bits_device, *output_real_device, *output_imag_device;
    CHECK_ACL(aclrtMalloc((void **)&input_bits_device, bits_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&output_real_device, symbols_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&output_imag_device, symbols_size, ACL_MEM_MALLOC_HUGE_FIRST));
    
    // 读取输入数据
    ReadFile("./input/input_bits.bin", bits_size, input_bits_host, bits_size);
    
    // 拷贝到Device
    CHECK_ACL(aclrtMemcpy(input_bits_device, bits_size, input_bits_host, bits_size, 
                         ACL_MEMCPY_HOST_TO_DEVICE));
    
    // 执行算子
    qam64_modulation_do(BLOCK_DIM, stream, input_bits_device, output_real_device, output_imag_device);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    
    // 拷贝回Host
    CHECK_ACL(aclrtMemcpy(output_real_host, symbols_size, output_real_device, symbols_size, 
                         ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(output_imag_host, symbols_size, output_imag_device, symbols_size, 
                         ACL_MEMCPY_DEVICE_TO_HOST));
    
    // 保存输出
    WriteFile("./output/output_symbols_real.bin", output_real_host, symbols_size);
    WriteFile("./output/output_symbols_imag.bin", output_imag_host, symbols_size);
    
    // 清理资源
    CHECK_ACL(aclrtFree(input_bits_device));
    CHECK_ACL(aclrtFree(output_real_device));
    CHECK_ACL(aclrtFree(output_imag_device));
    CHECK_ACL(aclrtFreeHost(input_bits_host));
    CHECK_ACL(aclrtFreeHost(output_real_host));
    CHECK_ACL(aclrtFreeHost(output_imag_host));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(device_id));
    CHECK_ACL(aclFinalize());
#endif

    return 0;
}