/**
 * @file main.cpp
 * 64-QAM Hard Demodulation - Binary mapping (NO Gray code)
 */
#include "data_utils.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern "C" void qam_demapper_do(uint32_t blockDim, void *stream,
                              float *input_I, float *input_Q,
                              uint8_t *output);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void qam_demapper(GM_ADDR input_I, GM_ADDR input_Q,
                                                      GM_ADDR output);
#endif

int32_t main(int32_t argc, char *argv[])
{
    constexpr int32_t TOTAL_ELEMENTS = 1192*220;  
    constexpr int32_t BITS_PER_SYMBOL = 6;   
    
    uint32_t blockDim = 8;
    size_t inputByteSize = TOTAL_ELEMENTS * sizeof(float); 
    size_t outputByteSize = TOTAL_ELEMENTS * BITS_PER_SYMBOL * sizeof(uint8_t);

#ifdef ASCENDC_CPU_DEBUG
    // CPU调试模式
    std::cout << "Running on CPU (Binary mapping)..." << std::endl;
    float *input_I = (float *)AscendC::GmAlloc(inputByteSize);
    float *input_Q = (float *)AscendC::GmAlloc(inputByteSize);
    uint8_t *output = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    
    if (!input_I || !input_Q || !output) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return -1;
    }
    
    memset(input_I, 0, inputByteSize);
    memset(input_Q, 0, inputByteSize);
    memset(output, 0, outputByteSize);
    
    std::cout << "Memory allocated: Input=" << inputByteSize 
              << " bytes, Output=" << outputByteSize << " bytes" << std::endl;
    
    // 读取输入
    if (!ReadFile("./input/input_input_I.bin", inputByteSize, input_I, inputByteSize)) {
        std::cerr << "Failed to read input_I!" << std::endl;
        return -1;
    }
    if (!ReadFile("./input/input_input_Q.bin", inputByteSize, input_Q, inputByteSize)) {
        std::cerr << "Failed to read input_Q!" << std::endl;
        return -1;
    }
    
    std::cout << "Input files loaded" << std::endl;
    
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    std::cout << "Running kernel (Binary mapping, NO Gray code)..." << std::endl;
    
    // 运行kernel
    ICPU_RUN_KF(qam_demapper, blockDim, 
                (GM_ADDR)input_I, 
                (GM_ADDR)input_Q, 
                (GM_ADDR)output);
    
    std::cout << "Kernel completed" << std::endl;
    
    WriteFile("./output/output.bin", output, outputByteSize);
    
    std::cout << "Output saved: " << outputByteSize << " bytes" << std::endl;
    
    AscendC::GmFree((void *)input_I);
    AscendC::GmFree((void *)input_Q);
    AscendC::GmFree((void *)output);

#else
    // NPU模式
    std::cout << "Running on NPU (Binary mapping)..." << std::endl;
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    
    // Host内存
    float *input_IHost, *input_QHost;
    uint8_t *outputHost;
    
    CHECK_ACL(aclrtMallocHost((void **)(&input_IHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&input_QHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&outputHost), outputByteSize));
    
    // Device内存
    float *input_IDevice, *input_QDevice;
    uint8_t *outputDevice;
    
    CHECK_ACL(aclrtMalloc((void **)&input_IDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&input_QDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&outputDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    
    // 读取输入
    ReadFile("./input/input_input_I.bin", inputByteSize, input_IHost, inputByteSize);
    ReadFile("./input/input_input_Q.bin", inputByteSize, input_QHost, inputByteSize);

    // Host->Device
    CHECK_ACL(aclrtMemcpy(input_IDevice, inputByteSize, input_IHost, inputByteSize, 
                         ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(input_QDevice, inputByteSize, input_QHost, inputByteSize, 
                         ACL_MEMCPY_HOST_TO_DEVICE));
    
    // 运行
    qam_demapper_do(blockDim, stream, input_IDevice, input_QDevice, outputDevice);
    
    CHECK_ACL(aclrtSynchronizeStream(stream));
    
    // Device->Host
    CHECK_ACL(aclrtMemcpy(outputHost, outputByteSize, outputDevice, outputByteSize, 
                         ACL_MEMCPY_DEVICE_TO_HOST));

    std::cout << "Kernel completed" << std::endl;
    WriteFile("./output/output.bin", outputHost, outputByteSize);
    std::cout << "Output saved" << std::endl;
    
    // 清理
    CHECK_ACL(aclrtFree(input_IDevice));
    CHECK_ACL(aclrtFree(input_QDevice));
    CHECK_ACL(aclrtFree(outputDevice));
    CHECK_ACL(aclrtFreeHost(input_IHost));
    CHECK_ACL(aclrtFreeHost(input_QHost));
    CHECK_ACL(aclrtFreeHost(outputHost));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
