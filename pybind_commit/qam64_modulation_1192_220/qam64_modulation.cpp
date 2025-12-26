/**
 * @file qam64_modulation_vectorized.cpp
 * QAM64调制算子 - 标准 Gray 码映射修正版
 */

#include "kernel_operator.h"

// ==================== 常量定义 ====================
constexpr int32_t BITS_PER_SYMBOL = 6;
constexpr int32_t NUM_SYMBOLS_PER_BATCH = 220;
constexpr int32_t BATCH_SIZE = 1192;
constexpr int32_t TOTAL_SYMBOLS = BATCH_SIZE * NUM_SYMBOLS_PER_BATCH;

constexpr int32_t USE_CORE_NUM = 8;
constexpr int32_t BLOCK_LENGTH = TOTAL_SYMBOLS / USE_CORE_NUM;  // 每个核处理的符号数
constexpr int32_t TILE_NUM = 16;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_LENGTH = 1024; // 每次搬运的符号数
constexpr int32_t REMAINDER = BLOCK_LENGTH % (TILE_NUM * BUFFER_NUM * TILE_LENGTH); 

// 归一化因子: 1/sqrt(42)
constexpr float NORMALIZATION_FACTOR = 0.15430335f;

// 向量化参数
constexpr int32_t VECTOR_LENGTH = 64;
constexpr int32_t NUM_VECTORS_PER_TILE = TILE_LENGTH / VECTOR_LENGTH;

class KernelQAM64Modulation {
public:
    __aicore__ inline KernelQAM64Modulation() {}
    
    __aicore__ inline void Init(GM_ADDR input_bits, GM_ADDR output_real, 
                               GM_ADDR output_imag, AscendC::TPipe *pipe)
    {
        // 确保 Global Memory 偏移正确
        uint32_t blockIdx = AscendC::GetBlockIdx();
        inputBitsGm.SetGlobalBuffer((__gm__ uint8_t *)input_bits + BLOCK_LENGTH * BITS_PER_SYMBOL * blockIdx, 
                                    BLOCK_LENGTH * BITS_PER_SYMBOL);
        outputRealGm.SetGlobalBuffer((__gm__ half *)output_real + BLOCK_LENGTH * blockIdx, BLOCK_LENGTH);
        outputImagGm.SetGlobalBuffer((__gm__ half *)output_imag + BLOCK_LENGTH * blockIdx, BLOCK_LENGTH);
        
        pipe->InitBuffer(inQueueBits, BUFFER_NUM, TILE_LENGTH * BITS_PER_SYMBOL * sizeof(uint8_t));
        pipe->InitBuffer(outQueueReal, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe->InitBuffer(outQueueImag, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe->InitBuffer(tempBuf, TILE_LENGTH * sizeof(half) * 2);
        pipe->InitBuffer(grayLutBuf, 8 * sizeof(half));
        
        this->pipe = pipe;
        
        InitGrayLookupTable();
    }
    
    __aicore__ inline void Process() {
        int32_t loopCount = TILE_NUM * BUFFER_NUM;
        
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            ComputeVectorized(i);
            CopyOut(i);
        }
        
        // 处理尾部剩余数据
        if (REMAINDER > 0) {
            ProcessRemainder();
        }
    }

private:
    // 标准 Gray 码映射查找表
    // 映射逻辑 (Binary -> Level): 
    // 0(000):-7, 1(001):-5, 2(010):-1, 3(011):-3, 4(100):+7, 5(101):+5, 6(110):+1, 7(111):+3
    __aicore__ inline void InitGrayLookupTable() {
        AscendC::LocalTensor<half> grayLut = grayLutBuf.Get<half>(8);
        const float lut[8] = {
            -7.0f * NORMALIZATION_FACTOR, // 000
            -5.0f * NORMALIZATION_FACTOR, // 001
            -1.0f * NORMALIZATION_FACTOR, // 010
            -3.0f * NORMALIZATION_FACTOR, // 011
             7.0f * NORMALIZATION_FACTOR, // 100
             5.0f * NORMALIZATION_FACTOR, // 101
             1.0f * NORMALIZATION_FACTOR, // 110
             3.0f * NORMALIZATION_FACTOR  // 111
        };
        for (int i = 0; i < 8; ++i) {
            grayLut.SetValue(i, static_cast<half>(lut[i]));
        }
    }
    
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<uint8_t> bitsLocal = inQueueBits.AllocTensor<uint8_t>();
        AscendC::DataCopy(bitsLocal, inputBitsGm[progress * TILE_LENGTH * BITS_PER_SYMBOL], 
                          TILE_LENGTH * BITS_PER_SYMBOL);
        inQueueBits.EnQue(bitsLocal);
    }
    
    __aicore__ inline void ComputeVectorized(int32_t progress) {
        AscendC::LocalTensor<uint8_t> bitsLocal = inQueueBits.DeQue<uint8_t>();
        AscendC::LocalTensor<half> outputRealLocal = outQueueReal.AllocTensor<half>();
        AscendC::LocalTensor<half> outputImagLocal = outQueueImag.AllocTensor<half>();
        AscendC::LocalTensor<half> grayLut = grayLutBuf.Get<half>(8);
        
        AscendC::LocalTensor<half> temp = tempBuf.Get<half>(TILE_LENGTH * 2);
        AscendC::LocalTensor<half> iLevels = temp;
        AscendC::LocalTensor<half> qLevels = temp[TILE_LENGTH];
        
        for (int32_t vec_idx = 0; vec_idx < NUM_VECTORS_PER_TILE; ++vec_idx) {
            int32_t symbol_start = vec_idx * VECTOR_LENGTH;
            ProcessSymbolVector(bitsLocal, iLevels, qLevels, grayLut, symbol_start, VECTOR_LENGTH);
        }
        
        AscendC::DataCopy(outputRealLocal, iLevels, TILE_LENGTH);
        AscendC::DataCopy(outputImagLocal, qLevels, TILE_LENGTH);
        
        outQueueReal.EnQue(outputRealLocal);
        outQueueImag.EnQue(outputImagLocal);
        inQueueBits.FreeTensor(bitsLocal);
    }
    
    __aicore__ inline void ProcessSymbolVector(AscendC::LocalTensor<uint8_t>& bitsLocal,
                                             AscendC::LocalTensor<half>& iLevels,
                                             AscendC::LocalTensor<half>& qLevels,
                                             AscendC::LocalTensor<half>& grayLut,
                                             int32_t start_symbol, int32_t num_symbols) {
        for (int32_t i = 0; i < num_symbols; ++i) {
            int32_t symbol_idx = start_symbol + i;
            int32_t bitStart = symbol_idx * BITS_PER_SYMBOL;
            
            // 计算 3-bit 索引 (MSB First)
            uint8_t iIndex = (bitsLocal.GetValue(bitStart + 0) << 2) | 
                             (bitsLocal.GetValue(bitStart + 1) << 1) | 
                              bitsLocal.GetValue(bitStart + 2);
            uint8_t qIndex = (bitsLocal.GetValue(bitStart + 3) << 2) | 
                             (bitsLocal.GetValue(bitStart + 4) << 1) | 
                              bitsLocal.GetValue(bitStart + 5);
            
            iLevels.SetValue(symbol_idx, grayLut.GetValue(iIndex));
            qLevels.SetValue(symbol_idx, grayLut.GetValue(qIndex));
        }
    }
    
    __aicore__ inline void ProcessRemainder() {
        uint32_t offset = TILE_NUM * BUFFER_NUM * TILE_LENGTH;
        AscendC::LocalTensor<half> grayLut = grayLutBuf.Get<half>(8);
        
        for (uint32_t sym = 0; sym < REMAINDER; sym++) {
            uint32_t bitStart = (offset + sym) * BITS_PER_SYMBOL;
            
            uint8_t iIdx = (inputBitsGm.GetValue(bitStart + 0) << 2) | 
                           (inputBitsGm.GetValue(bitStart + 1) << 1) | 
                            inputBitsGm.GetValue(bitStart + 2);
            uint8_t qIdx = (inputBitsGm.GetValue(bitStart + 3) << 2) | 
                           (inputBitsGm.GetValue(bitStart + 4) << 1) | 
                            inputBitsGm.GetValue(bitStart + 5);
            
            outputRealGm.SetValue(offset + sym, grayLut.GetValue(iIdx));
            outputImagGm.SetValue(offset + sym, grayLut.GetValue(qIdx));
        }
    }
    
    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<half> outputRealLocal = outQueueReal.DeQue<half>();
        AscendC::LocalTensor<half> outputImagLocal = outQueueImag.DeQue<half>();
        AscendC::DataCopy(outputRealGm[progress * TILE_LENGTH], outputRealLocal, TILE_LENGTH);
        AscendC::DataCopy(outputImagGm[progress * TILE_LENGTH], outputImagLocal, TILE_LENGTH);
        outQueueReal.FreeTensor(outputRealLocal);
        outQueueImag.FreeTensor(outputImagLocal);
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueBits;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueReal, outQueueImag;
    AscendC::TBuf<> tempBuf;
    AscendC::TBuf<> grayLutBuf;
    AscendC::GlobalTensor<uint8_t> inputBitsGm;
    AscendC::GlobalTensor<half> outputRealGm, outputImagGm;
};

extern "C" __global__ __aicore__ void qam64_modulation(GM_ADDR input_bits, 
                                                     GM_ADDR output_real, 
                                                     GM_ADDR output_imag)
{
    AscendC::TPipe pipe;
    KernelQAM64Modulation op;
    op.Init(input_bits, output_real, output_imag, &pipe);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void qam64_modulation_do(uint32_t block_dim, void *stream, 
                        uint8_t *input_bits, 
                        uint8_t *output_real, 
                        uint8_t *output_imag)
{
    qam64_modulation<<<block_dim, nullptr, stream>>>(input_bits, output_real, output_imag);
}
#endif