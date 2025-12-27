#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t TOTAL_ELEMENTS = 1192*220;
constexpr int32_t NUM_LEVELS = 8;
constexpr int32_t BITS_PER_SYMBOL = 6;
constexpr int32_t USE_CORE_NUM = 8;
constexpr int32_t BLOCK_LENGTH = TOTAL_ELEMENTS / USE_CORE_NUM;  // 32780

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_NUM = 8;
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / (TILE_NUM * BUFFER_NUM);  // 2048
constexpr int32_t TOTAL_PROGRESS = TILE_NUM * BUFFER_NUM;  // 16
constexpr int32_t REMAINDER = BLOCK_LENGTH - TOTAL_PROGRESS * TILE_LENGTH;  // 12


class KernelQamDemapper {
public:
    __aicore__ inline KernelQamDemapper() {}
    
    __aicore__ inline void Init(GM_ADDR input_I, GM_ADDR input_Q, GM_ADDR output, AscendC::TPipe* pipe)
    {
        this->blockLength = BLOCK_LENGTH;
        this->tileNum = TILE_NUM;
        this->tileLength = TILE_LENGTH;
        this->remainder = REMAINDER;
        this->pipe = pipe;
        
        // Bind global memory
        inputIGm.SetGlobalBuffer((__gm__ float*)input_I + this->blockLength * GetBlockIdx(), this->blockLength);
        inputQGm.SetGlobalBuffer((__gm__ float*)input_Q + this->blockLength * GetBlockIdx(), this->blockLength);
        outputGm.SetGlobalBuffer((__gm__ uint8_t*)output + BITS_PER_SYMBOL * this->blockLength * GetBlockIdx(), 
                                 BITS_PER_SYMBOL * this->blockLength);

        // Initialize input/output queues
        pipe->InitBuffer(inQueueI, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe->InitBuffer(inQueueQ, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe->InitBuffer(outQueue, BUFFER_NUM, BITS_PER_SYMBOL * this->tileLength * sizeof(uint8_t));
        
        // Initialize temporary buffers
        pipe->InitBuffer(temp, this->tileLength * sizeof(float));
        pipe->InitBuffer(level, this->tileLength * sizeof(float));
        pipe->InitBuffer(tempI, NUM_LEVELS * this->tileLength * sizeof(float));
        pipe->InitBuffer(tempQ, NUM_LEVELS * this->tileLength * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        int32_t loopCount = TOTAL_PROGRESS;
        
        // 处理所有完整的 tiles
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        
        // 每个 core 都需要处理剩余的 12 个符号
        if (this->remainder > 0) {
            ProcessRemainder();
        }
    }

private:
    // 常规 DataCopy
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> inputILocal = inQueueI.AllocTensor<float>();
        AscendC::LocalTensor<float> inputQLocal = inQueueQ.AllocTensor<float>();
        
        AscendC::DataCopy(inputILocal, inputIGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(inputQLocal, inputQGm[progress * this->tileLength], this->tileLength);
        
        inQueueI.EnQue(inputILocal);
        inQueueQ.EnQue(inputQLocal);
    }
    
    // 常规计算
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> inputILocal = inQueueI.DeQue<float>();
        AscendC::LocalTensor<float> inputQLocal = inQueueQ.DeQue<float>();
        AscendC::LocalTensor<uint8_t> outputLocal = outQueue.AllocTensor<uint8_t>();
        
        AscendC::LocalTensor<float> levelLocal = level.Get<float>();
        AscendC::LocalTensor<float> tempLocal = temp.Get<float>();
        AscendC::LocalTensor<float> tempILocal = tempI.Get<float>();
        AscendC::LocalTensor<float> tempQLocal = tempQ.Get<float>();

        for (int32_t i = 0; i < NUM_LEVELS; i++) {
            float level_value = (float)(-7 + 2 * i) / 6.4807406984f;
            AscendC::Duplicate(levelLocal, level_value, this->tileLength);
            AscendC::Sub(tempLocal, inputILocal, levelLocal, this->tileLength);
            AscendC::Abs(tempILocal[i * this->tileLength], tempLocal, this->tileLength);
            AscendC::Sub(tempLocal, inputQLocal, levelLocal, this->tileLength);
            AscendC::Abs(tempQLocal[i * this->tileLength], tempLocal, this->tileLength);
        }

        for (int32_t sym = 0; sym < this->tileLength; sym++) {
            float min_dist_I = tempILocal.GetValue(sym);
            uint8_t min_idx_I = 0;
            for (int32_t lev = 1; lev < NUM_LEVELS; lev++) {
                float dist = tempILocal.GetValue(lev * this->tileLength + sym);
                if (dist < min_dist_I) {
                    min_dist_I = dist;
                    min_idx_I = lev;
                }
            }
            
            float min_dist_Q = tempQLocal.GetValue(sym);
            uint8_t min_idx_Q = 0;
            for (int32_t lev = 1; lev < NUM_LEVELS; lev++) {
                float dist = tempQLocal.GetValue(lev * this->tileLength + sym);
                if (dist < min_dist_Q) {
                    min_dist_Q = dist;
                    min_idx_Q = lev;
                }
            }
            
            outputLocal.SetValue(sym * BITS_PER_SYMBOL + 0, (min_idx_I >> 2) & 1);
            outputLocal.SetValue(sym * BITS_PER_SYMBOL + 1, (min_idx_I >> 1) & 1);
            outputLocal.SetValue(sym * BITS_PER_SYMBOL + 2, min_idx_I & 1);
            outputLocal.SetValue(sym * BITS_PER_SYMBOL + 3, (min_idx_Q >> 2) & 1);
            outputLocal.SetValue(sym * BITS_PER_SYMBOL + 4, (min_idx_Q >> 1) & 1);
            outputLocal.SetValue(sym * BITS_PER_SYMBOL + 5, min_idx_Q & 1);
        }
        
        inQueueI.FreeTensor(inputILocal);
        inQueueQ.FreeTensor(inputQLocal);
        outQueue.EnQue(outputLocal);
    }
    
    // 常规输出
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<uint8_t> outputLocal = outQueue.DeQue<uint8_t>();
        AscendC::DataCopy(outputGm[progress * BITS_PER_SYMBOL * this->tileLength], 
                         outputLocal, BITS_PER_SYMBOL * this->tileLength);
        outQueue.FreeTensor(outputLocal);
    }
    
    // 处理剩余的 12 个符号（每个 core 都调用）
    __aicore__ inline void ProcessRemainder()
    {
        int32_t offset = TOTAL_PROGRESS * this->tileLength;  // 32768
        
        // 直接从 Global Memory 读取并计算，逐个符号处理
        for (int32_t sym = 0; sym < this->remainder; sym++) {
            float I_val = inputIGm.GetValue(offset + sym);
            float Q_val = inputQGm.GetValue(offset + sym);
            
            // 计算 I 分量的距离并找最小
            float min_dist_I = 1e9;
            uint8_t min_idx_I = 0;
            for (int32_t lev = 0; lev < NUM_LEVELS; lev++) {
                float level_value = (float)(-7 + 2 * lev) / 6.4807406984f;
                float dist = I_val - level_value;
                if (dist < 0) dist = -dist;
                if (dist < min_dist_I) {
                    min_dist_I = dist;
                    min_idx_I = lev;
                }
            }
            
            // 计算 Q 分量的距离并找最小
            float min_dist_Q = 1e9;
            uint8_t min_idx_Q = 0;
            for (int32_t lev = 0; lev < NUM_LEVELS; lev++) {
                float level_value = (float)(-7 + 2 * lev) / 6.4807406984f;
                float dist = Q_val - level_value;
                if (dist < 0) dist = -dist;
                if (dist < min_dist_Q) {
                    min_dist_Q = dist;
                    min_idx_Q = lev;
                }
            }
            
            // 直接写入 Global Memory
            int32_t bit_offset = (offset + sym) * BITS_PER_SYMBOL;
            outputGm.SetValue(bit_offset + 0, (min_idx_I >> 2) & 1);
            outputGm.SetValue(bit_offset + 1, (min_idx_I >> 1) & 1);
            outputGm.SetValue(bit_offset + 2, min_idx_I & 1);
            outputGm.SetValue(bit_offset + 3, (min_idx_Q >> 2) & 1);
            outputGm.SetValue(bit_offset + 4, (min_idx_Q >> 1) & 1);
            outputGm.SetValue(bit_offset + 5, min_idx_Q & 1);
        }
    }

private:
    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueI, inQueueQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    
    AscendC::GlobalTensor<float> inputIGm, inputQGm;
    AscendC::GlobalTensor<uint8_t> outputGm;
    
    AscendC::TBuf<AscendC::TPosition::VECIN> temp;
    AscendC::TBuf<AscendC::TPosition::VECIN> level;
    AscendC::TBuf<AscendC::TPosition::VECIN> tempI;
    AscendC::TBuf<AscendC::TPosition::VECIN> tempQ;
    
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t remainder;
};

extern "C" __global__ __aicore__ void qam_demapper(GM_ADDR input_I, GM_ADDR input_Q, GM_ADDR output) {
    AscendC::TPipe pipe;
    KernelQamDemapper op;
    
    op.Init(input_I, input_Q, output, &pipe);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void qam_demapper_do(uint32_t blockDim, void* stream,
                     float* input_I, float* input_Q,
                     uint8_t* output) {
    qam_demapper<<<blockDim, nullptr, stream>>>(input_I, input_Q, output);
}
#endif
