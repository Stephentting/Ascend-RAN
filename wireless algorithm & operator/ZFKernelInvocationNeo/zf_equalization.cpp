#include "kernel_operator.h"

// 基础常量定义
constexpr int32_t NUM_SUBCARRIERS = 256;
constexpr int32_t BATCH_SIZE = 1192; 
constexpr int32_t USE_CORE_NUM = 8;

// 计算得出的常量
constexpr int32_t BATCH_PER_CORE = BATCH_SIZE / USE_CORE_NUM;        // 149
constexpr int32_t BLOCK_LENGTH = BATCH_PER_CORE * NUM_SUBCARRIERS;  // 38144
constexpr int32_t TILE_LENGTH = NUM_SUBCARRIERS;                    // 256 (1个Batch)
constexpr int32_t BUFFER_NUM = 2;                                   // 双缓冲

class KernelZFEqualization {
public:
    __aicore__ inline KernelZFEqualization() {}
    
    __aicore__ inline void Init(GM_ADDR h_real, GM_ADDR h_imag, 
                               GM_ADDR y_real, GM_ADDR y_imag,
                               GM_ADDR x_hat_real, GM_ADDR x_hat_imag,
                               AscendC::TPipe *pipe)
    {
        // 计算每个核的起始偏移
        int32_t blockOffset = BLOCK_LENGTH * AscendC::GetBlockIdx();
        
        // 全局内存绑定
        hRealGm.SetGlobalBuffer((__gm__ half *)h_real + blockOffset, BLOCK_LENGTH);
        hImagGm.SetGlobalBuffer((__gm__ half *)h_imag + blockOffset, BLOCK_LENGTH);
        yRealGm.SetGlobalBuffer((__gm__ half *)y_real + blockOffset, BLOCK_LENGTH);
        yImagGm.SetGlobalBuffer((__gm__ half *)y_imag + blockOffset, BLOCK_LENGTH);
        xHatRealGm.SetGlobalBuffer((__gm__ half *)x_hat_real + blockOffset, BLOCK_LENGTH);
        xHatImagGm.SetGlobalBuffer((__gm__ half *)x_hat_imag + blockOffset, BLOCK_LENGTH);
        
        // 队列初始化：每个 Buffer 大小严格等于 1 个 Batch
        pipe->InitBuffer(inQueueHReal, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe->InitBuffer(inQueueHImag, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe->InitBuffer(inQueueYReal, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe->InitBuffer(inQueueYImag, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        
        pipe->InitBuffer(outQueueXHatReal, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe->InitBuffer(outQueueXHatImag, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        
        // 临时缓冲区：存放计算中间变量 (h2, temp1, temp2, recip)
        pipe->InitBuffer(tempBuf, TILE_LENGTH * sizeof(half) * 4);
        
        this->pipe = pipe;
    }
    
    __aicore__ inline void Process() {
        // 总循环次数改为每个核分到的 Batch 数
        int32_t loopCount = BATCH_PER_CORE; 
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> hRealLocal = inQueueHReal.AllocTensor<half>();
        AscendC::LocalTensor<half> hImagLocal = inQueueHImag.AllocTensor<half>();
        AscendC::LocalTensor<half> yRealLocal = inQueueYReal.AllocTensor<half>();
        AscendC::LocalTensor<half> yImagLocal = inQueueYImag.AllocTensor<half>();
        
        // 每次偏移 1 个 TILE_LENGTH (256)
        AscendC::DataCopy(hRealLocal, hRealGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(hImagLocal, hImagGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(yRealLocal, yRealGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(yImagLocal, yImagGm[progress * TILE_LENGTH], TILE_LENGTH);
        
        inQueueHReal.EnQue(hRealLocal);
        inQueueHImag.EnQue(hImagLocal);
        inQueueYReal.EnQue(yRealLocal);
        inQueueYImag.EnQue(yImagLocal);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> hRealLocal = inQueueHReal.DeQue<half>();
        AscendC::LocalTensor<half> hImagLocal = inQueueHImag.DeQue<half>();
        AscendC::LocalTensor<half> yRealLocal = inQueueYReal.DeQue<half>();
        AscendC::LocalTensor<half> yImagLocal = inQueueYImag.DeQue<half>();
        
        AscendC::LocalTensor<half> xHatRealLocal = outQueueXHatReal.AllocTensor<half>();
        AscendC::LocalTensor<half> xHatImagLocal = outQueueXHatImag.AllocTensor<half>();
        
        AscendC::LocalTensor<half> temp = tempBuf.Get<half>();
        AscendC::LocalTensor<half> h2 = temp[0];
        AscendC::LocalTensor<half> temp1 = temp[TILE_LENGTH];
        AscendC::LocalTensor<half> temp2 = temp[TILE_LENGTH * 2];
        AscendC::LocalTensor<half> recip = temp[TILE_LENGTH * 3];
        
        // ZF均衡向量化计算 (核心逻辑保持不变，确保计算长度为 TILE_LENGTH=256)
        AscendC::Mul(h2, hRealLocal, hRealLocal, TILE_LENGTH);
        AscendC::Mul(temp1, hImagLocal, hImagLocal, TILE_LENGTH);
        AscendC::Add(h2, h2, temp1, TILE_LENGTH);
        AscendC::Adds(h2, h2, (half)1e-6, TILE_LENGTH);
        AscendC::Reciprocal(recip, h2, TILE_LENGTH);
        
        // 实部计算
        AscendC::Mul(temp1, hRealLocal, yRealLocal, TILE_LENGTH);
        AscendC::Mul(temp2, hImagLocal, yImagLocal, TILE_LENGTH);
        AscendC::Add(temp1, temp1, temp2, TILE_LENGTH);
        AscendC::Mul(xHatRealLocal, temp1, recip, TILE_LENGTH);
        
        // 虚部计算
        AscendC::Mul(temp1, hRealLocal, yImagLocal, TILE_LENGTH);
        AscendC::Mul(temp2, hImagLocal, yRealLocal, TILE_LENGTH);
        AscendC::Muls(temp2, temp2, (half)(-1.0), TILE_LENGTH);
        AscendC::Add(temp1, temp1, temp2, TILE_LENGTH);
        AscendC::Mul(xHatImagLocal, temp1, recip, TILE_LENGTH);
        
        outQueueXHatReal.EnQue(xHatRealLocal);
        outQueueXHatImag.EnQue(xHatImagLocal);
        
        inQueueHReal.FreeTensor(hRealLocal);
        inQueueHImag.FreeTensor(hImagLocal);
        inQueueYReal.FreeTensor(yRealLocal);
        inQueueYImag.FreeTensor(yImagLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> xHatRealLocal = outQueueXHatReal.DeQue<half>();
        AscendC::LocalTensor<half> xHatImagLocal = outQueueXHatImag.DeQue<half>();
        
        AscendC::DataCopy(xHatRealGm[progress * TILE_LENGTH], xHatRealLocal, TILE_LENGTH);
        AscendC::DataCopy(xHatImagGm[progress * TILE_LENGTH], xHatImagLocal, TILE_LENGTH);
        
        outQueueXHatReal.FreeTensor(xHatRealLocal);
        outQueueXHatImag.FreeTensor(xHatImagLocal);
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueHReal, inQueueHImag;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueYReal, inQueueYImag;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueXHatReal, outQueueXHatImag;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf;
    AscendC::GlobalTensor<half> hRealGm, hImagGm, yRealGm, yImagGm;
    AscendC::GlobalTensor<half> xHatRealGm, xHatImagGm;
};

extern "C" __global__ __aicore__ void zf_equalization(GM_ADDR h_real, GM_ADDR h_imag,
                                                      GM_ADDR y_real, GM_ADDR y_imag,
                                                      GM_ADDR x_hat_real, GM_ADDR x_hat_imag)
{
    AscendC::TPipe pipe;
    KernelZFEqualization op;
    op.Init(h_real, h_imag, y_real, y_imag, x_hat_real, x_hat_imag, &pipe);
    op.Process();
}