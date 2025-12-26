/**
 * @file matmul_custom.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;


__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

// ... [省略前面的 CopyTiling, ComputeMinWorkSize, CalcGMOffset 函数，保持不变] ...
__aicore__ inline int ComputeMinWorkSize(int typeSize, int dataLength)
{
    int elementPerBlock = 32/typeSize;  //每个Block中的元素个数
    int elementPerRepeat = 256/typeSize; //每次repeat处理的元素个数

    int maxRepeat = dataLength / elementPerRepeat + 1;
    int localSizeNeed = Ceiling(maxRepeat, elementPerBlock)*elementPerBlock;
    return localSizeNeed;
}
__aicore__ inline void CopyTiling(TCubeTiling *tiling, uint64_t &localMemSize, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    localMemSize = *reinterpret_cast<__gm__ uint64_t *>(tilingGM + sizeof(TCubeTiling));
    return;
}

__aicore__ inline void CalcGMOffset(int blockIdx, const TCubeTiling &tiling, int &offsetA, int &offsetB, int &offsetC,
                                    int &tailM, int &tailN, bool isTransA, bool isTransB)
{
    uint32_t mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    uint32_t mCoreIndx = blockIdx % mSingleBlocks;
    uint32_t nCoreIndx = blockIdx / mSingleBlocks;  //0

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    if (isTransA) {
        offsetA = mCoreIndx * tiling.singleCoreM;
    }
    offsetB = nCoreIndx * tiling.singleCoreN;
    if (isTransB) {
        offsetB = nCoreIndx * tiling.Kb * tiling.singleCoreN;
    }
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;

    tailM = tiling.M - mCoreIndx * tiling.singleCoreM;
    tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;

    tailN = tiling.N - nCoreIndx * tiling.singleCoreN;
    tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
}


// 放在 matmul_custom 函数外部或上方

template<typename T>
class LdpcFlipping {
public:
    __aicore__ inline LdpcFlipping() {}
    
    // 【修改点1】：Init函数增加 maskTensor 参数
    __aicore__ inline void Init(GM_ADDR votesGM, GM_ADDR bitsGM, GM_ADDR outputGM, 
                                uint32_t totalRows, uint32_t rowLen, 
                                AscendC::TPipe* pipe) {
        m_votesGM = votesGM;
        m_bitsGM = bitsGM;
        m_outputGM = outputGM;
        m_totalRows = totalRows;
        m_rowLen = rowLen;
        m_pipe = pipe;
        
        m_lenBurstAligned = AscendC::AlignUp(rowLen, 32); 

        // 1. 队列初始化
        m_pipe->InitBuffer(Q_Votes, 2, m_lenBurstAligned * sizeof(int32_t));
        m_pipe->InitBuffer(Q_Bits, 2, m_lenBurstAligned * sizeof(int8_t));
        m_pipe->InitBuffer(Q_Out, 2, m_lenBurstAligned * sizeof(int8_t));
        
        // 2. 计算 Buffer 初始化
        m_pipe->InitBuffer(B_Calc, m_lenBurstAligned * sizeof(float)); 
        m_pipe->InitBuffer(B_Calc_Original, m_lenBurstAligned * sizeof(float));
        m_pipe->InitBuffer(reduceBuffer, m_lenBurstAligned * sizeof(float));
        
        // 3. Mask Buffer
        m_pipe->InitBuffer(selMaskBuffer, 128);
        m_pipe->InitBuffer(B_Scalar, 128);

        bitsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(m_bitsGM) , m_totalRows * m_rowLen);
        votesGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(m_votesGM), m_totalRows * m_rowLen);
    }

    __aicore__ inline void SetMask(AscendC::LocalTensor<int16_t> maskTensor) {
        m_maskTensor = maskTensor;
    }

    __aicore__ inline void Process() {
        for (int i = 0; i < m_totalRows; i++) {
            // 【修改点2】：检查当前行的 Mask 值
            // m_maskTensor 在 UB 上，可以直接通过 GetValue 获取标量值
            int16_t maskVal = m_maskTensor.GetValue(i);
            
            // 如果 mask 为 0，说明不需要操作
            if (maskVal == 0) {
                // 因为是原位修改(In-place)，跳过 Compute 和 CopyOut 意味着保持原数据不变
                // if(AscendC::GetBlockIdx() == 1){
                //     AscendC::printf("Skipping row %d due to mask=0\n", i);
                // }
                continue; 
            }

            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int index) {
        AscendC::LocalTensor<int32_t> votesLocal = Q_Votes.AllocTensor<int32_t>();
        AscendC::LocalTensor<int8_t> bitsLocal = Q_Bits.AllocTensor<int8_t>();
        AscendC::DataCopy(votesLocal, votesGlobal[index * m_rowLen], m_lenBurstAligned);
        AscendC::DataCopy(bitsLocal, bitsGlobal[index * m_rowLen], m_lenBurstAligned);
        Q_Votes.EnQue(votesLocal);
        Q_Bits.EnQue(bitsLocal);
    }

    __aicore__ inline void Compute(int index) {
        AscendC::LocalTensor<int32_t> votesLocal = Q_Votes.DeQue<int32_t>();
        AscendC::LocalTensor<int8_t> bitsLocal = Q_Bits.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> outLocal = Q_Out.AllocTensor<int8_t>();
        
        // 1. 获取 Buffer
        AscendC::LocalTensor<float> votesFloat = B_Calc_Original.Get<float>();
        AscendC::LocalTensor<float> maxValScalar = B_Scalar.Get<float>();
        AscendC::LocalTensor<float> maxValTensor = B_Calc.Get<float>();
        AscendC::LocalTensor<float> tempFloat = reduceBuffer.Get<float>();
        AscendC::LocalTensor<uint8_t> selMask = selMaskBuffer.Get<uint8_t>(); 

        // 2. 转换 Votes 为 Float 并找最大值
        AscendC::Cast(votesFloat, votesLocal, AscendC::RoundMode::CAST_NONE, m_rowLen);
        AscendC::ReduceMax(maxValScalar, votesFloat, tempFloat, m_rowLen, true);
        AscendC::PipeBarrier<PIPE_ALL>(); // 等待 ReduceMax 完成

        float maxVal = maxValScalar.GetValue(0);
        AscendC::Duplicate(maxValTensor, maxVal, m_rowLen);
        // 3. 生成相等掩码 (votes == max)
        // 注意：这里读取 votesFloat (B_Calc_Original)
        AscendC::Compare(selMask, votesFloat, maxValTensor, AscendC::CMPMODE::EQ, m_rowLen);
        
        // 【关键修改 1】：增加 Barrier，防止后续 Duplicate 覆盖 votesFloat 时 Compare 还没读完
        AscendC::PipeBarrier<PIPE_V>();

        // 4. 准备 Select 的输入
        AscendC::LocalTensor<float> oneTensor = B_Calc_Original.Get<float>(); // 复用 B_Calc_Original
        AscendC::LocalTensor<float> zeroTensor = reduceBuffer.Get<float>();   // 复用 reduceBuffer
        AscendC::LocalTensor<float> maskElement = B_Calc.Get<float>();

        AscendC::Duplicate(zeroTensor, (float)0.0, m_rowLen);
        
        // 【关键修改 2】：提高门限。只有当 maxVal >= 2.0 时才允许翻转。
        // 如果 maxVal == 1，翻转极易造成振荡和误码扩散。
        float fillValue = (maxVal > 0.5f) ? 1.0f : 0.0f; 
        
        // 写入 oneTensor (覆盖了 votesFloat，因前面加了 Barrier 所以安全)
        AscendC::Duplicate(oneTensor, fillValue, m_rowLen);

        // 使用 selMask 选择：如果 votes==max 且 max>1，则 mask=1，否则 mask=0
        AscendC::Select(maskElement, selMask, oneTensor, zeroTensor, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, m_rowLen);
        
        // 5. Cast Mask (Float -> Int16)
        // 使用 reduceBuffer 存放 xor_mask (覆盖 zeroTensor)
        AscendC::LocalTensor<int16_t> xor_mask = reduceBuffer.Get<int16_t>(); 
        AscendC::Cast(xor_mask, maskElement, AscendC::RoundMode::CAST_RINT, m_rowLen);
        AscendC::PipeBarrier<PIPE_V>(); // 确保 maskElement 用完
        
        // 6. 准备 Input Bits (Int8 -> Half -> Int16)
        // 使用 B_Calc 存放临时 half (覆盖 maskElement)
        AscendC::LocalTensor<half> tempHalf = B_Calc.Get<half>();
        // 使用 B_Calc_Original 存放 xor_bits (覆盖 oneTensor)
        AscendC::LocalTensor<int16_t> xor_bits = B_Calc_Original.Get<int16_t>();

        AscendC::Cast(tempHalf, bitsLocal, AscendC::RoundMode::CAST_NONE, m_rowLen);
        AscendC::Cast(xor_bits, tempHalf, AscendC::RoundMode::CAST_RINT, m_rowLen);
        AscendC::PipeBarrier<PIPE_V>();

        // 7. XOR 操作
        // 使用 B_Calc 存放结果 xor_out (覆盖 tempHalf)
        AscendC::LocalTensor<int16_t> xor_out = B_Calc.Get<int16_t>();
        AscendC::LocalTensor<uint8_t> xorTmp = selMaskBuffer.Get<uint8_t>(); 
        // Inputs: xor_bits (B_Calc_Original), xor_mask (reduceBuffer) -> Output: xor_out (B_Calc)
        AscendC::Xor(xor_out, xor_bits, xor_mask, xorTmp, m_rowLen);

        AscendC::PipeBarrier<PIPE_V>();

        // 8. 结果写回 (Int16 -> Half -> Int8)
        // 使用 reduceBuffer 存放临时 outHalf (覆盖 xor_mask)
        AscendC::LocalTensor<half> outHalf = reduceBuffer.Get<half>();
        AscendC::Cast(outHalf, xor_out, AscendC::RoundMode::CAST_NONE, m_rowLen);
        AscendC::Cast(outLocal, outHalf, AscendC::RoundMode::CAST_NONE, m_rowLen);
        
        Q_Out.EnQue(outLocal);
        Q_Votes.FreeTensor(votesLocal);
        Q_Bits.FreeTensor(bitsLocal);
    }

    __aicore__ inline void CopyOut(int index) {
        AscendC::LocalTensor<int8_t> outLocal = Q_Out.DeQue<int8_t>();
        AscendC::DataCopy(bitsGlobal[index * m_rowLen], outLocal, m_lenBurstAligned);
        Q_Out.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe* m_pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> Q_Votes;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> Q_Bits;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> Q_Out;
    
    AscendC::TBuf<AscendC::TPosition::VECCALC> B_Calc;
    AscendC::TBuf<AscendC::TPosition::VECCALC> B_Calc_Original;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuffer;
    
    AscendC::TBuf<AscendC::TPosition::VECCALC> selMaskBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> B_Scalar;
    
    // 【修改点3】：增加 m_maskTensor 成员
    AscendC::LocalTensor<int16_t> m_maskTensor;

    AscendC::GlobalTensor<int8_t> bitsGlobal;
    AscendC::GlobalTensor<int32_t> votesGlobal;
    GM_ADDR m_votesGM;
    GM_ADDR m_bitsGM;
    GM_ADDR m_outputGM;
    uint32_t m_totalRows;
    uint32_t m_rowLen;
    uint32_t m_lenBurstAligned;
};

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c1, GM_ADDR mask,
                                                    GM_ADDR c, GM_ADDR workspace,GM_ADDR tilingGm
                                                    )
{
    
    
    using A_T = int8_t;
    using B_T = int8_t;
    using C_T = int32_t;
    // if(AscendC::GetBlockIdx() == 0){
    //     // AscendC::printf("max_iter = %d\n",max_iter);
    //     AscendC::printf("max_iter = \n");
    // }
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    TCubeTiling tiling2;
    uint64_t localMemSize = 0;
    uint64_t localMemSize2 = 0;
    uint64_t sysWsSize = 0;
    uint64_t sysWsSize2 = 0;
    CopyTiling(&tiling, localMemSize, tilingGm);
    GM_ADDR tilingGm2 = tilingGm + 2048;
    CopyTiling(&tiling2, localMemSize2, tilingGm2);
    AscendC::GlobalTensor<A_T> aGlobal;
    AscendC::GlobalTensor<B_T> bGlobal;
    AscendC::GlobalTensor<int32_t> cGlobal;
    AscendC::GlobalTensor<int32_t> maskGlobal;
    AscendC::GlobalTensor<int8_t> c1Global;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(b), tiling.Ka * tiling.N);
    maskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mask) + AscendC::GetBlockIdx() * 8, 8);
    c1Global.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(c1), tiling.M * tiling.N); 
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(c), tiling2.M * tiling2.N);
    
    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    bool isTransA = false;
    bool isTransB = false;
    
    int tailM = 0;
    int tailN = 0;
    CalcGMOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, tailM, tailN, isTransA, isTransB);

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC1 = c1Global[offsetC];

    size_t batch_size = tiling.singleCoreM;
    AscendC::TBuf<AscendC::TPosition::A1> ABuf;
    pipe.InitBuffer(ABuf, tiling2.singleCoreM * tiling2.singleCoreK * sizeof(int8_t));
    
    AscendC::TBuf<AscendC::TPosition::VECCALC> cBuf; 
    pipe.InitBuffer(cBuf, tiling.singleCoreM * tiling.singleCoreN * sizeof(C_T)); 
    AscendC::LocalTensor<C_T> cLocal = cBuf.Get<C_T>(tiling.singleCoreM * tiling.singleCoreN); 
    AscendC::Duplicate<C_T>(cLocal, (int32_t)0, tiling.singleCoreM * tiling.singleCoreN);
    AscendC::TBuf<AscendC::TPosition::VECCALC> oneBuf;
    pipe.InitBuffer(oneBuf, tiling.singleCoreM * tiling.singleCoreN * sizeof(int16_t)); 
    
    AscendC::TBuf<AscendC::TPosition::VECCALC> sBuf;
    pipe.InitBuffer(sBuf, tiling.singleCoreM * tiling.singleCoreN * sizeof(int8_t));    
    AscendC::LocalTensor<int8_t> sLocal = sBuf.Get<int8_t>();
    
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf, scalarBuf, sharedTmpBuf;
    pipe.InitBuffer(maskBuf, batch_size * sizeof(int16_t));
    pipe.InitBuffer(scalarBuf, 64);
    int sharedWorkSpace = ComputeMinWorkSize(sizeof(int16_t),tiling.N);
    pipe.InitBuffer(sharedTmpBuf, tiling.singleCoreN * sizeof(half));
    
    AscendC::TBuf<AscendC::TPosition::VECCALC> int16tmpBuf;
    pipe.InitBuffer(int16tmpBuf, tiling.singleCoreM * tiling.singleCoreN * sizeof(int16_t)); 
    int16_t coreTotalMaskSum = 0; 
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T>,
           MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, C_T>> mm; 
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling); 
    // AscendC::PipeBarrier<PIPE_ALL>();
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int32_t>> mm2;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm2, &tiling2);
    int offsetA2 = 0;
    int offsetB2 = 0;
    int offsetC2 = 0;
    int tailM2 = 0;
    int tailN2 = 0;
    
    CalcGMOffset(GetBlockIdx(), tiling2, offsetA2, offsetB2, offsetC2, tailM2, tailN2, false, true);
    AscendC::GlobalTensor<int8_t> gmA2 = c1Global[offsetA2];
    AscendC::GlobalTensor<int8_t> gmB2 = bGlobal[offsetB2];
    auto gmC = cGlobal[offsetC2];

    auto votesPhy = const_cast<__gm__ int32_t*>(gmC.GetPhyAddr());
    GM_ADDR votesAddr = reinterpret_cast<__gm__ uint8_t*>(votesPhy);
    auto bitsPhy = const_cast<__gm__ int8_t*>(gmA.GetPhyAddr());
    GM_ADDR bitsAddr = reinterpret_cast<__gm__ uint8_t*>(bitsPhy);

    uint32_t rows = tiling2.singleCoreM;
    uint32_t rowLen = tiling2.singleCoreN;
    AscendC::PipeBarrier<PIPE_ALL>();
    LdpcFlipping<int8_t> flipper;
    
    // 【修改点4】：调用 Init 时传入 maskTensor
    flipper.Init(votesAddr, bitsAddr, bitsAddr, rows, rowLen, &pipe);
    int MAX_ITER = 20;
    for(int iter = 0;iter < MAX_ITER; iter++){
        if (GetBlockIdx() >= tiling.usedCoreNum) {
            return;
        }
        
        mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
        mm.SetTensorA(gmA, isTransA);
        mm.SetTensorB(gmB, isTransB);
        mm.SetTail(tailM, tailN);
        mm.IterateAll(cLocal);
        mm.End();

        // AscendC::TQueSync<PIPE_M, PIPE_V> sync2;
        // sync2.SetFlag(1);
        // sync2.WaitFlag(1);
        // AscendC::TQueSync<PIPE_M, PIPE_MTE3> sync;
        // sync.SetFlag(0);
        // sync.WaitFlag(0);
        AscendC::PipeBarrier<PIPE_ALL>(); 
        int dataLength = tiling.singleCoreM * tiling.singleCoreN;
        AscendC::LocalTensor<int16_t> int16dst = int16tmpBuf.Get<int16_t>();
        AscendC::LocalTensor<int16_t> oneTensor = oneBuf.Get<int16_t>();
        AscendC::Duplicate<int16_t>(oneTensor, (int16_t)1, tiling.singleCoreM * tiling.singleCoreN);
    
        AscendC::Cast(int16dst,cLocal,AscendC::RoundMode::CAST_NONE,dataLength);
        
        AscendC::And(int16dst, int16dst, oneTensor, dataLength);
        AscendC::LocalTensor<half> halfDst = oneTensor.ReinterpretCast<half>();
        AscendC::Cast(halfDst, int16dst, AscendC::RoundMode::CAST_NONE, dataLength);
        AscendC::LocalTensor<int8_t> c_int8t = int16dst.ReinterpretCast<int8_t>();
        AscendC::Cast(c_int8t, halfDst, AscendC::RoundMode::CAST_NONE, dataLength);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gmC1, c_int8t, dataLength);
        // AscendC::PipeBarrier<PIPE_ALL>();
        // // 计算 Mask
        AscendC::LocalTensor<int16_t> maskTensor = maskBuf.Get<int16_t>(); 
        AscendC::LocalTensor<half> sharedTmpLocal = sharedTmpBuf.Get<half>();
        AscendC::LocalTensor<half> scalarSum = scalarBuf.Get<half>(2);
        for(int i = 0; i < batch_size; i++)
        {
            AscendC::ReduceSum<half>(scalarSum, halfDst[i*tiling.N], sharedTmpLocal, tiling.N);
            AscendC::PipeBarrier<PIPE_V>(); 
            
            int16_t val = (int16_t)scalarSum.GetValue(0);
            maskTensor.SetValue(i, val); 
            
            // 【新增】在循环里直接累加，比后面再做一次 Vector ReduceSum 更简单且省 Buffer
            coreTotalMaskSum += val;
            
            // AscendC::printf("scalarSum[%d]=%d\n",i,scalarSum.GetValue(0));
        }
        // AscendC::PipeBarrier<PIPE_ALL>();
        // AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::LocalTensor<int32_t> cLocal2 = cLocal;

        mm2.SetOrgShape(tiling2.M, tiling2.N, tiling2.Ka, tiling2.Kb);
        mm2.SetTensorA(gmA2, false);
        mm2.SetTensorB(gmB2, true);
        mm2.SetTail(tailM2, tailN2);
        mm2.IterateAll(gmC);
        mm2.End();
        AscendC::PipeBarrier<PIPE_ALL>();
        flipper.SetMask(maskTensor);

        flipper.Process();
        AscendC::PipeBarrier<PIPE_ALL>(); // 确保之前对 cLocal 的使用结束
    }
    cLocal.SetValue(0, (int32_t)coreTotalMaskSum); // 转为 int32 存入
    
    // 此时 cLocal[0] 存放了 mask sum (int32)
    // 将其拷贝到 maskGlobal
    AscendC::PipeBarrier<PIPE_ALL>(); 
    AscendC::DataCopy(maskGlobal, cLocal, 8); // 拷贝 1 个 int3
}