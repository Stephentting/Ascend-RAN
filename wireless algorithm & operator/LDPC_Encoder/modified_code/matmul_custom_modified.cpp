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

/**
  * @brief  Copy tiling data to TCubeTiling ptr from tiling gm addr.
  * @param  tiling: TCubeTiling ptr which needs to copy tiling data.
  * @param  localMemSize: Temporary local memory size required by matmul calc.
  * @param  tilingGM: tiling gm addr.
  * @retval None
  */
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

/**
  * @brief  Calculate the gm offset and tail size based on the blockidx.
  * @param  blockIdx: Current Core blockidx.
  * @param  tiling: Matmul tiling data.
  * @param  offsetA: Gm offset of A matrix.
  * @param  offsetB: Gm offset of B matrix.
  * @param  offsetC: Gm offset of C matrix.
  * @param  tailM: SingleCoreM size of tail core.
  * @param  tailN: SingleCoreN size of tail core.
  * @param  isTransA: A matrix transpose.
  * @param  isTransB: B matrix transpose.
  * @retval None
  */
__aicore__ inline void CalcGMOffset(int blockIdx, const TCubeTiling &tiling, int &offsetA, int &offsetB, int &offsetC,
                                    int &tailM, int &tailN, bool isTransA, bool isTransB)
{
    uint32_t mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    uint32_t mCoreIndx = blockIdx % mSingleBlocks;
    uint32_t nCoreIndx = blockIdx / mSingleBlocks;

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

/**
  * @brief  matmul kernel function entry
  * @param  a: A matrix gm addr.
  * @param  b: B matrix gm addr.
  * @param  c: C matrix gm addr.
  * @param  workspace: Temporary gm space addr required by matmul calc.
  * @param  tilingGm: Tiling data addr. 
  * @retval None
  */
extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace,
                                                    GM_ADDR tilingGm)
{
    using A_T = int8_t;
    using B_T = int8_t;
    using C_T = int32_t;

    AscendC::TPipe pipe;
    TCubeTiling tiling;
    uint64_t localMemSize = 0;
    CopyTiling(&tiling, localMemSize, tilingGm);

    AscendC::GlobalTensor<A_T> aGlobal;
    AscendC::GlobalTensor<B_T> bGlobal;
    AscendC::GlobalTensor<int16_t> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(b), tiling.Ka * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(c), tiling.M * tiling.N);
    
    AscendC::TBuf<AscendC::TPosition::CO1,2> co1Que;
    pipe.InitBuffer(co1Que, tiling.baseM * tiling.baseN * sizeof(int32_t));
    AscendC::LocalTensor<int32_t> int32src = tmpBuf.Get<int32_t>();

    AscendC::TBuf<AscendC::TPosition::VECOUT> tmpBuf2;
    pipe.InitBuffer(tmpBuf2, tiling.baseM * tiling.baseN * sizeof(int16_t));
    AscendC::LocalTensor<int16_t> int16dst = tmpBuf2.Get<int16_t>();

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    bool isTransA = false;
    bool isTransB = false;

    int tailM = 0;
    int tailN = 0;
    // Calculate the gm offset and tail size based on the blockidx.
    CalcGMOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, tailM, tailN, isTransA, isTransB);

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T>> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling); // Initialize the matmul object.
    if (GetBlockIdx() >= tiling.usedCoreNum) {
        return;
    }
#ifdef CUSTOM_ASCEND310P
    // Set temp UB space when on ASCEND310P.
    AscendC::TBuf<> tmpMMFormatUb;
    AscendC::LocalTensor<uint8_t> mmFormatUb;
    pipe.InitBuffer(tmpMMFormatUb, localMemSize);
    mmFormatUb = tmpMMFormatUb.Get<uint8_t>(localMemSize);
    mm.SetLocalWorkspace(mmFormatUb);
#endif
    mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
    mm.SetTensorA(gmA, isTransA);
    mm.SetTensorB(gmB, isTransB);
    mm.SetTail(tailM, tailN);
    // mm.IterateAll(int32src);
    mm.template Iterate<false>();
    int mLoop = tiling.singleCoreM / baseM;
    int nLoop = tiling.singleCoreN / baseN;
    for (int n = 0; n < nLoop; ++n) {
        for (int m = 0; m < mLoop; ++m) {
            // ----------------------------------------------------------
            // Cube 生产阶段 (Producer)
            // ----------------------------------------------------------
            // 1. 从队列中申请一个空闲的 int32 Tensor (如果队列满了，这里会阻塞等待 Vector 释放)
            AscendC::LocalTensor<int32_t> co1Local = co1Que.AllocTensor<int32_t>();

            // 2. 命令 Cube 将计算结果搬运到这个 Tensor 中
            // <false> 表示异步，发完指令立刻往下走，不等待数据写完
            mm.template GetTensorC<false>(co1Local);

            // 3. 将 Tensor 推入队列。
            // 注意：虽然 GetTensorC 还没写完，但 EnQue 会设置硬件同步标志。
            // Vector 侧的 DeQue 会自动根据这个标志等待，直到 Cube 真的写完数据。
            co1Que.EnQue(co1Local);

            // ----------------------------------------------------------
            // Vector 消费阶段 (Consumer)
            // ----------------------------------------------------------
            // 4. 从队列取出存有数据的 Tensor
            // 硬件会自动阻塞，直到对应的 GetTensorC 完成，确保没有数据竞争 (RAW依赖解决)
            AscendC::LocalTensor<int32_t> int32Src = co1Que.DeQue<int32_t>();

            // 5. 此时数据已就绪，执行 Vector 计算 (Cast)
            AscendC::Cast(int16dst, int32Src, AscendC::RoundMode::CAST_NONE, baseM * baseN);

            // 6. 数据已转存到 int16Tile，原 int32Src 不再需要，释放回队列供 Cube 复用
            // 这步非常关键，释放后 Cube 才能申请到 Tensor 进行下一轮计算
            co1Que.FreeTensor(int32Src);

            // ----------------------------------------------------------
            // 搬运结果到 Global Output Buffer (int16dst)
            // ----------------------------------------------------------
            // 计算在 int16dst 中的偏移并搬运 (这部分逻辑和之前一样)
            uint32_t dstStartRow = m * baseM;
            uint32_t dstStartCol = n * baseN;
            for (int row = 0; row < baseM; ++row) {
                uint32_t dstOffset = (dstStartRow + row) * tiling.singleCoreN + dstStartCol;
                uint32_t srcOffset = row * baseN;
                // 注意：这里使用了 LocalTensor 的切片操作
                AscendC::DataCopy(gm[dstOffset], int16dst[srcOffset], baseN);
            }
        }
    mm.End();
    // AscendC::TQueSync<PIPE_M, PIPE_V> sync;
    // sync.SetFlag(0);
    // sync.WaitFlag(0);
    // AscendC::Cast(int16dst,int32src,AscendC::RoundMode::CAST_NONE,tiling.singleCoreM * tiling.singleCoreN);
    // if(0 == AscendC::GetBlockIdx())
    // {
    //     AscendC::printf("int16dst50个元素为:\n");
    //     AscendC::DumpTensor(int16dst,5,50);
    //     AscendC::printf("int32src50个元素为:\n");
    //     AscendC::DumpTensor(int32src,5,50);
    // }

    // int32_t length = int16dst.GetSize();
    // AscendC::printf("int16dst元素个数为:%d\n",length);
    // int32_t length2 = int32src.GetSize();
    // AscendC::printf("int32src元素个数为:%d\n",length2);
    // AscendC::DataCopy(gmC, int16dst, tiling.singleCoreM * tiling.singleCoreN);
}
