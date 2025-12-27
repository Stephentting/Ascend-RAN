/**
 * @file matmul_custom.cpp
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

__aicore__ inline void CopyTiling(TCubeTiling *tiling, uint64_t &localMemSize, GM_ADDR tilingGM) {
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    localMemSize = *reinterpret_cast<__gm__ uint64_t *>(tilingGM + sizeof(TCubeTiling));
}

__aicore__ inline void CalcGMOffsetBatch(int blockIdx, const TCubeTiling &tiling, 
                                        int &offsetA, int &offsetB, int &offsetC, 
                                        int &tailM, int &tailN) {
    uint32_t mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    uint32_t mCoreIndx = blockIdx % mSingleBlocks;
    uint32_t nCoreIndx = blockIdx / mSingleBlocks;
    
    // 使用 tiling.Ka (即 32) 计算偏移
    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM; 
    offsetB = nCoreIndx * tiling.singleCoreN;                
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    
    tailM = tiling.M - mCoreIndx * tiling.singleCoreM;
    tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
    tailN = tiling.N - nCoreIndx * tiling.singleCoreN;
    tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
}

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm) {
    using A_T = half;
    using B_T = half;
    using C_T = float;
    
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    uint64_t localMemSize = 0;
    CopyTiling(&tiling, localMemSize, tilingGm);
    
    if (GetBlockIdx() >= tiling.usedCoreNum) return;
    
    AscendC::GlobalTensor<A_T> aGlobal;
    AscendC::GlobalTensor<B_T> bGlobal;
    AscendC::GlobalTensor<C_T> cGlobal;
    
    // ★ 关键：根据 tiling.M 动态设置 Global Buffer 大小
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(c), tiling.M * tiling.N);
    
    int offsetA = 0, offsetB = 0, offsetC = 0, tailM = 0, tailN = 0;
    CalcGMOffsetBatch(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, tailM, tailN);

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];
    
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T>> mm;
           
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

#ifdef CUSTOM_ASCEND310P
    AscendC::TBuf<> tmpMMFormatUb;
    pipe.InitBuffer(tmpMMFormatUb, localMemSize);
    mm.SetLocalWorkspace(tmpMMFormatUb.Get<uint8_t>(localMemSize));
#endif
    
    mm.SetOrgShape(tailM, tailN, tiling.Ka); // 移除多余参数
    mm.SetTensorA(gmA, false);
    mm.SetTensorB(gmB, false);
    mm.SetTail(tailM, tailN);
    
    mm.IterateAll(gmC);
    mm.End();
}