/**
 * @file matmul_custom.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

// 向上取整辅助函数
__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

// 将 Tiling 数据从 Global Memory 拷贝到 Local
__aicore__ inline void CopyTiling(TCubeTiling *tiling, uint64_t &localMemSize, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    localMemSize = *reinterpret_cast<__gm__ uint64_t *>(tilingGM + sizeof(TCubeTiling));
}

// 计算当前核 (Core) 在 A、B、C 矩阵中的偏移量以及尾块大小
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
 * @brief LDPC 编码自定义算子核心入口
 * 逻辑：C = (A * B) mod 2
 */
extern "C" __global__ __aicore__ void LDPCEnc_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm)
{
    using A_T = int8_t;
    using B_T = int8_t;
    using C_T = int32_t;

    // 1. 初始化与 Tiling 数据获取
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    uint64_t localMemSize = 0;
    CopyTiling(&tiling, localMemSize, tilingGm);

    // 2. Global Tensor 设置
    AscendC::GlobalTensor<A_T> aGlobal;
    AscendC::GlobalTensor<B_T> bGlobal;
    AscendC::GlobalTensor<int16_t> cGlobal; 

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(b), tiling.Ka * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t *>(c), tiling.M * tiling.N);

    // 3. 申请 Local Buffer
    // 存放 Matmul 结果 (int32)
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    pipe.InitBuffer(tmpBuf, tiling.singleCoreM * tiling.singleCoreN * sizeof(int32_t));
    AscendC::LocalTensor<int32_t> int32src = tmpBuf.Get<int32_t>();

    // 存放 Cast 后结果 (int16)
    AscendC::TBuf<AscendC::TPosition::VECOUT> tmpBuf2;
    pipe.InitBuffer(tmpBuf2, tiling.singleCoreM * tiling.singleCoreN * sizeof(int16_t));
    AscendC::LocalTensor<int16_t> int16dst = tmpBuf2.Get<int16_t>();

    // 存放最终模2运算结果
    AscendC::TBuf<AscendC::TPosition::VECOUT> tmpBuf3;
    pipe.InitBuffer(tmpBuf3, tiling.singleCoreM * tiling.singleCoreN * sizeof(int16_t));
    AscendC::LocalTensor<int16_t> zLocal = tmpBuf3.Get<int16_t>();

    // 辅助向量：全1矩阵，用于模拟 mod 2 运算
    AscendC::TBuf<AscendC::TPosition::VECCALC> oneBuf;
    pipe.InitBuffer(oneBuf, tiling.singleCoreM * tiling.singleCoreN * sizeof(int16_t));
    AscendC::LocalTensor<int16_t> yLocal = oneBuf.Get<int16_t>();
    AscendC::Duplicate(yLocal, (int16_t)1, tiling.singleCoreM * tiling.singleCoreN);

    // 4. 计算当前核的偏移量
    int offsetA = 0, offsetB = 0, offsetC = 0;
    int tailM = 0, tailN = 0;
    bool isTransA = false, isTransB = false;

    if (GetBlockIdx() >= tiling.usedCoreNum) return;

    CalcGMOffset(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, tailM, tailN, isTransA, isTransB);

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];

    // 5. 执行矩阵乘法 (Matmul)
    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T>,
           MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, C_T>> mm;

    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

#ifdef CUSTOM_ASCEND310P
    // 310P 需要显式设置 UB 空间 无需在意
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

    mm.IterateAll(int32src);
    mm.End();

    // 6. 后处理：模拟模2运算 (GF(2))
    AscendC::PipeBarrier<PIPE_ALL>();

    uint32_t dataLength = tiling.singleCoreM * tiling.singleCoreN;

    // 类型转换: int32 -> int16
    AscendC::Cast(int16dst, int32src, AscendC::RoundMode::CAST_NONE, dataLength);

    // 模2运算: 结果与 1 进行位与操作 (x & 1 等价于 x % 2)
    AscendC::And(zLocal, int16dst, yLocal, dataLength);

    AscendC::PipeBarrier<PIPE_ALL>();

    // 拷贝结果回 Global Memory
    AscendC::DataCopy(gmC, zLocal, dataLength);
}