
#ifndef HEADER_ACLRTLAUNCH_MATMUL_CUSTOM_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_MATMUL_CUSTOM_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_matmul_custom(uint32_t blockDim, void* stream, void* a, void* b, void* c1, void* mask, void* c, void* workspace, void* tilingGm);

inline uint32_t matmul_custom(uint32_t blockDim, void* hold, void* stream, void* a, void* b, void* c1, void* mask, void* c, void* workspace, void* tilingGm)
{
    (void)hold;
    return aclrtlaunch_matmul_custom(blockDim, stream, a, b, c1, mask, c, workspace, tilingGm);
}

#endif
