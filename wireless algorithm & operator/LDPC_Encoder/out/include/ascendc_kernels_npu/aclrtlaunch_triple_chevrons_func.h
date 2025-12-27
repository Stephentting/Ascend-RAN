
#ifndef HEADER_ACLRTLAUNCH_LDPCENC_CUSTOM_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_LDPCENC_CUSTOM_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_LDPCEnc_custom(uint32_t blockDim, void* stream, void* a, void* b, void* c, void* workspace, void* tilingGm);

inline uint32_t LDPCEnc_custom(uint32_t blockDim, void* hold, void* stream, void* a, void* b, void* c, void* workspace, void* tilingGm)
{
    (void)hold;
    return aclrtlaunch_LDPCEnc_custom(blockDim, stream, a, b, c, workspace, tilingGm);
}

#endif
