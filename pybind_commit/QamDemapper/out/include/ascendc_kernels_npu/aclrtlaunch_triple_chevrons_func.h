
#ifndef HEADER_ACLRTLAUNCH_QAM_DEMAPPER_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_QAM_DEMAPPER_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_qam_demapper(uint32_t blockDim, void* stream, void* input_I, void* input_Q, void* output);

inline uint32_t qam_demapper(uint32_t blockDim, void* hold, void* stream, void* input_I, void* input_Q, void* output)
{
    (void)hold;
    return aclrtlaunch_qam_demapper(blockDim, stream, input_I, input_Q, output);
}

#endif
