
#ifndef HEADER_ACLRTLAUNCH_ZF_EQUALIZATION_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_ZF_EQUALIZATION_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_zf_equalization(uint32_t blockDim, void* stream, void* h_real, void* h_imag, void* y_real, void* y_imag, void* x_hat_real, void* x_hat_imag);

inline uint32_t zf_equalization(uint32_t blockDim, void* hold, void* stream, void* h_real, void* h_imag, void* y_real, void* y_imag, void* x_hat_real, void* x_hat_imag)
{
    (void)hold;
    return aclrtlaunch_zf_equalization(blockDim, stream, h_real, h_imag, y_real, y_imag, x_hat_real, x_hat_imag);
}

#endif
