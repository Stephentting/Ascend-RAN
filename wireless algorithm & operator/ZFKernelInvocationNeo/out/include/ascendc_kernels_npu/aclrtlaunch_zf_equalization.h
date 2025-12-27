#ifndef HEADER_ACLRTLAUNCH_ZF_EQUALIZATION_H
#define HEADER_ACLRTLAUNCH_ZF_EQUALIZATION_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_zf_equalization(uint32_t blockDim, aclrtStream stream, void* h_real, void* h_imag, void* y_real, void* y_imag, void* x_hat_real, void* x_hat_imag);
#endif
