#ifndef HEADER_ACLRTLAUNCH_QAM_DEMAPPER_H
#define HEADER_ACLRTLAUNCH_QAM_DEMAPPER_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_qam_demapper(uint32_t blockDim, aclrtStream stream, void* input_I, void* input_Q, void* output);
#endif
