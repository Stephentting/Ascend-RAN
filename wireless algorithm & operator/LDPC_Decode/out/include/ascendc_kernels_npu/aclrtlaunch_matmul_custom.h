#ifndef HEADER_ACLRTLAUNCH_MATMUL_CUSTOM_H
#define HEADER_ACLRTLAUNCH_MATMUL_CUSTOM_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_matmul_custom(uint32_t blockDim, aclrtStream stream, void* a, void* b, void* c1, void* mask, void* c, void* workspace, void* tilingGm);
#endif
