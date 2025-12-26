cmake_minimum_required(VERSION 3.16.0)
project(Ascend_C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 基础配置
set(SOC_VERSION "Ascend310P3" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "ASCEND CANN package installation directory")
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)

# CANN 路径检查
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist.")
endif()

set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# 2. 编译算子内核 (Kernel)
ascendc_library(kernels STATIC
    LDPCEnc_custom.cpp
)

ascendc_compile_definitions(kernels PRIVATE
    HAVE_WORKSPACE
    HAVE_TILING
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
)

# 3. 获取 PyTorch 和 Pybind11 路径 (关键点)
execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_PATH)
execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_NPU_PATH)
execute_process(COMMAND python3 -m pybind11 --includes
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYBIND11_INC)
execute_process(COMMAND python3-config --extension-suffix
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYBIND11_SUFFIX)

# 将 pybind11 的返回结果转为列表，防止包含路径时出现空格导致找不到
string(REPLACE " " ";" PYBIND11_INC_LIST "${PYBIND11_INC}")
# 修正：有些 pybind11 返回的结果带 "-I" 前缀，需要去掉才能用于 target_include_directories
string(REPLACE "-I" "" PYBIND11_INC_LIST "${PYBIND11_INC_LIST}")

# 4. 编译 Pybind11 共享库
add_library(pybind11_lib SHARED
    pybind11.cpp
    LDPCEnc_custom_tiling.cpp
)

# 必须在 target_include_directories 之前定义，确保头文件搜索顺序
target_include_directories(pybind11_lib PRIVATE
    ${PYBIND11_INC_LIST}
    ${TORCH_PATH}/include
    ${TORCH_PATH}/include/torch/csrc/api/include
    ${TORCH_NPU_PATH}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include/kernels  # 关键：包含自动生成的头文件路径
)

target_compile_definitions(pybind11_lib PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
)

target_link_directories(pybind11_lib PRIVATE
    ${TORCH_PATH}/lib
    ${TORCH_NPU_PATH}/lib
)

target_link_libraries(pybind11_lib PRIVATE
    kernels
    torch_npu
    ascendcl
    tiling_api
    register
    platform
    ascendalog
    dl
)

target_compile_options(pybind11_lib PRIVATE
    -D_GLIBCXX_USE_CXX11_ABI=1
    -fPIC
)

set_target_properties(pybind11_lib PROPERTIES
    OUTPUT_NAME ldpc_encode_custom${PYBIND11_SUFFIX}  
    PREFIX "" SUFFIX ""
)