#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import sys
import numpy as np

# for float32
relative_tol = 1e-6
absolute_tol = 1e-9
error_tol = 1e-4


def verify_result(output, golden):
    output = np.fromfile(output, dtype=np.int16).reshape(-1)
    golden = np.fromfile(golden, dtype=np.int16).reshape(-1)
    print("output.bin的总长度为: %d\n" %len(output))
    print("golden.bin的总长度为: %d\n" %len(golden))
    different_element_results = np.isclose(output,
                                           golden,
                                           rtol=relative_tol,
                                           atol=absolute_tol,
                                           equal_nan=True)
    different_element_indexes = np.where(different_element_results == False)[0]
    print("错误的总长度为:%d \n" %len(different_element_indexes))
    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
            (real_index, golden_data, output_data,
             abs(output_data - golden_data) / golden_data))
        if index == 100:
            break
    error_ratio = float(different_element_indexes.size) / golden.size
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol

def read_bin_file(filename):
    try:
        data = np.fromfile(filename, dtype=np.uint16)
        count = 10
        print(f"===={filename}====")
        for i in range(count):
            print(f" {i+1}:{data[i]}")
        return 0

    except FileNotFoundError:
        print(f"错误 文件{filename}不存在")
        return None
    
if __name__ == '__main__':
    try:
        read_bin_file(sys.argv[1])
        read_bin_file(sys.argv[2])
        res = verify_result(sys.argv[1], sys.argv[2])
        if not res:
            raise ValueError("[ERROR] result error")
        else:
            print("test pass")
    except Exception as e:
        print(e)
        sys.exit(1)
