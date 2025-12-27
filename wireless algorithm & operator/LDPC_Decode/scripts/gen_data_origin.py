#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
import os


def gen_golden_data():
    M = 256
    N = 256
    K = 512

    x1_gm = np.random.randint(0, 2, [M, K]).astype(np.uint8)
    # x2_gm = np.random.randint(0, 2, [K, N]).astype(np.uint8)
    x2_gm = np.fromfile("matrix_H_transposed_padded_512x256.bin",dtype = np.uint8).reshape([K,N])
    golden1 = np.matmul(x1_gm, x2_gm).astype(np.int32)
    golden2 = np.mod(golden1, 2)
    # 计算x2_gm的转置
    x2_gm_T = x2_gm.T

    golden = np.matmul(golden2, x2_gm_T).astype(np.int32)
    # golden = golden2.astype(np.int8)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()