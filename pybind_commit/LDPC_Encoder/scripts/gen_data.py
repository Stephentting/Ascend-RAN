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
    N = 512
    K = 256

    x1_gm = np.random.randint(0, 2, [M, K]).astype(np.int16) #随机生成的比特流
    x2_gm = np.fromfile("./matrix_G_padded_256x512.bin", dtype = np.int8).reshape(K,N).astype(np.int16) #读入LDPC矩阵
    golden = np.matmul(x1_gm,x2_gm).astype(np.int16)
    golden = np.mod(golden,2)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    print(x1_gm.shape)
    x1_gm = x1_gm.astype(np.int8)
    x2_gm = x2_gm.astype(np.int8)
    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
