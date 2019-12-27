#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

## 将整数序列编码为二进制矩阵
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results
