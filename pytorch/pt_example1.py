"""
author: zhangbz
project: python-test
date: 2022/11/2
description:
"""
import numpy as np
import torch

# 基本方法
# a = torch.randn(3, 4, requires_grad=True)
# print(a)
#
# b = torch.randn(3, 4, requires_grad=True)
# t = a + b
# y = t.sum()
# print(y)
#
# y.backward()
# print(b.grad)

# 计算流程
x = torch.rand(1)
b = torch.rand(1, requires_grad=True)
w = torch.rand(1, requires_grad=True)
y = w * x
z = y + b

z.backward(retain_graph=True)  # 如果梯度不清零，会累加
print(b.grad)

# 线性回归尝试
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

