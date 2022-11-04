import torch
import numpy as np

print(torch.__version__)

# 基本使用
m1 = torch.empty(5, 3)
print(m1)

m2 = torch.rand(5, 3)
print(m2)

m3 = torch.zeros(5, 3, dtype=torch.int)
print(m3)
print(m3.size())

print(m2[:, 1])
print(m2[2, :])

mm1 = torch.randn(4, 4)
mm2 = mm1.view(16)
mm3 = mm1.view(-1, 8)
print(mm2.size())
print(mm3.size())

# 与numpy转换
a = torch.ones(5)
b = a.numpy()
print(b)

c = np.ones(4)
d = torch.from_numpy(c)
print(d)
