from __future__ import print_function
import torch
import numpy as np
x=torch.Tensor(5,3)
x = torch.rand(5,3)

y = torch.rand(5,3)

print(x)
print(y)
y.add_(x)

print(y)


#a = np.ones(5)
a=np.array([1,2,3,4,5])
print(a.shape)
b = torch.from_numpy(a)
print(b.size())

np.add(a,1,out=a)
print(a)
print(b)
