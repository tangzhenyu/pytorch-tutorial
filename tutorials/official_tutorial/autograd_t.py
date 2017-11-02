from torch.autograd import Variable
import torch
x = Variable(torch.ones(2, 2), requires_grad = True)
y = x + 2
print(y.creator)
z = y * y * 3
print(z.size())
#z.backward()
#gradients = torch.FloatTensor([[0.1, 0.1] [0.2 , 0.2]])
#gradients = torch.ones(2,2) * 2
gradients = torch.Tensor(2)
gradients_z = torch.ones(2,2) * 2
out = z.mean()
#out.backward(retain_variables=True)
#print(x.grad)
#print(y.grad)
#print(z.grad)
#out.backward(gradients)
z.backward(gradients_z)
print(x.grad)
print(y.grad)
print(z.grad)
