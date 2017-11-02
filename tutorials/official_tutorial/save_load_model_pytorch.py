#coding=utf-8
import torch  
from torch.autograd import Variable  
import matplotlib.pyplot as plt  
  
torch.manual_seed(1) # 设定随机数种子  
  
# 创建数据  
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  
y = x.pow(2) + 0.2*torch.rand(x.size())  
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)  
  
# 将待保存的神经网络定义在一个函数中  
def save():  
    # 神经网络结构  
    net1 = torch.nn.Sequential(  
        torch.nn.Linear(1, 10),  
        torch.nn.ReLU(),  
        torch.nn.Linear(10, 1),  
        )  
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)  
    loss_function = torch.nn.MSELoss()  
  
    # 训练部分  
    for i in range(300):  
        prediction = net1(x)  
        loss = loss_function(prediction, y)  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
  
    # 绘图部分  
    plt.figure(1, figsize=(10, 3))  
    plt.subplot(131)  
    plt.title('net1')  
    plt.scatter(x.data.numpy(), y.data.numpy())  
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)  
  
    # 保存神经网络  
    torch.save(net1, '7-net.pkl')                     # 保存整个神经网络的结构和模型参数  
    torch.save(net1.state_dict(), '7-net_params.pkl') # 只保存神经网络的模型参数  
  
# 载入整个神经网络的结构及其模型参数  
def reload_net():  
    net2 = torch.load('7-net.pkl')  
    prediction = net2(x)  
  
    plt.subplot(132)  
    plt.title('net2')  
    plt.scatter(x.data.numpy(), y.data.numpy())  
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)  
  
# 只载入神经网络的模型参数，神经网络的结构需要与保存的神经网络相同的结构  
def reload_params():  
    # 首先搭建相同的神经网络结构  
    net3 = torch.nn.Sequential(  
        torch.nn.Linear(1, 10),  
        torch.nn.ReLU(),  
        torch.nn.Linear(10, 1),  
        )  
  
    # 载入神经网络的模型参数  
    net3.load_state_dict(torch.load('7-net_params.pkl'))  
    prediction = net3(x)  
  
    plt.subplot(133)  
    plt.title('net3')  
    plt.scatter(x.data.numpy(), y.data.numpy())  
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)  
  
# 运行测试  
save()  
reload_net()  
reload_params()
