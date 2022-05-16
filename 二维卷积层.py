from re import X
from turtle import forward
from typing import ForwardRef
import torch 
from torch import nn

def corr2d(X, K):#接受输入数组X与核数组K，并输出数组Y。
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))#输出Y的形状
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super(Conv2D,self).__init__()
        self.weight =  nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.rand(1))
    def forward(self,x):
        return corr2d(x,self.weight)+self.bias

# 构造一个核数组形状是(1, 2)的二维卷积层
conv2d = Conv2D(kernel_size=(1, 2))
#构造一张6×8的图像。它中间4列为黑（0），其余为白（1）。
X = torch.ones(6, 8)
X[:, 2:6] = 0
K = torch.tensor([[1, -1]])
#将从白到黑的边缘和从黑到白的边缘分别检测成了1和-1。其余部分的输出全是0。
Y = corr2d(X, K)
#训练模型
num_epochs ,lr = 20 , 0.01
for i in range(num_epochs):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y)**2).sum()
    l.backward()
    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad
    # 梯度清0
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    print('Step %d, loss %.3f' % (i + 1, l.item()))
print("weight: ", conv2d.weight.data)
print("bias: ", conv2d.bias.data)