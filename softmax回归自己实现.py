import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from d2lzh_pytorch import *

def softmax(X): #矩阵X的行数是样本数，列数是输出个数。softmax运算的输出矩阵中的任意一行元素代表了一个样本在各个输出类别上的预测概率。
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)#dim=1为同一行求和，dim=0为同一列
    return X_exp / partition  # 这里应用了广播机制
def net(X):#定义神经网络模型
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
def cross_entropy(y_hat, y):#定义交叉熵（小批量样本）返回n*1 tensor
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))
def accuracy(y_hat, y):#定义准确度，
    #y_hat.argmax(dim=1)返回矩阵y_hat每行中最大元素的索引，(y_hat.argmax(dim=1) == y)是一个类型为ByteTensor的Tensor，float()将其转换为0或1浮点型Tensor
    return (y_hat.argmax(dim=1) == y).float().mean().item()


#读取数据
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)# 如出现“out of memory”的报错信息，可减小batch_size或resize
#初始化参数
num_inputs = 784#28*28个像素点，通道为1
num_outputs = 10#10个标签
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float,requires_grad=True)
b = torch.zeros(num_outputs, dtype=torch.float,requires_grad=True)
#训练模型
lr = 0.03#单位样本学习率
num_epochs = 5
for epoch in range(num_epochs):
    for X,y in train_iter:
        y_hat = net(X)
        loss_entropy = cross_entropy(y_hat,y).mean()
        loss_entropy.backward()
        sgd([W,b],lr,batch_size)
        W.grad.data.zero_()
        b.grad.data.zero_()
print(evaluate_accuracy(test_iter,net))


