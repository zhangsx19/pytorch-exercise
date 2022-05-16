import torch
from torch import nn 
from torch.nn import init
import numpy as np
import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l
#构建网络
num_inputs, num_outputs, num_hiddens = 784, 10, 256
net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs,num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens,num_outputs)
)
#初始化参数
for params in net.parameters():
    init.normal_(params,mean=0,std=0.01)
#定义损失函数
loss = nn.CrossEntropyLoss()#分开定义softmax运算和交叉熵损失函数可能会造成数值不稳定，pytorch提供了包括softmax运算和交叉熵损失计算的函数
#定义优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
#读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#训练模型
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)