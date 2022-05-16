import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l


#读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#构建网络
num_inputs = 784
num_outputs = 10

net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs,num_outputs)
)#按前后顺序写
#初始化参数
init.normal_(net[1].weight, mean=0, std=0.01)
init.constant_(net[1].bias, val=0) 
#定义交叉熵损失函数
loss = nn.CrossEntropyLoss()#分开定义softmax运算和交叉熵损失函数可能会造成数值不稳定，pytorch提供了包括softmax运算和交叉熵损失计算的函数
#定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
#训练模型
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
