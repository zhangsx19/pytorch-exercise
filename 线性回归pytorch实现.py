from pickletools import optimize
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.nn import init
#预先构筑一个线性回归模型y=wx+b+noise来验证
num_inputs = 2
num_outputs = 1
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)#加噪声
print('第一个输入和输出是：')
print(features[0],labels[0].item())#只有一个元素的张量才能转为元素
#定义模型
net = nn.Sequential(
    nn.Linear(num_inputs,num_outputs)
)#会初始化参数
#初始化参数
init.normal_(net[0].weight,mean=0,std=0.01)#正态分布
init.constant_(net[0].bias, val=0)#常数
#定义损失函数
loss = nn.MSELoss()
#定义优化算法
optimizer = optim.SGD(net.parameters(),lr=0.03)
#训练模型
num_epochs = 20
batch_size = 10
dataset = Data.TensorDataset(features, labels)# 将训练数据的特征和标签组合
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)# 随机读取小批量，shuffle决定随机
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)#输入X得到output
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()#通过调用optim实例的step函数来迭代模型参数,即w,b的迭代
    print('epoch %d, loss: %f' % (epoch, l.item()))
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
