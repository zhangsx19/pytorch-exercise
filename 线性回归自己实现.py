import torch
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
sys.path.append("..")
from d2lzh_pytorch import *
#预先构筑一个线性回归模型y=wx+b+noise来验证
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)
print('第一个输入和输出是：')
print(features[0],labels[0].item())#只有一个元素的张量才能转为元素
#图像展示y与x的第一维/第二维的线性关系
set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(),s=1);#s为点的大小面积
#plt.show()#展示图像
#将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。需要对这些参数求梯度来迭代参数的值，因此requires_grad=True
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)),dtype=torch.float32,requires_grad=True)
b = torch.zeros(1, dtype=torch.float32,requires_grad=True)#广播机制wx+b
print('初始化w,b:')
print(w,b)
#训练模型
batch_size = 10#按10个样本一次迭代输入数据
net = linreg#神经网络是线性回归
loss = squared_loss#损失函数是平方和
lr = 0.03#单位样本学习率
num_epoch = 20
for epoch in range(num_epoch):
    for X,y in data_iter(batch_size,features,labels):#读取数据
        y_hat = net(X, w, b)#代入模型，算出y_hat
        losstotal = loss(y_hat, y).sum()#tensor(1)
        losstotal.backward()
        sgd([w,b], lr, batch_size)#尽可能使用矢量[w,b]计算，省时间
        #每次使用grad后记得梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_() 
    train_loss = loss(net(features,w,b),labels)#计算一次大迭代的损失率
    print('epoch %d, loss %f' % (epoch + 1, train_loss.mean().item()))
print('拟合的w,b:')
print(w,b)#与true_w，true_b极为接近


