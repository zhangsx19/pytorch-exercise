import time
import torch
from torch import nn, optim
import sys
sys.path.append("..") 
import os
import d2l
#gpu计算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#定义网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        #卷积层块 --提取特征
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2),#默认strike与size同
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2)
        )
        #fully connected layer --对特征进行分类（MLP分类器 --Linear)
        self.fc = nn.Sequential(
            nn.Flatten(),#把[batchsize,16,5,5]展平成[batchsize,400]
            nn.Linear(in_features=16*5*5,out_features=120),
            nn.Sigmoid(),
            nn.Linear(in_features=120,out_features=84),
            nn.Sigmoid(),
            nn.Linear(in_features=84,out_features=10)#要算交叉熵softmax所以最后一层不做激活
        )
    def forward(self,img):
        features = self.conv(img)
        return self.fc(features)

net = LeNet()
#print(net)
#读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
loss = nn.CrossEntropyLoss()
#训练模型
lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)