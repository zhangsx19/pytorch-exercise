import torch
import numpy as np
import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l

def fit_and_plot(train_features, test_features, train_labels, test_labels,loss,num_epochs):#模型定义函数
    net = torch.nn.Linear(train_features.shape[-1], 1)
    # 通过Linear文档可知，pytorch已经将参数初始化了，所以我们这里就不手动初始化w,b了
    batch_size = min(10, train_labels.shape[0])    
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()#若梯度有值则清零梯度
            l.backward()
            optimizer.step()#调用优化方法优化参数
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,'\nbias:', net.bias.data)

#人工生成数据集，三阶多项式函数
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5#训练数据集和测试数据集的样本数都设为100
features = torch.randn((n_train + n_test, 1))
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1) 
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
#定义迭代次数和损失函数
num_epochs, loss = 100, torch.nn.MSELoss()
#拟合三阶多项式,即输入层有3个节点
#fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],labels[:n_train], labels[n_train:],loss,num_epochs)
#拟合线性函数，即输入层1个节点（欠拟合）
#fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train],labels[n_train:],loss,num_epochs)
#训练样本不足（过拟合） 三阶多项式
fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2], labels[n_train:],loss,num_epochs)