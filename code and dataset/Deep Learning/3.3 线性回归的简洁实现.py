'''
线性回归的简洁实现
'''

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn


# 样本生成
def synthetic_data(w,b,num_examples):
    """生成y = wx + b + 噪声"""
    x = torch.normal(0,1,(num_examples,len(w))) # x 是样本
    # 均值为0，标准差为1，生成形如(样本数，特征数)的张量。在这里，特征数要和w的个数相对应，否则无法进行下一步的矩阵惩罚
    y = torch.matmul(x,w) + b
    y += torch.normal(0,0.01,y.shape) # 添加误差项，误差是来自均值0和标准差0.01
    return x,y.reshape((-1,1)) # -1是自动计算y的维度，也就是行数。1是y的列数

true_w = torch.tensor([0.05,-0.01]) # 1D 张量，不要用矩阵来理解它！这里true_w的形状是(2,)单纯表示一维向量，长度为2而已！
true_b = 4.2 # 偏置
features , labels = synthetic_data(true_w,true_b,12) # 开始生成样本

def load_array(data_arrays, batch_size, is_train=True):
    """构建Pytorch 数据迭代器"""
    dataset = data.TensorDataset(*data_arrays) # 先将features和labels打包成数据对象
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # 然后又利用Dataloader将数据打乱和分批后返回 shuffle 是打乱、洗牌的意思
#这个数据迭代器调用一次会产生一次已经打乱过的训练集

batch_size = 10
data_iter = load_array((features, labels),batch_size)

net = nn.Sequential(nn.Linear(2, 1))
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(),lr = 0.03)
num_epoch = 200
for epoch in range(num_epoch):
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch:{epoch+1},loss:{float(l):f}')