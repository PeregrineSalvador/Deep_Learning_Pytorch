'''
需要注意的是,torch默认不会跟踪requires_grad=False时的张量计算历史。但是在一些明确不需跟踪张量的地方
d2l只是一个起到很小作用的辅助库，不太需要在意
'''

import random
import torch
from d2l import torch as d2l

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
features , labels = synthetic_data(true_w,true_b,12)

# d2l.set_figsize() # 调整图片大小
# d2l.plt.scatter(features[:, 1].detach().numpy(),labels.detach().numpy(),1) # 以1000个样本的第二个特征作为横轴，所有labels作为纵轴
# d2l.plt.savefig('scatter_plot.png')  # 保存到当前目录

# 数据迭代器，iter = iterator n. 迭代器
def data_iter(batch_size, features, labels):
    """生成一个小批量batch.传入参数有批量的大小，特征和标签。虽然这样把事情变麻烦了，不过将处理内部化，减少了出错的可能；并且也减少了样板的代码，简洁易懂"""
    num_examples = len(features) # features这个张量形状是(1000,2),但是len值返回张量第一个维度的大小，也就是样本数量，这个是正确的
    indices = list(range(num_examples)) # 生成索引表
    random.shuffle(indices) # 打乱索引表的顺序，避免模型按固定顺序学习数据导致偏差
    for i in range(0, num_examples, batch_size): # 一次循环是针对一批batch
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)]
        ) # 截取从i开始的batch_size个索引，取min的目的是为了避免超限（最后一个batch不一定就满足batch_size）
        yield features[batch_indices],labels[batch_indices]
'''用切片的方式的到不同的batch,可以更方便GPU进行并行运算处理。并行运算处理比处理单个样本所增加的时间开销其实并不多'''

# 初始化参数
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1, requires_grad = True)

# 定义线性回归模型
def linreg(x,w,b):
    return torch.matmul(x,w) + b

# MSE 均方差损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法 Stochastic Gradient Descent SGD 只负责更新参数！
def sgd(params, lr, batch_size):
    with torch.no_grad(): # 禁止使用梯度跟踪
        for param in params: # 遍历需要优化的参数
            param -= lr * param.grad / batch_size  
            # 更新参数 lr以控制更新步长；由于我们接收的是一批样本损失的总和，因而要用batch_size归一化，相当于对损失取平均
            param.grad.zero_()
batch_size=10
lr = 0.03
num_epochs = 5
net = linreg # 函数名可直接赋值给变量，相当于起了一个函数的别名，其实很无聊
loss = squared_loss

for epoch in range(num_epochs):
    for x , y in data_iter(batch_size, features, labels): # 相当于取样
        l = loss(net(x,w,b),y)
        l.sum().backward()
        # 这里不需要担心l.sum()是一个标量。虽然反向传播要求计算的是函数对变量的偏导数，但是pytorch会动态跟踪所有张量操作，构成计算图。并且backward是需要标量输入的！
        sgd([w,b], lr, batch_size) # 更新参数
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch:{epoch+1},loss:{float(train_l.mean()):f}')
