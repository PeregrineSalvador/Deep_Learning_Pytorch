import torch
from torch import nn
from d2l import torch as d2l

num_inputs, num_outputs, num_hiddens = 784, 10, 256 # 784个输入接收 10个输出选择和256个隐藏层神经元

# 用于输入到隐藏层
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad= True) * 0.01
)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

# 用于隐藏到输出层
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad= True) * 0.01
)
b2 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

params = [W1, b1, W2, b2]

def relu(X):
    a = torch.zeros_like(X) # 创建一个参数和X一样的全0张量
    return torch.max(X, a)

def net(X):
    X = X.reshape(-1, num_inputs) # 输入层
    H = relu(X@W1 + b1) # 隐藏层
    return (H@W2 + b2) # 输出层

loss = nn.CrossEntropyLoss(reduce='none')

num_epochs, lr = 10, 0.1

updater = torch.optim.SGD(params, lr)

# 同3.7 后面的训练模型就不输入了