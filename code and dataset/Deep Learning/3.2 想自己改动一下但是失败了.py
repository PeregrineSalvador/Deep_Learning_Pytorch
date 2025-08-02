import torch
import random
from d2l import torch as d2l

# 生成数据集函数
def create_data(w,b,max_example):
    n_features, n_outputs = w.shape    
    X = torch.normal(0,2,(max_example,n_features))
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.05,y.shape)
    return X, y

# 进行数据集生成
create_w = torch.tensor([[0.02, 0.5], [-0.01, 0.3], [0.03, -0.2]])
create_b = torch.tensor([4.2, 1.0])
features, labels = create_data(create_w, create_b, 500)
print(features)
# 生成训练所用到的batch
def create_batch(batch_size, features, labels):
    num_example = len(features)
    index = list(range(num_example))
    random.shuffle(index)
    for i in range(0, num_example, batch_size):
        batch_index = torch.tensor(
            index[i:min(i+batch_size,num_example)]
        )
        yield features[batch_index], labels[batch_index]

w = torch.normal(0,0.01,size=(3,1), requires_grad=True)
b = torch.zero(1, requires_grad = True)

def linreg(X, w, b):
    return torch.matmul(X,w) + b

# y_hat 是预报值， 用预报值减去真实值得到误差 真实值的输入是高维度向量，为了使其能够与y_hat相匹配，应对其进行reshape的操作
def MSE_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat)) ** 2 / 2

def SGD(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

batch_size = 10
lr = 0.05
num_epochs = 5

for epoch in range(num_epochs):
    for X, y in create_batch(batch_size, features, labels):
        loss = MSE_loss(linreg(X,w,b) - y)
        loss.sum().backward()
        SGD([w,b], lr, batch_size)
    with torch.no_grad():
        print(f'epoch:{epoch+1},loss:{float(loss.mean()):f}')