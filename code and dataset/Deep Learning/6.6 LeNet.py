import torch
from torch import nn
from d2l import torch as d2l

 # Sequence是一个容器，用以存储不同的神经网络层
 # 输入通道数 输出通道数（卷积核数量） 卷积核大小5*5 图像外层往外扩充2个像素 实现28*28(1) --》》 28*28(6)
 # sigmoid激活一下，避免过大值
 # Avgpool 平均池化一下 28*28(6) --》》 14*14(6)
 # 输入通道为6，因为上一层卷积输出为6 完成14*14(6) --》》 10*10(16)
 # sigmoid激活一下，避免过大值
 # 平均池化 10*10(16) --》》 5*5(16)
 # 将输出铺平
 # 全连接层输入 16*5*5

'''Sequential是用以维护由Module组成的有序列表;本质是管理层的快'''
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10),
)

'''
@param:
批大小 表示同时要处理的图像数量
通道数 表示图像的色彩通道数
高度 图像的垂直像素数
宽度 图像的水平像素数
'''
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32) # 这里的像素维度不能随意改变，因为他明确了全连接层输入层的大小。这也是LeNet的弊端
# 逐层处理X，并且输出处理过程的信息 确保X的正确


batch_size = 256
# iterator 迭代器
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

'''
class Accumulator: # 累加器
    def __init__(self, n):
        self.data = [0.0] * n # 这里不是乘法运算，是重复的次数，也就是维度 n = 2


    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 能自动检测GPU 兼容模型输入
def evaluate_accuracy_gpu(net, data_iter, device = None):
    if isinstance(net, nn.Module):  # 判断net是不是nn.Module的实例或者子类的实例;sequential里面所有的层都是pytorch的实例
        net.eval() # 将模型切换为评估模式
        if device is None:
            device = next(iter(net.parameters())).device
        net.to(device)
    metric = d2l.Accumulator(2) # 返回的是长度为2的数组，请注意这个就是给你一个类，并不是进行正确率的计算
    # 类是类
    # 方法是方法
    # 函数是函数
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list): # 检查X是否为列表，如果是的话就把所有的东西都移动到指定设备上。这里的列表实际指的是多输入。
                # 例如孪生神经网络 Siamese 就是两个输入
                X = [x.to(device) for x in X]
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0]/metric[1]
'''


def evaluate_accuracy_gpu(net, data_iter, device=None):
    # 1. 确保模型是Module且切换到评估模式
    if isinstance(net, nn.Module):
        net.eval()
        # 2. 自动获取设备（如果未指定）
        if device is None:
            device = next(iter(net.parameters())).device
        # 3. 显式将模型转移到目标设备
        net.to(device)
    else:  # 如果是自定义模型（非nn.Module）
        if device is None:
            device = torch.device('cpu')  # 默认CPU

    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            # 4. 统一设备转移逻辑
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            
            # 5. 安全验证（可选）
            assert next(iter(net.parameters())).device == device, \
                f"Model is on {next(iter(net.parameters())).device}, but data is on {device}"
                
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

'''
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print("Training on", device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            test_acc = evaluate_accuracy_gpu(net, test_iter)
            print(f'loss {train_l:.3f},train_acc {train_acc:.3f},'f'test_acc{test_acc:.3f}')
'''

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print("Training on", device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)
    
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_samples
        net.train()
        timer.start()  # 计时整个 epoch
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        timer.stop()
        
        # 每个 epoch 结束后计算一次指标
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)  # 移到 batch 循环外
        print(f'epoch {epoch + 1}, loss {train_l:.3f}, '
              f'train_acc {train_acc:.3f}, test_acc {test_acc:.3f}, '
              f'time {timer.sum():.1f} sec')
        
lr, num_epochs = 0.05, 10
if __name__ == '__main__':
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, "output shape: \t",X.shape)
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())