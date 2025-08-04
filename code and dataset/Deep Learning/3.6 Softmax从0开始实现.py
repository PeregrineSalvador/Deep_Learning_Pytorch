import torch
from IPython import display
from d2l import torch as d2l
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 256
def load_data_fashion_mnist(batch_size, num_workers=4): 
    # 在windows上，如果想用多线程，必须定义__name__进入主循环;否则就只能用单线程
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST 的均值和标准差
    ])
    
    # 加载训练集和测试集
    train_data = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_data = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # 创建 DataLoader
    train_iter = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_iter = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_iter, test_iter

# 使用自定义的加载函数
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size, num_workers=4) 

num_inputs = 784 # 输入是28*28 = 784 的
num_outputs = 10 # 输出共有十个类别

class Accumulator: # 累加器
    def __init__(self, n):
        self.data = [0.0] * n # 这里不是乘法运算，是重复的次数，也就是维度 n = 2


    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


W = torch.normal(0, 0.01, size = (num_inputs, num_outputs), requires_grad= True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    # 这里输出的行数是batch_size 用以表示不同批次的训练 列数表示分类的类别数量
    X_exp = torch.exp(X) # 先将每一个元素都求幂指数
    partition = X_exp.sum(1, keepdim=True) # 每一行累加求得行内分母，因为是分批次进行训练的，每一个批次要单独计算
    return X_exp / partition 

def net(X):
    # 就单纯一个线性网络而已
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y)), y])

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: #判断y_hat是否为高维，如果维度高则进行处理
        y_hat = y_hat.argmax(axis=1) # 获取概率最大的类别索引
    cmp = y_hat.type(y.dtype) == y # 生成布尔张量
    return float(cmp.type(y.dtype).sum()) # 保证类型一致

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X), y), y.numel()) # y.numel()返回的是当前批量的样本数
    return metric[0]/metric[1] #将X输入，与正确的y标签进行判断正确率

if __name__ == '__main__':
    print(evaluate_accuracy(net, test_iter)) # 这里是随机猜测，因而打印出来值应当接近0.1