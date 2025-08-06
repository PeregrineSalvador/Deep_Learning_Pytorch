import torch 
from torch import nn
from d2l import torch as d2l


'互相关运算,关于行数和列数的计算参考P160'
'其中K是kernel，核心的意思，也就是卷积核'
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1), X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

'卷积层就是用卷积核对输入进行互相关运算'
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zero(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
    
# 生成训练所需数据集
X = torch.ones((6, 8))
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)

# 创造二维卷积层，具有一个输出通道和形状为(1, 2)的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias = False)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2

'循环学习十次，让卷积核自行调整，达到学习效果'
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if i % 2 == 1:
        print(f'epoch{i + 1}, loss {l.sum():.3f}')

print(conv2d.weight.data.reshape((1, 2)))

# 多输入通道
def corr2d_multi_in(X, K):
    # X 是输入的图片信息，可能有多条通道输入
    # 例如 X 可为(2, 3, 3) 表示为两个通道 单个通道为3*3
    # 同理 K 可为(2, 2, 2, 2) 表示为两个输出通道，两个输入通道，卷积核本身为2*2
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K)) # 此时计算出来的结果是互相关预算

# 多输出通道
def corr2d_multi_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0) # 这里的0并不是运算，而是沿着dim=0的维度进行堆叠形成一个新的张量。也就是在输出通道上形成堆叠
'''
需注意，这里面的维度并不一定是等位的。例如K的维度分别意味着：输出通道、输入通道、每一个卷积核的高、每一个卷积核的宽
且另需注意，多输出必须建立在多输入上。因为多输出必须要兼容多输入，是多输入的拓展。
'''

'''
例如X = torch.tensor([[[1., 2.], [3., 4.]]]) 的表示
最外面括号代表通道数，往里一层是行数，再往里是列数，表示的是X.shape = (1, 2, 2)

# 卷积核 K: (2, 1, 2, 2) -> 2输出通道, 1输入通道, 2x2核
K = torch.tensor([
    [[[0., 1.], [2., 3.]]],  # 输出通道1的核
    [[[1., 0.], [0., 1.]]]   # 输出通道2的核
])  # shape=(2, 1, 2, 2)
'''

# 汇聚层 这里因为X未定义为张量，因而无法获取其shape
def pool2d(X, pool_size, mode):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1), (X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i ,j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'mean':
                Y[i ,j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

# 填充和步幅