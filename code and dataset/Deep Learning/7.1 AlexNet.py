import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Flatten(),

    nn.Linear(6400, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)

X = torch.randn(1, 1, 224, 224)
batch_size = 128

lr, num_epochs = 0.01, 10

acc_queue = [0.1, 0, 0.1, 0]

def calcu_delta_acc(cur_train_acc, cur_test_acc):
    acc_queue[1] = cur_train_acc
    acc_queue[3] = cur_test_acc
    delta_train_acc = acc_queue[1] - acc_queue[0]
    delta_test_acc = acc_queue[3] - acc_queue[2]
    acc_queue[0] = cur_train_acc
    acc_queue[2] = cur_test_acc
    delta_acc = [delta_train_acc, delta_test_acc]
    return delta_acc

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
        delta_acc = calcu_delta_acc(train_acc, test_acc)
        
        print(f'epoch {epoch + 1}, loss {train_l:.3f}, '
              f'train_acc {train_acc:.3f}, test_acc {test_acc:.3f}, '
              f'time {timer.sum():.1f} sec,'
              f'delta_train_acc {delta_acc[0]:.3f},'
              f'delta_test_acc {delta_acc[1]:.3f}')
        if(epoch == num_epochs):
            print('已经达到训练次数极限')
        if(test_acc > 0.9 or train_acc > 0.95):
            print("已经达到预期效果，提前退出训练")
            break

# 这里就不导入数据集了，因为AlexNet是基于巨大的ImageNet数据集训练的，如果想要达到同样的效果会占用极大的内存空间
# AlexNet要求输入图片必须是224*224像素
if __name__ == '__main__':
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())