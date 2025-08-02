'''
手搓图像分类数据集
'''

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

trans = transforms.ToTensor()
# 先将数据集的存在形式从PIL类型转换为Tensor，这样才能输入到Dataloader中去
minist_train = torchvision.datasets.FashionMNIST(
    root = "../data", train = True, transform=trans, download = True 
)

minist_test = torchvision.datasets.FashionMNIST(
    root = "../data", train = False, transform=trans, download = True 
)

def get_dataset_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker,', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_img(imgs, num_rows, num_cols, titles = None, scales =1.5):
    figsize = (num_cols * scales, num_rows * scales)
    _,axes = d2l.plt.subplot(num_rows, num_cols , figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy)
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

batch_size = 256

def get_dataloader_workers():
    return 4

# workers 可以理解成为工人数量，现在一共有四个工人并行进行数据处理。数据的加载和处理主要是在CPU上完成；而模型计算才是在GPU完成的
# workers 的数量通常为 workers = min(4, num_core - 1)
# 如果你手贱给多了，超过CPU所具有的核心数，就会死机
train_iter = data.DataLoader(minist_train, batch_size, shuffle = True, num_workers= get_dataloader_workers())

X, y = next(iter(data.DataLoader(minist_train, batch_size = 18)))

def load_data_fashion_minist(batch_size, resize = None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)
    minist_train = torchvision.datasets.FashionMNIST(
        root = "../data", train = True, transform=trans, download = True 
    )
    minist_test = torchvision.datasets.FashionMNIST(
        root = "../data", train = False, transform=trans, download = True 
    )
    return (data.DataLoader(minist_train,batch_size,shuffle=True,  num_workers= get_dataloader_workers()), 
            data.DataLoader(minist_test,batch_size,shuffle=True,  num_workers= get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_minist(32, resize = 36)

for X,y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break