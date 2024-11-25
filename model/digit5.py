import torch.nn as nn
import torch.nn.functional as F
from model.MNs9 import SBatchNorm2d
from model.MNs9 import SBatchNorm1d
from model.enhancement import augment_data


class CNN(nn.Module):
    def __init__(self, data_parallel=True):
        super(CNN, self).__init__()
        encoder = nn.Sequential()  # 创建容器用来堆叠模型
        # 第一个卷积块
        encoder.add_module("conv1", nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))
        encoder.add_module("bn1", SBatchNorm2d(64))  # 64是输入通道数，与卷积层输出通道数保持一致
        encoder.add_module("relu1", nn.ReLU())
        encoder.add_module("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        # 第二个卷积块
        encoder.add_module("conv2", nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        encoder.add_module("bn2",SBatchNorm2d(64))
        encoder.add_module("relu2", nn.ReLU())
        encoder.add_module("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        # 第三个卷积块
        encoder.add_module("conv3", nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2))
        encoder.add_module("bn3", SBatchNorm2d(128))
        encoder.add_module("relu3", nn.ReLU())
        if data_parallel:
            self.encoder = nn.DataParallel(encoder)
        else:
            self.encoder = encoder
        linear = nn.Sequential()
        # 定义两个全连接层
        linear.add_module("fc1", nn.Linear(8192, 3072))
        linear.add_module("bn4", SBatchNorm1d(3072))
        linear.add_module("relu4", nn.ReLU())
        linear.add_module("dropout", nn.Dropout())
        linear.add_module("fc2", nn.Linear(3072, 2048))     # 这里定义的都是Linear(全连接层)
        linear.add_module("bn5", SBatchNorm1d(2048))
        linear.add_module("relu5", nn.ReLU())
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, x):                           # x:[128,3,.32,32],self是定义好的CNN网络
        batch_size = x.size(0)                      # 128
        feature = self.encoder(x)                   # [128，128，8，8]
        feature = feature.view(batch_size, 8192)    # [128,8192]
        feature = self.linear(feature)              # [128,2048]
        return feature


class Classifier(nn.Module):
    def __init__(self, data_parallel=True):
        super(Classifier, self).__init__()
        linear = nn.Sequential()
        linear.add_module("fc", nn.Linear(2048, 10))
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, x):           # [128,2048]
        x = self.linear(x)          # [128,10]
        return x

