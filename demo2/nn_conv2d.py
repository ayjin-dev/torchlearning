import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./demo2/data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

"""
batch_size: 一次取64个图片
"""
dataloader = DataLoader(dataset,batch_size=64)

class myNetwork(nn.Module):
    def __init__(self):
        # 父类初始化
        super(myNetwork,self).__init__()
        """
        定义一个卷积层
        Q1: 这里的几个参数是什么
        in_channels=3, 因为我们是彩色图，三通道所以为3
        out_channels=6, 
        kernel_size=3, 卷积核的大小3*3
        stride=1, 一次卷积移动的步长，左右
        padding=0 填充数值
        """
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

        """
        池化操作
        最大池化，取我们池化核范围内的最大值。
        ceil_model:不取整保存
        floor_model:取整保存
        """
    def forward(self,x):
        x = self.conv1(x)
        return x

myNetwork = myNetwork()
print(myNetwork)

writer = SummaryWriter("./demo2/logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = myNetwork(imgs)
    # print(output.shape)
    writer.add_images("input",imgs, step)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output", output, step)
    step = step + 1