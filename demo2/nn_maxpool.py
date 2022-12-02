import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

"""
为什么要使用池化呢？
尽量的去保留数据的主要特征，同时减少数据量。

"""
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]],dtype=torch.float32)
# reshape(input_data,(union[]))
input = torch.reshape(input,(-1,1,5,5))
print(input)

dataset = torchvision.datasets.CIFAR10("./demo2/data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset, batch_size=64)
class myNetwork(nn.Module):
    def __init__(self):
        # 初始化父类
        super(myNetwork, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,input):
        output = self.maxpool1(input)
        return output

myNetwork = myNetwork()
# output = myNetwork(input)
# print(output)

writer = SummaryWriter("./logs_maxpool")
step = 0
for data in dataloader:
    imgs, target = data
    writer.add_images("input",imgs,step)
    output = myNetwork(imgs)
    writer.add_images("output",output,step)
    step = step + 1

writer.close()