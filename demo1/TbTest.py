from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter(log_dir="logs")
img_path = "C:\\Users\\Jin Au-yeung\\Desktop\\Project\\DeepLearning\\demo1\\dataset\\train\\bees_image\\16838648_415acd9e3f.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)


# dataformats="HWC 调整数据格式
writer.add_image("test",img_array,2, dataformats="HWC")

for i in range(100):
    writer.add_scalar("y=2x",3*i,i)

writer.close()