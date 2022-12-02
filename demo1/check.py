from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

"""
import torch
print('torch gpu status: ',torch.cuda.is_available())
print('torch gpu counts: ',torch.cuda.device_count())
print('torch gpu device name: ',torch.cuda.get_device_name(0))
print('torch gpu current device name: ',torch.cuda.current_device())

# torch gpu status:  True
# torch gpu counts:  1
# torch gpu device name:  NVIDIA GeForce RTX 3060
# torch gpu current device name:  0
"""

writer = SummaryWriter("logs")
img_url = "../demo01/dataset/train/ants_image/0013035.jpg"
img = Image.open(img_url)
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x1A26E1A02C8>
tensor_trans = transforms.ToTensor()
# Finish PIL to tensor.
tensor_img = tensor_trans(img)
print(tensor_img)
writer.add_image("Tensor Image",tensor_img)

writer.close()

# img.show()