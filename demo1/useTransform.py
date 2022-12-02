from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img_url = "../demo01/dataset/train/ants_image/0013035.jpg"
img = Image.open(img_url)

# ToTensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)
writer.add_image("Tensor Image",tensor_img)

# Normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([6,3,2],[9,3,2])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image("Normalize Image",img_norm,2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
# print(img_resize)<PIL.Image.Image image mode=RGB size=512x512 at 0x1BD84EB7448>
img_resize = tensor_trans(img_resize)
writer.add_image("Resize Image",img_resize,0)

# Compost - resize -2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, tensor_trans])
img_resize_2 = trans_compose(img)
writer.add_image("Resize Image",img_resize_2,1)

writer.close()
