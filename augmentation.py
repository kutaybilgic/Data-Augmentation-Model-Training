import time
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from customDataset import ProductDataset

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(brightness=0.5) ,
    transforms.RandomRotation(degrees=45) ,
    transforms.RandomHorizontalFlip(p=0.5) ,
    transforms.RandomHorizontalFlip(p=0.05),
    transforms.RandomGrayscale(p=0.2) ,
    transforms.ToTensor() ,
    transforms.Normalize(mean=[0.0, 0.0,0.0], std=[1.0,1.0,1.0])
    ])

dataset = ProductDataset(csv_file = 'products.csv', root_dir = '/Users/drivers/Desktop/ceng318',
                             transform = my_transforms)



img_num = 0

for _ in range(50):
    for img, label in dataset:
        if label == 0:
            os.chdir('/Users/drivers/Desktop/ceng318/train/burcak')
            save_image(img, "img"+str(img_num) + ".png")
            img_num += 1
        #keep going like that

print(img_num)