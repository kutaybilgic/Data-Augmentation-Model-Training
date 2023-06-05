import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from customDataset import ProductDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

my_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.05),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])

root_dir = 'D:/Github/Data-Augmentation-Model-Training/products'
product_folders = os.listdir(root_dir)

# Tüm ürün klasörleri için augmentation yap
for product_folder in product_folders:
    label_dir = f'D:/Github/Data-Augmentation-Model-Training/train/{product_folder}'
    os.makedirs(label_dir, exist_ok=True)

    # Ürün klasörü içindeki her bir fotoğrafı augmente et
    product_path = os.path.join(root_dir, product_folder)
    image_files = os.listdir(product_path)
    for image_file in image_files:
        image_path = os.path.join(product_path, image_file)

        # HEIC dosyası ise JPEG formatına dönüştür
        if os.path.splitext(image_file)[1].lower() == '.heic':
            image = Image.open(image_path)
            image = image.convert('RGB')
        else:
            image = Image.open(image_path)

        # Her bir fotoğrafı 10 farklı şekilde augmente et
        for i in range(10):
            augmented_img = my_transforms(image)
            save_image(augmented_img, f'{label_dir}/{image_file}_copy{i}.png')

print("Veriler başarıyla kaydedildi.")
