import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from customDataset import ProductDataset
from sklearn.model_selection import StratifiedKFold

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.05),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0,0.0], std=[1.0,1.0,1.0])
    ])

dataset = ProductDataset(csv_file='products.csv', root_dir='/Users/drivers/Desktop/ceng318', transform=my_transforms)

n_splits = 5  # Change this value to the desired number of folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Get labels for stratification
_, labels = zip(*dataset)

# Perform the K-Fold split
img_num = 0
for fold, (train_idx, val_idx) in enumerate(skf.split(torch.zeros(len(labels)), labels)):
    os.makedirs(f'/Users/drivers/Desktop/ceng318/fold_{fold}', exist_ok=True)

    for idx in train_idx:
        img, label = dataset[idx]
        label_dir = f'/Users/drivers/Desktop/ceng318/fold_{fold}/train/label_{label}'
        os.makedirs(label_dir, exist_ok=True)
        save_image(img, f'{label_dir}/img{img_num}.png')
        img_num += 1

    for idx in val_idx:
        img, label = dataset[idx]
        label_dir = f'/Users/drivers/Desktop/ceng318/fold_{fold}/val/label_{label}'
        os.makedirs(label_dir, exist_ok=True)
        save_image(img, f'{label_dir}/img{img_num}.png')
        img_num += 1

print(img_num)
