import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torchvision
import pathlib
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#checking for device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Transforms
transformer=transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])

writer = SummaryWriter('runs/experiment_2')


#Dataloader

#Path for training and testing directory
# C:\Users\Çağdaş\Desktop\pytorch_projects\scene_detection\seg_train
train_path='D:/Github\Data-Augmentation-Model-Training/train'
test_path='D:/Github\Data-Augmentation-Model-Training/test'

train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transformer),
    batch_size=64, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=transformer),
    batch_size=32, shuffle=True
)

#categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)

# CNN Network


class ConvNet(nn.Module):
    def __init__(self, num_classes=12):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.3)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.4)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.5)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.6)
        )

        self.fc = nn.Linear(in_features=18 * 18 * 32, out_features=num_classes)

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = output.view(-1, 18 * 18 * 32)
        output = self.fc(output)
        return output

model=ConvNet(num_classes=12).to(device)
model.cuda()


#Optmizer and loss function
optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
loss_function=nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 30

#calculating the size of training and testing images
train_count=len(glob.glob(train_path+'/**/*.png'))
test_count=len(glob.glob(test_path+'/**/*.jpg'))

#Optmizer, loss function and learning rate scheduler


# Model training and saving best model

best_accuracy = 0.0

train_acc_arr = []
test_acc_arr = []
train_loss_arr = []

for epoch in range(num_epochs):

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count
    scheduler.step()

    # Evaluation on testing dataset
    model.eval()

    test_accuracy = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

    test_acc_arr.append(test_accuracy)
    train_acc_arr.append(train_accuracy)
    train_loss_arr.append(train_loss)

    writer.add_scalar('Training Loss', train_loss, epoch)
    writer.add_scalar('Training Accuracy', train_accuracy, epoch)
    writer.add_scalar('Test Accuracy', test_accuracy, epoch)

    # Save the best model
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), 'best_checkpoint.model')
        best_accuracy = test_accuracy

def plot_train_acc():

    accuracies = train_acc_arr
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()

def plot_test_acc():
    accuracies = test_acc_arr
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('test accuracy')
    plt.title('Test accuracy vs. No. of epochs')
    plt.show()


def plot_train_loss():
    accuracies = train_loss_arr
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.title('Train loss vs. No. of epochs')
    plt.show()


plot_train_acc()
plot_test_acc()
plot_train_loss()