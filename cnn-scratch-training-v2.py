import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transformer = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

train_path = '/Users/drivers/Desktop/ceng318/train/'
test_path = '/Users/drivers/Desktop/ceng318/test/'

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=32, shuffle=True
)

root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)

#Define a class that contains the architecture of the pre-trained model
class MyPretrainedModel(nn.Module):
    def __init__(self, num_classes=16):
        super(MyPretrainedModel, self).__init__()
        # Define your model layers here

    def forward(self, x):
        # Define forward propagation here
        return x

# Load your pre-trained model
pretrained_model_path = "best_checkpoint.model"
num_existing_classes = 16
num_new_classes = 4
num_total_classes = num_existing_classes + num_new_classes
model = MyPretrainedModel(num_existing_classes)
model.load_state_dict(torch.load(pretrained_model_path))

# Change your classification layer and set the optimizer to train the weights of this layer only
model.fc = nn.Linear(model.fc.in_features, num_total_classes)
optimizer = Adam(model.fc.parameters(), lr=0.001, weight_decay=0.0001)

model = model.to(device)

loss_function = nn.CrossEntropyLoss()

num_epochs = 30

train_count = len(glob.glob(train_path + '/**/*.png'))
test_count = len(glob.glob(test_path + '/**/*.jpeg'))

best_accuracy = 0.0
train_acc_arr = []
test_acc_arr = []
train_loss_arr = []

for epoch in range(num_epochs):
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

