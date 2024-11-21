"""
Sem 1. Train NN
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

# functions to show an image
import matplotlib.pyplot as plt
import numpy as np

from model import build_model, classes

batch_size = 4
model_path = 'models/cifar_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_images(trainloader):
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))


def train(optimizer_name='SGD'):
    print('Load data')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    net = build_model().to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    else:
        raise ValueError("Unsupported optimizer. Choose 'SGD' or 'Adam'.")

    # Логирование потерь
    losses_per_epoch = []

    for epoch in range(10):  # Number of epochs
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss for epoch
            running_loss += loss.item()
            epoch_loss += loss.item()

            # Print running loss every 200 mini-batches
            if i % 200 == 199:
                print(f'Epoch [{epoch + 1}], Batch [{i + 1}] Loss: {running_loss / 200:.3f}')
                running_loss = 0.0

        # Средняя потеря за эпоху
        avg_epoch_loss = epoch_loss / len(trainloader)
        losses_per_epoch.append(avg_epoch_loss)
        print(f'Epoch [{epoch + 1}] Average Loss: {avg_epoch_loss:.3f}')

    # Построение графика
    plot_loss(losses_per_epoch, optimizer_name)

    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    torch.save(net.state_dict(), f'models/{optimizer_name.lower()}_model.pth')


def plot_loss(losses, optimizer_name):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', label=f'Loss ({optimizer_name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss per Epoch ({optimizer_name})')
    plt.legend()
    plt.grid()
    plt.savefig(f'loss_plot_{optimizer_name.lower()}.png')
    plt.close()
    print(f"Loss plot saved as 'loss_plot_{optimizer_name.lower()}.png'")


if __name__ == '__main__':
    # Train with SGD
    train(optimizer_name='SGD')

    # Train with SparseAdam
    train(optimizer_name='Adam')
