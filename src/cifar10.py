import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cifar10_dataset():
    """
    Load the CIFAR-10 dataset and apply the necessary preprocessing steps.

    Returns:
    - train_dataset (Dataset): The training dataset.
    - test_dataset (Dataset): The test dataset.
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


class SimpleCIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

