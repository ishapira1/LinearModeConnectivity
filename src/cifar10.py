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

import torch
import torch.nn as nn

# ReducedVGG
class SimpleCIFAR10CNN(nn.Module):
    """
    we use a reduced version of the VGG-16 architecture (Simonyan and
    Zisserman [2014]). The modified model consists of four layers in total, with two convolutional
    layers followed by two fully connected layers. Max-pooling is performed after each convolutional
    layer, and dropout is applied after each of the fully connected layers. This reduced VGG model has
    855,000 parameters, making it less computationally intensive while maintaining the basic structure
    of the original VGG-16 architecture.
    """
    def __init__(self, num_classes=10):
        super(SimpleCIFAR10CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


