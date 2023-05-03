

import torch
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_imagenet_model():
    #  load a ResNet-50 model that has already been trained on the ImageNet dataset
    # pre-trained ResNet-50 model provided by PyTorch
    model = models.resnet18(pretrained=True) # Legacy weights with accuracy 76.130%
    # Acc@1 69.758
    # Acc@5 89.078
    # Params 11.7M
    # GFLOPS 1.81
    model = model.to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 200)
    return model



# Load a pre-trained ResNet-50 model

