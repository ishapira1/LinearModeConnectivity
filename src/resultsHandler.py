from lmc_utilis import get_network_parameters, l2_distance_weight_space, compute_loss, get_model_predictions

from lmc_utilis import compute_interpolated_metrics
import time
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from os.path import join, exists, isfile
import hashlib
import datetime
import os
import re






def save_model(model, optimizer, save_path, name, epoch=0):
    """
    Save a PyTorch model and its optimizer to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        save_path (str): The path where the model and optimizer will be saved.
        epoch (int): The current training epoch.
    """
    full_path = os.path.join(save_path, f"model_checkpoint_{name}_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, full_path)


def load_checkpoint(model, optimizer, load_path, device):
    """
    Load a PyTorch model and its optimizer from a file.

    Args:
        model (torch.nn.Module): The model to load.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        load_path (str): The path from where the model and optimizer will be loaded.
        device (str): The device to load the model on (e.g., 'cpu' or 'cuda').
    """
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    model.to(device)

    return model, optimizer, epoch


