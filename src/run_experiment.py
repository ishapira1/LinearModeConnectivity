# @ itai shapira

from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import os
from resultsHandler import *

from train import train_model, train_one_epoch
import json
from lmc_utilis import get_network_parameters, set_network_parameters
import time
import fnmatch
import re

# datasets*:
from mnist import *
from cifar10 import *
from imagenet import *

def load_dataset(dataset_name):
    if dataset_name == "MNIST":
        return load_mnist_dataset()
    elif dataset_name == "CIFAR-10":
        return load_cifar10_dataset()
    elif dataset_name == "IMAGENET":
        return load_imagenet_dataset()
    return

def load_model(dataset_name):
    if dataset_name == "MNIST":
        return LeNet5()
    elif dataset_name == "CIFAR-10":
        return SimpleCIFAR10CNN()
    elif dataset_name == "IMAGENET":
        return load_imagenet_model()
    return




def process_results_two_checkpoints(pretrained_model, continued_train_model, dataset, batch_size, k1, k2):
    """
    Compare the performance of two models (pretrained and continued trained) on a specific dataset
    by computing various metrics such as losses and accuracy.

    Args:
        pretrained_model (nn.Module): The pretrained neural network model.
        continued_train_model (nn.Module): The continued trained neural network model.
        dataset (str): The dataset used for training and evaluating the models ('MNIST', 'CIFAR-10', etc.).
        batch_size (int): The batch size used during training.
        k1 (int): The epoch number for the pretrained model.
        k2 (int): The epoch number for the continued trained model.

    Returns:
        res (dict): A dictionary containing the computed metrics for the pretrained and continued trained models.
    """
    # Load dataset
    train_dataset, test_dataset = load_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    res = {"dataset": dataset, "k1": k1, "k2": k2}

    res.update({"distance": distance_in_weight_space(pretrained_model, continued_train_model)})

    losses_train, accuracy_lst_train, alphas = compute_metrics_models(pretrained_model, continued_train_model,
                                                                      train_loader, 25,
                                                                      device)
    print("barrier", losses_train[-1], losses_train[0], np.max(losses_train), np.max(losses_train) / losses_train[-1])
    losses_test, accuracy_lst_test, alphas = compute_metrics_models(pretrained_model, continued_train_model,
                                                                    test_loader, 25,
                                                                    device)
    print("barrier", losses_test[-1], losses_test[0], np.max(losses_test))
    print(losses_test)
    res.update({"alphas": alphas,
                "losses_train": losses_train, "accuracy_lst_train": accuracy_lst_train,
                "losses_test": losses_test, "accuracy_lst_test": accuracy_lst_test})
    return res



def tensor_to_list(d):
    """
    Convert any torch.Tensor objects in a dictionary to lists.

    Args:
        d (dict): A dictionary that may contain torch.Tensor objects as values.

    Returns:
        d (dict): The input dictionary with torch.Tensor objects converted to lists.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = tensor_to_list(value)
        elif isinstance(value, torch.Tensor):
            d[key] = value.tolist()
    return d


def process_results(dataset, batch_size, max_epochs, epochs_range):
    """
    Process the results of two models (pretrained and continued trained) for a specific dataset
    and a range of epochs.

    Args:
        dataset (str): The dataset used for training and evaluating the models ('MNIST', 'CIFAR-10', etc.).
        batch_size (int): The batch size used during training.
        max_epochs (int): The maximum number of epochs for training.
        epochs_range (list): The list of epochs for which the models are evaluated.

    Returns:
        res_converted (dict): A dictionary containing the performance metrics for the pretrained
                              and continued trained models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k1 in epochs_range:
        for k2 in range(k1 + 1, max_epochs, 2):

            res_full_file = f"./../processed/{dataset}_batch_{batch_size}_k1_{k1}_k2_{k2}.json"
            # if os.path.isfile(res_full_file):
            #
            #     res = load_dict_from_file(res_full_file)
            #     if "losses_train" in res.keys():
            #         continue
            #     else:
            #         print("file exists, but without train")
            #
            # path1, path2 = get_model_paths(k1, k2, dataset, batch_size, save_directory)
            # if not os.path.isfile(path1):
            #     continue
            #     raise FileNotFoundError(f"Checkpoint file not found: {path1}")
            # if not os.path.isfile(path2):
            #     continue
            #     raise FileNotFoundError(f"Checkpoint file not found: {path2}")

            # Load model
            model = load_model(dataset)
            model.to(device)
            optimizer = optim.Adam(model.parameters())
            # Load checkpoint
            pretrained_model, optimizer1, _ = load_checkpoint(model, optimizer, path1, device)

            model = load_model(dataset)
            model.to(device)
            optimizer = optim.Adam(model.parameters())
            # Load checkpoint
            continued_train_model, optimizer2, _ = load_checkpoint(model, optimizer, path2, device)

            res = process_results_two_checkpoints(pretrained_model, continued_train_model, dataset, batch_size, k1, k2)
            res_converted = tensor_to_list(res)
            save_dict_to_file(res, res_full_file)

    return res_converted

def run_pretrained(dataset, max_epochs, batch_size = 128, save_directory="./../checkpoints"):
    """
    Train a model from scratch for a given dataset, batch size, and maximum number of epochs.

    Args:
        dataset (str): The name of the dataset to be used for training (e.g., 'MNIST', 'CIFAR-10').
        max_epochs (int): The maximum number of epochs to train the model.
        batch_size (int, optional): The batch size for training and testing. Default is 128.
        save_directory (str, optional): The directory where the checkpoint files are saved. Default is "./../checkpoints".

    Returns:
        None

    Example:
        # Train a model from scratch for the MNIST dataset for a total of 10 epochs
        run_pretrained('MNIST', 10, batch_size=128, save_directory="./../checkpoints")
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = load_dataset(dataset)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    initial_model = load_model(dataset)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(initial_model.parameters())
    save_model(initial_model, optimizer, save_directory, name = f"pre_trained_{dataset}_batch_{batch_size}", epoch=0)

    # Set the device for training
    for epoch in range(1, max_epochs):
        # train for one epoch
        initial_model = train_model(initial_model, optimizer, criterion, train_loader, test_loader, device, num_epochs=1)
        save_model(initial_model, optimizer, save_directory, name = f"pre_trained_{dataset}_batch_{batch_size}",
                   epoch=epoch)
    return


def continue_training_from_pretrained(dataset, epoch, max_epochs, batch_size=128, save_directory="./../checkpoints"):
    """
    Continue training a pre-trained model from a specified epoch using a checkpoint file.

    Args:
        dataset (str): The name of the dataset to be used for training (e.g., 'MNIST', 'CIFAR-10').
        epoch (int): The epoch number to continue training from.
        max_epochs (int): The maximum number of epochs to train the model.
        batch_size (int, optional): The batch size for training and testing. Default is 128.
        save_directory (str, optional): The directory where the checkpoint files are saved. Default is "./checkpoints".

    Returns:
        model (nn.Module): The trained model after the specified number of epochs.

    Raises:
        FileNotFoundError: If the specified checkpoint file is not found in the save_directory.

    Example:
        # Continue training the model from the 5th epoch for the MNIST dataset for a total of 10 epochs
        continue_training_from_pretrained('MNIST', 5, 10, batch_size=128, save_directory="./checkpoints")
    """
    # Find checkpoint file
    checkpoint_file = os.path.join(save_directory,
                                   f"model_checkpoint_pre_trained_{dataset}_batch_{batch_size}_epoch_{epoch}.pt")

    # Check if checkpoint file exists
    if not os.path.isfile(checkpoint_file):
        print(checkpoint_file)
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    # Load dataset
    train_dataset, test_dataset = load_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Load model
    model = load_model(dataset)
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    # Load checkpoint
    model, optimizer, _ = load_checkpoint(model, optimizer, checkpoint_file, device)
    model.to(device)

    # Train model from the specified epoch to max_epochs
    for current_epoch in range(epoch + 1, max_epochs + 1):
        # Train for one epoch
        name=f"continue_train_from_{epoch}_{dataset}_batch_{batch_size}"
        full_path = os.path.join(save_directory, f"model_checkpoint_{name}_epoch_{epoch}.pt")

        if os.path.isfile(full_path):
          pass
        else:

          model = train_model(model, optimizer, criterion, train_loader, test_loader, device, num_epochs=1)
        # Save the model after each epoch
          save_model(model, optimizer, save_directory, name=f"continue_train_from_{epoch}_{dataset}_batch_{batch_size}",
                   epoch=current_epoch)

    return model






def run_algorithm(dataset, max_epochs, batch_size=128, save_path = "../checkpoints"):
    run_pretrained(dataset, max_epochs, batch_size, save_directory=save_path)
    for k1 in [0, 1, 2, 3, 5, 7, 10]:
        continue_training_from_pretrained(dataset, k1, max_epochs=max_epochs, batch_size=batch_size, save_directory=save_path)
    return


def main():
    # Run the algorithm for the MNIST dataset
    run_algorithm("MNIST", 13, 1024)
    process_results("MNIST", 1024, 13, [0, 1, 2, 5, 7, 8])

    # Run the algorithm for the CIFAR-10 dataset
    run_algorithm("CIFAR-10", 15,  128)
    process_results("CIFAR-10", 128, 15, [0, 1, 2, 5, 7, 8])

    # Run the algorithm for the TINYIMAGENET dataset
    run_algorithm("TINYIMAGENET", 100, 64)
    process_results("CIFAR-10", 64, 100, [0, 1, 2, 5, 7, 8])
    return