from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import os
from resultsHandler import *

from train import train_model, train_one_epoch

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



def load_model_from_saved(load_path, device, dataset):
    """
    Load a PyTorch model and its optimizer from a file.

    This function loads the model's parameters (state_dict) and the optimizer's state
    (state_dict) from a file. This allows you to continue training the model or use it
    for inference.

    Args:
        load_path (str): The path from where the model and optimizer will be loaded.
        device (torch.device): The device to load the model and optimizer on.

    Returns:
        model (torch.nn.Module): The loaded model.
        optimizer (torch.optim.Optimizer): The loaded optimizer.
        epoch (int): The last saved training epoch.
    """
    checkpoint = torch.load(load_path, map_location=device)

    # Initialize the model and optimizer based on the dataset
    model = load_model(dataset)
    optimizer = torch.optim.Adam(model.parameters())

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch


def train_and_save_epochs(dataset, max_epochs, batch_size):
    """
    Train the model and save its state for every epoch.

    Args:
        dataset (str): The dataset to use (MNIST, CIFAR-10, or ImageNet).
        max_epochs (int): The maximum number of epochs to train the model.
        batch_size (int): The batch size for training.
    """

    # Load the dataset
    train_dataset, test_dataset = load_dataset(dataset)
    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    # Load the model and optimizer
    initial_model = load_model(dataset)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(initial_model.parameters())

    # Set the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model and save its state for every epoch
    total_steps = 0
    for epoch in range(max_epochs + 1):
        name = f"pre_trained_{dataset}_b{batch_size}_steps_{total_steps}_epoch_{epoch}_run2"
        os.makedirs("./results/pre_trained", exist_ok=True)
        save_model(initial_model, optimizer, name, epoch=epoch, save_path="./results/pre_trained")

        print(f"Training epoch {epoch}")
        initial_model, epoch_loss, steps = train_one_epoch(initial_model, optimizer, criterion, train_loader, device)
        total_steps += steps





def find_saved_model_file(dataset, batch_size, epoch, directory="./results/pre_trained", run2 = False):
    pattern = f"pre_trained_{dataset}_b{batch_size}_steps_*_epoch_{epoch}_pre_trained.pt"
    if run2:
        # exe: pre_trained_MNIST_b256_steps_470_epoch_2_run2_pre_trained
        directory = os.path.join(directory, "run2")
        pattern = f"pre_trained_{dataset}_b{batch_size}_steps_*_epoch_{epoch}_run2_pre_trained.pt"

    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, pattern):
            full_path = os.path.join(directory, filename)
            return full_path
    raise FileNotFoundError(f"No matching model found for dataset: {dataset}, batch_size: {batch_size}, epoch: {epoch}")

def find_continued_model_file(dataset, batch_size, k1, k2,  directory="./results/continued/", run2 = False):
    directory = os.path.join(directory, f"epoch_{k1}", f"b{batch_size}")
    pattern = f"continued_{dataset}_b{batch_size}_pre_trined_steps_"


    # if run2:
    #     # exe: pre_trained_MNIST_b256_steps_470_epoch_2_run2_pre_trained
    #     directory = os.path.join(directory, "run2")
    #     pattern = f"pre_trained_{dataset}_b{batch_size}_steps_*_epoch_{epoch}_run2_pre_trained.pt"
    for filename in os.listdir(directory):
        if pattern in filename:
            if f"_epoch_{k2}_new_steps_" in filename:
                print(f'===== > > > file has been found. k2={k2}', filename, f"_epoch_{k2}_new_steps_" in filename)
                full_path = os.path.join(directory, filename)
                return full_path

    raise FileNotFoundError(f"No matching model found for dataset: {dataset}, batch_size: {batch_size}, epoch: {k1}, directory={directory}, pattern = {pattern}")

# path1 = f"continued_{dataset}_b{batch_size}_pre_trined_steps_0_epoch_10_new_steps_1960.pt"

def extract_steps_from_model_path(model_path):
    print(model_path, "model_path")
    pattern = r"_steps_(\d+)_"
    match = re.search(pattern, model_path)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not find the number of steps in the model path: {model_path}")


def file_with_prefix_exists(directory, prefix):
    """
    Check if a file with the given prefix exists in the specified directory.

    Args:
        directory (str): The path to the directory where you want to search for the file.
        prefix (str): The prefix of the file name you want to search for.

    Returns:
        bool: True if a file with the given prefix exists, False otherwise.
    """
    try:
        for filename in os.listdir(directory):
            if filename.startswith(prefix):
                return True

        return False
    except:
        return False
def load_and_continue_training(dataset, batch_size, epoch, max_epoch, device, run2 = False):
    """
    Load a saved model based on the input parameters, train it for the remaining epochs, and save the new model.

    Args:
        dataset (str): The dataset to use (MNIST, CIFAR-10, or ImageNet).
        batch_size (int): The batch size for training.
        epoch (int): The current epoch of the saved model.
        max_epoch (int): The maximum number of epochs to train the model.
    """

    """
    The load_model function loads the checkpoint from the provided load_path.
    It retrieves the dataset from the filename, which is used to initialize the appropriate model architecture.
    The optimizer is initialized as an Adam optimizer.
    The model's state_dict (weights) and the optimizer's state_dict are loaded from the checkpoint.
    The model is moved to the specified device.
    The last saved training epoch is returned.
    By using this function, you are effectively resuming the training process from the point where the model was last saved.
    The model will have the same architecture and weights, and the optimizer will also have the same state as it did when 
    the model was saved. This will allow you to continue training the model seamlessly.
    
    """

    # Find the model saved in the results directory
    model_path = find_saved_model_file(dataset, batch_size, epoch, directory="./results/pre_trained", run2 = run2)
    total_steps = extract_steps_from_model_path(model_path)

    additional_new_steps = 0



    # Load the model and optimizer
    model, optimizer, _ = load_model_from_saved(model_path, device, dataset)


    # Load the dataset and create a new train_loader
    train_dataset, test_dataset = load_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # Continue training the model for the remaining epochs
    for current_epoch in range(epoch + 1, max_epoch + 1):
        new_name = f"continued_{dataset}_b{batch_size}_pre_trined_steps_{total_steps}_epoch_{current_epoch}"
        if not run2:
            full_path = os.path.join(f"./results/continued/epoch_{epoch}/b{batch_size}")
            if file_with_prefix_exists(full_path, new_name): continue
        else:
            full_path = os.path.join(f"./results/continued/run2/epoch_{epoch}/b{batch_size}", new_name + ".pt")
            if file_with_prefix_exists(full_path, new_name): continue


        print(f"Training epoch {current_epoch}")
        model, epoch_loss, steps = train_one_epoch(model, optimizer, criterion, train_loader, device)
        additional_new_steps += steps



        # Save the new model
        new_name += "_new_steps_{additional_new_steps}"
        if run2:
            os.makedirs(f"./results/continued/run2/epoch_{epoch}/b{batch_size}", exist_ok=True)
            save_model(model, optimizer, new_name, epoch=current_epoch, save_path=f"./results/continued/run2/epoch_{epoch}/b{batch_size}")
        else:
            os.makedirs(f"./results/continued/epoch_{epoch}/b{batch_size}", exist_ok=True)
            save_model(model, optimizer, new_name, epoch=current_epoch, save_path=f"./results/continued/epoch_{epoch}/b{batch_size}")


def process_results_two_checkpoints(pretrained_model, continued_train_model, dataset, batch_size, k1, k2):
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
    # res.update({"alphas": alphas,
    #        "losses_test":losses_test, "accuracy_lst_test":accuracy_lst_test})
    return res


import json


def tensor_to_list(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = tensor_to_list(value)
        elif isinstance(value, torch.Tensor):
            d[key] = value.tolist()
    return d


def process_results(dataset, batch_size, max_epochs, epochs_range, save_directory=save_path):
    for k1 in epochs_range:
        for k2 in range(k1 + 1, max_epochs, 2):

            res_full_file = f"./processed/{dataset}_batch_{batch_size}_k1_{k1}_k2_{k2}.json"
            if os.path.isfile(res_full_file):

                res = load_dict_from_file(res_full_file)
                if "losses_train" in res.keys():
                    print(f"k1={k1}, k2={k2}. exists.")
                    continue
                else:
                    print("file exists, but without train")

            path1, path2 = get_model_paths(k1, k2, dataset, batch_size, save_directory)
            if not os.path.isfile(path1):
                continue
                raise FileNotFoundError(f"Checkpoint file not found: {path1}")
            if not os.path.isfile(path2):
                continue
                raise FileNotFoundError(f"Checkpoint file not found: {path2}")

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

            print(f"k1 = {k1}, k2={k2}")
            res = process_results_two_checkpoints(pretrained_model, continued_train_model, dataset, batch_size, k1, k2)
            res_converted = tensor_to_list(res)
            save_dict_to_file(res, res_full_file)

    return

def run_pretrained(dataset, max_epochs, batch_size = 128, save_directory="./checkpoints"):
    print(dataset)
    train_dataset, test_dataset = load_dataset(dataset)
    print("split Train/Loss")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("LOAD MODEL")
    initial_model = load_model(dataset)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(initial_model.parameters())
    save_model(initial_model, optimizer, save_directory, name = f"pre_trained_{dataset}_batch_{batch_size}", epoch=0)

    # Set the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(1, max_epochs):
        # train for one epoch
        initial_model = train_model(initial_model, optimizer, criterion, train_loader, test_loader, device, num_epochs=1)
        save_model(initial_model, optimizer, save_directory, name = f"pre_trained_{dataset}_batch_{batch_size}",
                   epoch=epoch)
    return


def continue_training_from_pretrained(dataset, epoch, max_epochs, batch_size=128, save_directory="./checkpoints"):
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






def run_algorithm(dataset, max_epochs, batch_size=128, save_path = "./checkpoints"):
    run_pretrained(dataset, max_epochs, batch_size, save_directory=save_path)
    for k1 in [0, 1, 2, 3, 5, 7, 10]:
        continue_training_from_pretrained(dataset, k1, max_epochs=max_epochs, batch_size=batch_size, save_directory=save_path)
    return


def main():
    run_algorithm("MNIST", 13, 1024)
    process_results("MNIST", 1024, 13, [0, 1, 2, 5, 7, 8])

    run_algorithm("CIFAR-10", 10, 128)
    process_results("MNIST", 128, 10, [0, 1, 2, 5, 7, 8])

    run_algorithm("TINYIMAGENET", 100, 64)
    return