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



def name_the_results_file(dataset_name, batch_size, k1):
    return f"{dataset_name}_b{batch_size}_{k1}"



def create_experiment_id():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    hashed_time = hashlib.md5(current_time.encode()).hexdigest()
    return hashed_time

def check_experiment_exists(filename, directory = "./results"):
    """
    Check if a file exists in the specified directory.

    Args:
        filename (str): The name of the file to check.
        directory (str): The path to the directory to search in.

    Returns:
        bool: True if the file exists in the directory, False otherwise.
    """
    file_path = join(directory, filename)
    return exists(file_path) and isfile(file_path)



def load_experiment_results(filename, save_path_dir="./results"):
    """
    Load experiment results from a .npz file in the specified directory.

    Args:
        filename (str): The name of the .npz file containing the experiment results.
        save_path_dir (str): The path to the directory containing the results file.
                            Defaults to "./results".

    Returns:
        dict: A dictionary containing the experiment results.
    """
    file_path = join(save_path_dir, filename)

    if not exists(file_path) or not isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist or is not a file.")

    data = np.load(file_path, allow_pickle=True)
    results = {key: data[key].item() for key in data}
    return results


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

def save_experiment_results(results, file_name, save_path_dir = "./results"):
    """
    Save the experiment results to a file using NumPy's .npz format.

    Args:
        results (dict): A dictionary containing the results of the experiment.
        save_path (str): The path to the file where the results will be saved.
    """
    # Convert the results dictionary to a format suitable for np.savez
    np_results = {}

    save_path = join(save_path_dir, file_name)
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            np_results[key] = value
        else:
            np_results[key] = np.array(value)

    # Save the results to the specified file
    np.savez(save_path, **np_results)

    print(f"Experiment results saved to {save_path}")


def parse_results_one_model(model, train_dataloader, test_dataloader, device):
    print("parse_results_one_model function")
    criterion = CrossEntropyLoss()

    # Compute accuracy and loss for the train dataset
    start_time = time.time()
    train_loss = compute_loss(model, criterion, train_dataloader, device)
    end_time = time.time()
    print(f"compute_loss time (first time, we are doing it twice): {end_time - start_time} seconds", end="||")


    test_loss = compute_loss(model, criterion, test_dataloader, device)

    start_time = time.time()
    accuracy_train = compute_loss(model, "accuracy", train_dataloader, device)
    end_time = time.time()
    print(f"compute_accuracy time (first time, we are doing it twice): {end_time - start_time} seconds", end="||")
    accuracy_test = compute_loss(model, "accuracy", test_dataloader, device)

    start_time = time.time()
    end_time = time.time()
    print(f"l2_distance_weight_space: {end_time - start_time} seconds", end="||")

    results = {
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test,
        'loss_train': train_loss,
        'loss_test': test_loss,
    }

    return results



def parse_results(model1, model2, train_dataloader, test_dataloader, init_params, device, sample_num = 10, verbose = True):
    """
    Parse the results of the models after training.

    Args:
        model1 (torch.nn.Module): The first trained model.
        model2 (torch.nn.Module): The second trained model.
        train_dataloader (DataLoader): The train data loader.
        test_dataloader (DataLoader): The test data loader.
        init_params (list): The initial parameters of the models before training.
        device (str): The device to use for the computations.

    Returns:
        dict: A dictionary containing the parsed results.
    """
    criterion = CrossEntropyLoss()
    model1_params = get_network_parameters(model1)
    model2_params = get_network_parameters(model2)


    if verbose: print(f"\rparse_results: l2 distances  in weight space. ===>", end=" ")

    l2_distance_between_models = l2_distance_weight_space(model1_params, model2_params)
    l2_distance_model1_init = l2_distance_weight_space(model1_params, init_params)
    l2_distance_model2_init = l2_distance_weight_space(model2_params, init_params)

    if verbose: print(f"\rparse_results: interpolated_losses. sample_num={sample_num}. Train.  ===>", end=" ")
    interpolated_losses_train, interpolated_accuracy_train, alphas = compute_interpolated_metrics(model1, model2, sample_num, criterion, train_dataloader, device)

    if verbose: print(f"\rparse_results: interpolated_losses. sample_num={sample_num}. Test.  ===>", end=" ")
    interpolated_losses_test, interpolated_accuracy_test, alphas = compute_interpolated_metrics(model1, model2, sample_num, criterion, test_dataloader, device)

    results = {
        "l2_distance_between_models": l2_distance_between_models,
        "l2_distance_model1_init": l2_distance_model1_init,
        "l2_distance_model2_init": l2_distance_model2_init,
        "alphas":alphas,
        "interpolated_losses_train": interpolated_losses_train,
        "interpolated_accuracy_train":interpolated_accuracy_train,
        "interpolated_losses_test":interpolated_losses_test,
        "interpolated_accuracy_test": interpolated_accuracy_test
    }

    return results
