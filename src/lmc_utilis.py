# ishapira, April 2023
#  This script contains utility functions for working with PyTorch neural networks,
#  including interpolation, parameter extraction, distance calculation, and performance evaluation.
import torch
import numpy as np
import torch.jit
import time

def get_network_parameters(model):
    """
    Get the parameters of a PyTorch network.
    
    Args:
        model (torch.nn.Module): The input neural network.
    
    Returns:
        A list of parameter tensors.
    """
    params = []
    for param in model.parameters():
        params.append(param.clone())  # .detach()?
    return params

def compute_metrics_models(model1, model2, dataloader, sample_num, device):
    criterion = nn.CrossEntropyLoss()
    return compute_interpolated_metrics(model1, model2, sample_num, criterion, dataloader, device, verbose=True)

def set_network_parameters(model, params):
    """
    Set the parameters of a PyTorch network.
    
    Args:
        model (torch.nn.Module): The input neural network.
        params (list): A list of parameter tensors.
    
    Returns:
        None
    """
    for model_param, input_param in zip(model.parameters(), params):
        model_param.data.copy_(input_param)

def interpolate_networks(model1, model2, alpha, device):
    """
    Interpolate the parameters of two networks with the same architecture.

    Args:
        model1 (torch.nn.Module): The first input neural network.
        model2 (torch.nn.Module): The second input neural network.
        alpha (float): The interpolation factor (between 0 and 1).

    Returns:
        A new network with the same architecture and interpolated parameters.
    """
    assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"
    
    net1_params = get_network_parameters(model1)
    net2_params = get_network_parameters(model2)
    
    assert type(model1) == type(model2), "Networks must have the same architecture"
    
    # Create a new network with the same architecture
    interpolated_net = type(model1)().to(device)
    interpolated_params = get_network_parameters(interpolated_net)

    # Interpolate the parameters
    for p1, p2, p_interpolated in zip(net1_params, net2_params, interpolated_params):
        #print(f"interpolate_networks, p1 = {p1}")
        #print(f"interpolate_networks, alpha * p1 + (1 - alpha) * p2 = {alpha * p1 + (1 - alpha) * p2}")
        p_interpolated.data.copy_(alpha * p1 + (1 - alpha) * p2)

    set_network_parameters(interpolated_net, interpolated_params)
    return interpolated_net


def is_network_on_device(network, device):
    """
    Check if the network's parameters are on the given device.

    Args:
        network (torch.nn.Module): The input neural network.
        device (str or torch.device): The target device.


    Returns:
        bool: True if the network is on the given device, False otherwise.
    """
    first_param_device = next(network.parameters()).device
    return first_param_device == device


def compute_loss(network, criterion, dataloader, device):
    print("compute_loss")

    start_time = time.time()
    if not is_network_on_device(network, device):
        network = network.to(device)
    end_time = time.time()
    print(f"is_network_on_device: {end_time - start_time} seconds", end="||")

    network.eval()
    total_loss = 0.0
    total_samples = 0

    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = network(inputs)

            if criterion == "accuracy":
                _, predicted = torch.max(outputs.data, 1)
                total_samples += inputs.size(0)
                total_loss += (predicted == targets).sum().item()
            else:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
    end_time = time.time()
    print(f"actual loss computing: {end_time - start_time} seconds", end="||")
    return total_loss / total_samples






def compute_interpolated_metrics(model1, model2, sample_num, criterion, dataloader, device, verbose=True):
    """
    Compute the interpolated metrics (losses and accuracies) for interpolated models.

    Args:
        model1 (torch.nn.Module): The first trained model.
        model2 (torch.nn.Module): The second trained model.
        sample_num (int): The number of samples to interpolate between the two models.
        criterion (nn.Module): The loss function to use for evaluating the models.
        dataloader (DataLoader): The data loader to use for evaluating the models.
        device (str or torch.device): The device to use for the computations.

    Returns:
        tuple: A tuple containing interpolated losses and accuracies, and alphas.
    """


    alphas = torch.linspace(0, 1, sample_num)
    losses = []
    accuracy_lst = []

    for alpha in alphas:
        interpolated_net = interpolate_networks(model1, model2, alpha.item(), device)
        loss, accuracy = compute_loss_and_accuracy(interpolated_net, criterion, dataloader, device)
        losses.append(loss)
        accuracy_lst.append(accuracy)

    return losses, accuracy_lst, alphas

def compute_loss_and_accuracy(network, criterion, dataloader, device):
    if not is_network_on_device(network, device): network.to(device)
    network.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_accuracy += (predicted == targets).sum().item()

            total_samples += inputs.size(0)

    return total_loss / total_samples, total_accuracy / total_samples



def l2_distance_weight_space(params1, params2):
    """
    Compute the L2 distance between two sets of network parameters.

    Args:
        params1 (list): List of parameter tensors for the first network.
        params2 (list): List of parameter tensors for the second network.

    Returns:
        float: The L2 distance between the network parameter sets.
    """
    try:
        params1_device = params1[0].device

        # Move both parameter sets to the same device
        params1 = [param.to(params1_device) for param in params1]
        params2 = [param.to(params1_device) for param in params2]

        total_distance = 0.0
        for param1, param2 in zip(params1, params2):
            diff = param1 - param2
            total_distance += torch.sum(diff ** 2).item()
        return total_distance ** 0.5

    except:
        return np.nan



def get_model_predictions(model, dataloader, device):
    """
    Get predictions from a model given a dataloader.


    Args:
        model (nn.Module): The model to make predictions with.
        dataloader (DataLoader): The dataloader with input data.
        device (str): The device to run the model on, e.g. 'cpu' or 'cuda'.
    
    Returns:
        numpy array: The model's predictions.
    """
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.detach().cpu().numpy())

    return np.array(predictions)

