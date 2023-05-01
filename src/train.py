import torch
import torch.nn as nn
import torch.optim as optim


def train_one_epoch(model, optimizer, criterion, dataloader, device, verbose = False):
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function.
        dataloader (torch.utils.data.DataLoader): The data loader for the training set.
        device (torch.device): The device to use for training (e.g., 'cpu' or 'cuda').
        
    Returns:
        float: The average training loss for this epoch.
    """
    model.train()
    running_loss = 0.0
    steps = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        steps += 1
        if verbose: print(f"\r running_loss={running_loss:.2f}, steps={steps}", end="")

    epoch_loss = running_loss / len(dataloader.dataset)
    return model, epoch_loss, steps

def evaluate(model, criterion, data_loader, device):
    """
    Evaluate the model on a validation or test set.
    
    Args:
        model (torch.nn.Module): The neural network model.
        criterion (torch.nn.Module): The loss function.
        data_loader (torch.utils.data.DataLoader): The data loader for the validation or test set.
        device (torch.device): The device to use for evaluation (e.g., 'cpu' or 'cuda').
        
    Returns:
        float: The average loss for this dataset.
        float: The accuracy for this dataset.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    return total_loss / len(data_loader), correct / total


def train_model(model, optimizer, criterion, train_loader, val_loader, device, num_epochs, patience = 5):
    """
    Train the model with early stopping.
    
    Args:
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        device (torch.device): The device to use for training and evaluation (e.g., 'cpu' or 'cuda').
        num_epochs (int): The maximum number of epochs to train the model.
        patience (int): The number of epochs to wait for validation loss improvement before stopping training.

    Returns:
        torch.nn.Module: The trained model.
    """
    if num_epochs == 0: return model, 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    total_steps = 0

    model.to(device)

    for epoch in range(num_epochs):
        model, train_loss, steps = train_one_epoch(model, optimizer, criterion, train_loader, device)
        total_steps += steps
        val_loss, val_accuracy = evaluate(model, criterion, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping triggered. Training stopped.")
            break

        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model, total_steps
