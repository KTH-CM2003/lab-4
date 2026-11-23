import torch
import torch.nn as nn

def compute_loss(loss_fn, output, target):
    """
    Compute the loss for a batch of segmentation predictions.

    Args:
        loss_fn (callable): Loss function, e.g. torch.nn.CrossEntropyLoss() or a MONAI loss such as
            monai.losses.DiceLoss(to_onehot_y=True, softmax=True).
        output (torch.Tensor): Model output (logits) of shape (B, C, H, W).
        target (torch.Tensor): Label map of shape (B, 1, H, W) with the class index of each pixel.

    Returns:
        torch.Tensor: The loss.
    """
    if isinstance(loss_fn, nn.CrossEntropyLoss):
        # nn.CrossEntropyLoss expects class indices of shape (B, H, W) as integers
        return loss_fn(output, target.squeeze(1).long())
    # MONAI losses take the label map with its channel dimension, (B, 1, H, W)
    return loss_fn(output, target)

def run_batch(model, data, target, loss_fn, optimizer):
    """
    Run a single training batch.

    Args:
        model (nn.Module): The neural network model.
        data (torch.Tensor): Input data for the batch.
        target (torch.Tensor): Target labels for the batch.
        loss_fn (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.

    Returns:
        float: The loss value for this batch.
    """
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = compute_loss(loss_fn, output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        loss_fn (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to run the training on (CPU or GPU).

    Returns:
        float: Average loss for the epoch.
    """
    model.to(device)
    model.train()
    total_loss = 0
    for batch in dataloader:
        data, target = batch["image"].to(device), batch["label"].to(device)
        loss = run_batch(model, data, target, loss_fn, optimizer)
        total_loss += loss
    return total_loss / len(dataloader)

def train(model, train_loader, loss_fn, optimizer, device, epochs):
    """
    Train the model for a specified number of epochs.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training data.
        loss_fn (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to run the training on (CPU or GPU).
        epochs (int): Number of epochs to train for.
    """
    for epoch in range(epochs):
        epoch_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    return model

