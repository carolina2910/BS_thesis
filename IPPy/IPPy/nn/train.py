import time
import numpy as np

import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from ..utilities import _utilities, metrics




def train(
    model: nn.Module,
    training_data: Dataset,
    loss_fn=nn.Module,
    n_epochs: int = 100,
    batch_size: int = 12,
    save_each: int | None = None,
    weights_path: str | None = None,
    device: str = "cpu",
    graphic_path: str | None = None,
    loss_file:str | None = None,
) -> None:
    r"""
    Train a given pytorch model on an input training set. Note that if save_each is defined not to be None, then a weigths_path
    has to be given as input as well. Otherwise, this function does not save the resulting model weights, and the user should
    save them by himself.
    :param nn.Module model: The model to be trained.
    :param training_data: A pytorch training dataset. Has to be initialized by the function
    :param loss_fn: A pytorch loss function.
    :param int n_epochs: The number of epochs of the training process.
    :param int batch_size: Number of samples in each batch.
    :param int save_each: If given, saves a model checkpoint every X epochs, where X is the value of save_each.
    :param str weights_path: If save_each is given, represents the path on which saving
    :param str device: The device on which the operations are performed.
    :param str graphic_path: If given, represents the path where to save the training loss graphic.
    :param str loss_file: If given, represents the path where to save the training loss"""

    ### Initialize training
    # Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Define dataloader
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    print('train_loader length:', len(train_loader))
    # Verbose
    print(f"Training NN model for {n_epochs} epochs and batch size of {batch_size}.")

    # Cycle over the epochs
    loss_total = torch.zeros((n_epochs,))
    ssim_total = torch.zeros((n_epochs,))
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        ssim_loss = 0.0
        start_time = time.time()

        # Cycle over the batches
        for t, data in enumerate(train_loader):
            x, y = data

            # Debug: check shapes of x and y
            print(f"Batch {t+1}: x shape: {x.shape}, y shape: {y.shape}")

            # Send x and y to gpu
            x = x.to(device)
            y = y.to(device)

            # Debug: Check if data is on the correct device
            print(f"x device: {x.device}, y device: {y.device}")

            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x)

            # Debug: check shape of prediction
            print(f"y_pred shape: {y_pred.shape}")

            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            # Update loss
            epoch_loss += loss.item()

            # Measure time
            formatted_time = time.time() - start_time

            # Verbose
            print(
                f"({formatted_time:.2f} sec) Epoch ({epoch+1}/{n_epochs}) - Batch {t+1} -> "
                f"Loss = {epoch_loss / (t + 1):0.4f}"
            )

        loss_total[epoch] = epoch_loss / (t + 1)

        # Save the weights of the model
        if save_each is not None and (epoch + 1) % save_each == 0:
            torch.save(
                model.state_dict(),
                weights_path,
            )

    print(f"Training completed after {n_epochs} epochs.")

    plt.figure(figsize=(10, 6))
    plt.plot(range(n_epochs), loss_total.numpy(), label="Loss", color="blue", linewidth=2)
    plt.title("Loss during Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()


    if graphic_path:
        plt.savefig(graphic_path)
        print(f"Plot saved at {graphic_path}")
    else:
        plt.show()


    if loss_file:
        np.savetxt(loss_file, loss_total.numpy(), delimiter=",", header="Epoch,Loss", comments="")
        print(f"Loss values saved at {loss_file}")

