import os
import random

import numpy as np

import torch

import matplotlib.pyplot as plt

from config import RANDOM_SEED


def seed_everything(
    seed=RANDOM_SEED
):
    """
    Reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Select GPU if available.
    """

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


def save_model(
    model,
    path
):
    """
    Save model weights.
    """

    torch.save(
        model.state_dict(),
        path
    )


def load_model(
    model,
    path,
    device
):
    """
    Load saved weights.
    """

    model.load_state_dict(

        torch.load(
            path,
            map_location=device
        )
    )

    return model


def create_directory(
    directory
):
    """
    Create output folder.
    """

    os.makedirs(
        directory,
        exist_ok=True
    )


def plot_accuracy(
    rounds,
    accuracies,
    title,
    save_path
):
    """
    Accuracy vs rounds.
    """

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        rounds,
        accuracies,
        marker="o"
    )

    plt.xlabel(
        "Rounds"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        title
    )

    plt.grid(True)

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()


def plot_loss(
    rounds,
    losses,
    title,
    save_path
):
    """
    Loss vs rounds.
    """

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        rounds,
        losses,
        marker="o"
    )

    plt.xlabel(
        "Rounds"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        title
    )

    plt.grid(True)

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()