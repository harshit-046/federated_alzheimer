import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from torch.utils.data import (
    DataLoader,
    Subset
)

from config import (
    BATCH_SIZE,
    LOCAL_EPOCHS,
    LEARNING_RATE
)

from model import get_model


def evaluate(
    model,
    dataloader,
    device
):

    model.eval()

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            predictions = (
                outputs.argmax(1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        100.0 * correct / total
    )


def train_centralized(
    dataset,
    train_indices,
    test_indices,
    device
):
    """
    Centralized baseline.
    """

    train_dataset = Subset(
        dataset,
        train_indices
    )

    test_dataset = Subset(
        dataset,
        test_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = get_model(
        pretrained=True
    )

    model.to(device)

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_acc = 0

    for epoch in range(
        LOCAL_EPOCHS
    ):

        model.train()

        correct = 0

        total = 0

        running_loss = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}"
        )

        for images, labels in pbar:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
            )

            predictions = (
                outputs.argmax(1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += (
                labels.size(0)
            )

        train_acc = (
            100.0 * correct / total
        )

        test_acc = evaluate(
            model,
            test_loader,
            device
        )

        print(
            f"\nEpoch {epoch+1}"
        )

        print(
            f"Train Accuracy:"
            f" {train_acc:.2f}%"
        )

        print(
            f"Test Accuracy:"
            f" {test_acc:.2f}%"
        )

        if test_acc > best_acc:

            best_acc = test_acc

            torch.save(
                model.state_dict(),
                "best_centralized_model.pth"
            )

    print(
        f"\nBest Accuracy:"
        f" {best_acc:.2f}%"
    )

    return {
        "model": model,
        "best_accuracy": best_acc
    }