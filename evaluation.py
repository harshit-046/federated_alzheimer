import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    Subset
)

from config import (
    BATCH_SIZE
)


def evaluate_model(
    model,
    dataset,
    indices,
    device
):
    """
    Evaluate model on subset.
    """

    subset = Subset(
        dataset,
        indices
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    model.eval()

    total_loss = 0

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
            )

            predictions = (
                outputs.argmax(1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = (
        100.0 * correct / total
    )

    avg_loss = (
        total_loss /
        len(loader)
    )

    return (
        avg_loss,
        accuracy
    )


def evaluate_global_model(
    model,
    dataset,
    test_indices,
    device
):
    """
    Global model evaluation.
    """

    loss, accuracy = (
        evaluate_model(
            model,
            dataset,
            test_indices,
            device
        )
    )

    return {
        "loss": loss,
        "accuracy": accuracy
    }