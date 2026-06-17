import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    DataLoader,
    Subset
)

from model import get_model

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOCAL_EPOCHS
)


def evaluate_model(
    model,
    dataloader,
    device
):

    model.eval()

    correct = 0

    total = 0

    loss_sum = 0

    criterion = (
        nn.CrossEntropyLoss()
    )

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

            loss = criterion(
                outputs,
                labels
            )

            loss_sum += loss.item()

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
        loss_sum /
        len(dataloader)
    )

    return avg_loss, accuracy


def train_single_client(
    dataset,
    client_indices,
    test_indices,
    device
):

    train_subset = Subset(
        dataset,
        client_indices
    )

    test_subset = Subset(
        dataset,
        test_indices
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_subset,
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

    for _ in range(
        LOCAL_EPOCHS
    ):

        model.train()

        for images, labels in train_loader:

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

    _, accuracy = (
        evaluate_model(
            model,
            test_loader,
            device
        )
    )

    return accuracy


def run_siloed_training(
    dataset,
    client_partitions,
    test_indices,
    device
):

    print(
        "\nRunning Siloed Training\n"
    )

    client_results = []

    for client_id, indices in enumerate(
        client_partitions
    ):

        accuracy = (
            train_single_client(
                dataset=dataset,
                client_indices=indices,
                test_indices=test_indices,
                device=device
            )
        )

        client_results.append(
            accuracy
        )

        print(
            f"Client {client_id} "
            f"Accuracy: "
            f"{accuracy:.2f}%"
        )

    mean_accuracy = (

        sum(client_results)
        / len(client_results)

    )

    print(
        f"\nMean Siloed Accuracy:"
        f" {mean_accuracy:.2f}%"
    )

    return {

        "client_accuracies":
            client_results,

        "mean_accuracy":
            mean_accuracy
    }