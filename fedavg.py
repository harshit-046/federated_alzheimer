import random

import torch

from client import FederatedClient

from server import FederatedServer

from evaluation import (
    evaluate_global_model
)

from config import (
    NUM_ROUNDS,
    CLIENTS_PER_ROUND,
    CLIENT_DROPOUT_RATE
)


def select_clients(
    clients,
    clients_per_round
):

    return random.sample(
        clients,
        clients_per_round
    )


def apply_dropout(
    selected_clients
):
    """
    Simulate dropped clients.
    """

    active_clients = []

    for client in selected_clients:

        if random.random() > (
            CLIENT_DROPOUT_RATE
        ):

            active_clients.append(
                client
            )

    if len(active_clients) == 0:

        active_clients.append(
            random.choice(
                selected_clients
            )
        )

    return active_clients


def run_fedavg(
    dataset,
    test_indices,
    client_partitions,
    device
):

    print(
        "\nStarting FedAvg Training\n"
    )

    server = FederatedServer(
        device
    )

    clients = []

    for client_id, indices in enumerate(
        client_partitions
    ):

        clients.append(

            FederatedClient(
                client_id=client_id,
                dataset=dataset,
                indices=indices,
                device=device
            )
        )

    round_losses = []

    round_accuracies = []

    best_accuracy = 0

    # Federated Rounds

    for round_num in range(
        NUM_ROUNDS
    ):

        print(
            f"\nRound "
            f"{round_num+1}"
            f"/{NUM_ROUNDS}"
        )

        selected_clients = (
            select_clients(
                clients,
                CLIENTS_PER_ROUND
            )
        )

        selected_clients = (
            apply_dropout(
                selected_clients
            )
        )

        client_updates = []

        for client in (
            selected_clients
        ):

            update = client.train(
                server.get_global_model()
            )

            client_updates.append(
                update
            )

        server.aggregate(
            client_updates
        )

        avg_loss = (
            server.average_loss(
                client_updates
            )
        )

        metrics = (
            evaluate_global_model(
                model=server.get_global_model(),
                dataset=dataset,
                test_indices=test_indices,
                device=device
            )
        )

        round_losses.append(
            avg_loss
        )

        round_accuracies.append(
            metrics["accuracy"]
        )

        print(
            f"Loss: {avg_loss:.4f}"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.2f}%"
        )

        if (
            metrics["accuracy"]
            > best_accuracy
        ):

            best_accuracy = (
                metrics["accuracy"]
            )

            torch.save(
                server.get_global_model()
                .state_dict(),

                "best_fedavg_model.pth"
            )

    print(
        "\nFedAvg Training Completed"
    )

    print(
        f"Best Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    return {

        "model":
            server.get_global_model(),

        "losses":
            round_losses,

        "accuracies":
            round_accuracies,

        "best_accuracy":
            best_accuracy
    }