import copy

import torch
import torch.nn as nn
import torch.optim as optim

from client import FederatedClient
from server import FederatedServer

from evaluation import (
    evaluate_global_model
)

from config import (
    NUM_ROUNDS,
    CLIENTS_PER_ROUND,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    CLIENT_DROPOUT_RATE,
    MU
)


class FedProxClient(
    FederatedClient
):

    def train(
        self,
        global_model
    ):

        model = self.get_local_model(
            global_model
        )

        criterion = (
            nn.CrossEntropyLoss()
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE
        )

        model.train()

        total_loss = 0

        global_params = {

            name: param.detach().clone()

            for name, param in
            global_model.named_parameters()
        }

        for _ in range(
            LOCAL_EPOCHS
        ):

            for images, labels in self.loader:

                images = images.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                optimizer.zero_grad()

                outputs = model(
                    images
                )

                ce_loss = criterion(
                    outputs,
                    labels
                )

                prox_term = 0

                for name, param in (
                    model.named_parameters()
                ):

                    prox_term += (
                        (
                            param
                            -
                            global_params[
                                name
                            ]
                        ).norm(2) ** 2
                    )

                loss = (
                    ce_loss
                    +
                    (MU / 2)
                    * prox_term
                )

                loss.backward()

                optimizer.step()

                total_loss += (
                    loss.item()
                )

        avg_loss = (
            total_loss
            /
            len(self.loader)
        )

        return {

            "weights":
                copy.deepcopy(
                    model.state_dict()
                ),

            "loss":
                avg_loss,

            "num_samples":
                self.num_samples
        }


def apply_dropout(
    selected_clients
):

    import random

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
            selected_clients[0]
        )

    return active_clients


def run_fedprox(
    dataset,
    test_indices,
    client_partitions,
    device
):

    print(
        "\nStarting FedProx Training\n"
    )

    server = FederatedServer(
        device
    )

    clients = []

    for client_id, indices in enumerate(
        client_partitions
    ):

        clients.append(

            FedProxClient(
                client_id=client_id,
                dataset=dataset,
                indices=indices,
                device=device
            )
        )

    best_accuracy = 0

    losses = []

    accuracies = []

    import random

    for round_num in range(
        NUM_ROUNDS
    ):

        print(
            f"\nRound "
            f"{round_num+1}"
            f"/{NUM_ROUNDS}"
        )

        selected_clients = (
            random.sample(
                clients,
                CLIENTS_PER_ROUND
            )
        )

        selected_clients = (
            apply_dropout(
                selected_clients
            )
        )

        updates = []

        for client in (
            selected_clients
        ):

            updates.append(

                client.train(
                    server.get_global_model()
                )
            )

        server.aggregate(
            updates
        )

        avg_loss = (
            server.average_loss(
                updates
            )
        )

        metrics = (
            evaluate_global_model(
                model=
                server.get_global_model(),

                dataset=
                dataset,

                test_indices=
                test_indices,

                device=
                device
            )
        )

        losses.append(
            avg_loss
        )

        accuracies.append(
            metrics[
                "accuracy"
            ]
        )

        print(
            f"Loss: "
            f"{avg_loss:.4f}"
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

                server
                .get_global_model()
                .state_dict(),

                "best_fedprox_model.pth"
            )

    print(
        "\nFedProx Training Completed"
    )

    print(
        f"Best Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    return {

        "model":
            server.get_global_model(),

        "losses":
            losses,

        "accuracies":
            accuracies,

        "best_accuracy":
            best_accuracy
    }