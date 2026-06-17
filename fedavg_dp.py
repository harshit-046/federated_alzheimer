# fedavg_dp.py

import random

from server import (
    FederatedServer
)

from dp_client import (
    DPClient
)

from evaluation import (
    evaluate_global_model
)

from config import (
    NUM_ROUNDS,
    CLIENTS_PER_ROUND
)


def run_dp_fedavg(
    dataset,
    test_indices,
    client_partitions,
    device
):

    print(
        "\nStarting DP-FedAvg\n"
    )

    server = (
        FederatedServer(
            device
        )
    )

    clients = []

    for client_id, indices in enumerate(
        client_partitions
    ):

        clients.append(

            DPClient(
                client_id,
                dataset,
                indices,
                device
            )
        )

    best_accuracy = 0

    epsilon_history = []

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

        avg_epsilon = (

            sum(
                update["epsilon"]
                for update in updates
            )

            /

            len(updates)
        )

        epsilon_history.append(
            avg_epsilon
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.2f}%"
        )

        print(
            f"Epsilon: "
            f"{avg_epsilon:.2f}"
        )

        if (
            metrics["accuracy"]
            > best_accuracy
        ):

            best_accuracy = (
                metrics["accuracy"]
            )

    print(
        "\nDP-FedAvg Completed"
    )

    print(
        f"Best Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    return {

        "best_accuracy":
            best_accuracy,

        "epsilon_history":
            epsilon_history
    }