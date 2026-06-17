# partition.py

import random
import numpy as np

from collections import Counter

from config import (
    NUM_CLIENTS,
    DIRICHLET_ALPHA,
    RANDOM_SEED
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def get_labels(dataset):

    return np.array(
        dataset.targets
    )


def dirichlet_partition(
    dataset,
    num_clients=NUM_CLIENTS,
    alpha=DIRICHLET_ALPHA
):
    """
    Non-IID partition using
    Dirichlet distribution.
    """

    labels = get_labels(
        dataset
    )

    num_classes = len(
        np.unique(labels)
    )

    client_indices = [
        [] for _ in range(
            num_clients
        )
    ]

    for class_id in range(
        num_classes
    ):

        class_indices = np.where(
            labels == class_id
        )[0]

        np.random.shuffle(
            class_indices
        )

        proportions = (
            np.random.dirichlet(
                [alpha] * num_clients
            )
        )

        split_sizes = (
            proportions *
            len(class_indices)
        ).astype(int)

        split_sizes[-1] += (
            len(class_indices)
            - split_sizes.sum()
        )

        start = 0

        for client_id in range(
            num_clients
        ):

            end = (
                start +
                split_sizes[
                    client_id
                ]
            )

            client_indices[
                client_id
            ].extend(

                class_indices[
                    start:end
                ]
            )

            start = end

    for client_id in range(
        num_clients
    ):

        random.shuffle(
            client_indices[
                client_id
            ]
        )

    return client_indices


def iid_partition(
    dataset,
    num_clients=NUM_CLIENTS
):

    indices = np.arange(
        len(dataset.targets)
    )

    np.random.shuffle(
        indices
    )

    split_size = (
        len(indices)
        // num_clients
    )

    client_indices = []

    for client_id in range(
        num_clients
    ):

        start = (
            client_id *
            split_size
        )

        if (
            client_id ==
            num_clients - 1
        ):

            end = len(indices)

        else:

            end = (
                client_id + 1
            ) * split_size

        client_indices.append(

            indices[
                start:end
            ].tolist()
        )

    return client_indices


def print_client_distribution(
    dataset,
    client_indices
):

    labels = get_labels(
        dataset
    )

    print(
        "\nClient Distribution\n"
    )

    num_classes = len(
        np.unique(labels)
    )

    for client_id, indices in enumerate(
        client_indices
    ):

        client_labels = labels[
            indices
        ]

        counts = Counter(
            client_labels
        )

        print(
            f"Client {client_id}"
        )

        for cls in range(
            num_classes
        ):

            print(
                f"  Class {cls}:",
                counts.get(
                    cls,
                    0
                )
            )

        print(
            f"  Total: "
            f"{len(indices)}"
        )

        print("-" * 30)