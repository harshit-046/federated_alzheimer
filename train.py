from dataset import (
    load_dataset,
    create_train_test_split
)

from partition import (
    dirichlet_partition,
    print_client_distribution
)

from centralized import (
    train_centralized
)

from siloed import (
    run_siloed_training
)

from fedavg import (
    run_fedavg
)

from utils import (
    seed_everything,
    get_device
)

from config import (
    DATASET_PATH
)

from fedprox import (
    run_fedprox
)

from fedavg_dp import (
    run_dp_fedavg
)


def main():

    # Setup

    seed_everything()

    device = get_device()

    print(
        f"\nUsing Device: {device}"
    )

    # Load Dataset

    dataset = load_dataset(
        DATASET_PATH
    )

    print(
        f"\nDataset Size: "
        f"{len(dataset)}"
    )

    # Train/Test Split

    train_indices, test_indices = (
        create_train_test_split(
            dataset
        )
    )

    print(
        f"Train Samples: "
        f"{len(train_indices)}"
    )

    print(
        f"Test Samples: "
        f"{len(test_indices)}"
    )

    # Create Train Wrapper

    train_labels = [

        dataset.targets[idx]

        for idx in train_indices
    ]

    class TrainWrapper:

        def __init__(
            self,
            targets
        ):

            self.targets = targets

        def __len__(self):

            return len(
                self.targets
            )

    wrapped_train = (
        TrainWrapper(
            train_labels
        )
    )

    # Non-IID Partition

    local_client_partitions = (
        dirichlet_partition(
            wrapped_train
        )
    )

    print_client_distribution(
        wrapped_train,
        local_client_partitions
    )

    # Convert Local Indices
    # To Original Dataset Indices

    client_partitions = []

    for local_partition in (
        local_client_partitions
    ):

        global_partition = [

            train_indices[idx]

            for idx in local_partition
        ]

        client_partitions.append(
            global_partition
        )

    # Centralized Baseline

    print(
        "CENTRALIZED TRAINING :"
    )

    centralized_results = (
        train_centralized(
            dataset=dataset,
            train_indices=train_indices,
            test_indices=test_indices,
            device=device
        )
    )

    # Siloed Baseline

    print(
        "SILOED TRAINING :"
    )

    siloed_results = (
        run_siloed_training(
            dataset=dataset,
            client_partitions=
            client_partitions,
            test_indices=
            test_indices,
            device=device
        )
    )

    # FedAvg

    print(
        "FEDAVG TRAINING :"
    )

    fedavg_results = (
        run_fedavg(
            dataset=dataset,
            test_indices=
            test_indices,
            client_partitions=
            client_partitions,
            device=device
        )
    )
    
    # FredProx

    print(
        "FEDPROX TRAINING :"
    )

    fedprox_results = (
        run_fedprox(
            dataset=dataset,
            test_indices=test_indices,
            client_partitions=client_partitions,
            device=device
        )
    )


    # DP-FedAvg

    print(
        "DP-FEDAVG TRAINING"
    )

    dp_results = (
        run_dp_fedavg(
            dataset=dataset,
            test_indices=test_indices,
            client_partitions=
            client_partitions,
            device=device
        )
    )

    # Final Summary

    print(
        "FINAL RESULTS :"
    )


    print(
        f"\nCentralized : "
        f"{centralized_results['best_accuracy']:.2f}%"
    )

    print(
        f"Siloed Mean : "
        f"{siloed_results['mean_accuracy']:.2f}%"
    )

    print(
        f"FedAvg : "
        f"{fedavg_results['best_accuracy']:.2f}%"
    )

    print(
        f"FedProx : "
        f"{fedprox_results['best_accuracy']:.2f}%"
    )

    print(
        f"DP-FedAvg : "
        f"{dp_results['best_accuracy']:.2f}%"
    )


if __name__ == "__main__":

    main()