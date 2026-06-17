import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    DataLoader,
    Subset
)

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOCAL_EPOCHS
)


class FederatedClient:

    def __init__(
        self,
        client_id,
        dataset,
        indices,
        device
    ):

        self.client_id = client_id

        self.device = device

        self.indices = indices

        self.num_samples = len(
            indices
        )

        self.local_dataset = Subset(
            dataset,
            indices
        )

        self.loader = DataLoader(
            self.local_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

    # Clone global model

    def get_local_model(
        self,
        global_model
    ):

        local_model = copy.deepcopy(
            global_model
        )

        local_model.to(
            self.device
        )

        return local_model

    # Standard Local Training

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

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()

                optimizer.step()

                total_loss += (
                    loss.item()
                )

        avg_loss = (
            total_loss /
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

    # Local Evaluation

    def evaluate(
        self,
        model
    ):

        model.eval()

        criterion = (
            nn.CrossEntropyLoss()
        )

        total_loss = 0

        correct = 0

        total = 0

        with torch.no_grad():

            for images, labels in self.loader:

                images = images.to(
                    self.device
                )

                labels = labels.to(
                    self.device
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

                total += (
                    labels.size(0)
                )

        accuracy = (
            100.0 * correct / total
        )

        avg_loss = (
            total_loss /
            len(self.loader)
        )

        return {
            "loss":
                avg_loss,

            "accuracy":
                accuracy
        }