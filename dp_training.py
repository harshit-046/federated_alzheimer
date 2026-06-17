# dp_training.py

import torch
import torch.nn as nn
import torch.optim as optim

from opacus import PrivacyEngine

from config import (
    LEARNING_RATE,
    LOCAL_EPOCHS,
    MAX_GRAD_NORM,
    NOISE_MULTIPLIER,
    DELTA
)


def train_with_dp(
    model,
    train_loader,
    device
):

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    privacy_engine = (
        PrivacyEngine()
    )

    model, optimizer, train_loader = (
        privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=
            NOISE_MULTIPLIER,
            max_grad_norm=
            MAX_GRAD_NORM
        )
    )

    model.train()

    total_loss = 0

    for _ in range(
        LOCAL_EPOCHS
    ):

        for images, labels in (
            train_loader
        ):

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

            total_loss += (
                loss.item()
            )

    epsilon = (
        privacy_engine
        .accountant
        .get_epsilon(
            delta=DELTA
        )
    )

    avg_loss = (
        total_loss
        /
        len(train_loader)
    )

    if hasattr(model, "_module"):

        weights = (
            model._module.state_dict()
        )

    else:

        weights = (
            model.state_dict()
        )

    return {

        "weights": weights,

        "loss": avg_loss,

        "epsilon": epsilon
    }