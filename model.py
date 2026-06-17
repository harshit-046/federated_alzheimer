# model.py

import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from config import NUM_CLASSES

# --------------------------------------------------
# Opacus is optional
# Project should run even if opacus is not installed
# --------------------------------------------------

try:

    from opacus.validators import (
        ModuleValidator
    )

    OPACUS_AVAILABLE = True

except ImportError:

    ModuleValidator = None

    OPACUS_AVAILABLE = False


def get_model(pretrained=True):
    """
    ResNet18 backbone.

    Uses ImageNet pretrained weights.

    Final layer modified for Alzheimer's
    4-class classification.

    If Opacus is installed:
        BatchNorm -> GroupNorm
    automatically for DP-SGD compatibility.
    """

    if pretrained:

        weights = (
            ResNet18_Weights.IMAGENET1K_V1
        )

    else:

        weights = None

    model = resnet18(
        weights=weights
    )

    in_features = (
        model.fc.in_features
    )

    model.fc = nn.Sequential(

        nn.Dropout(0.3),

        nn.Linear(
            in_features,
            256
        ),

        nn.ReLU(),

        nn.Dropout(0.2),

        nn.Linear(
            256,
            NUM_CLASSES
        )
    )

    # ------------------------------------
    # Opacus Compatibility
    # ------------------------------------

    if OPACUS_AVAILABLE:

        model = ModuleValidator.fix(
            model
        )

    return model


def count_parameters(model):
    """
    Count trainable parameters.
    """

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad
    )


def get_model_parameters(model):
    """
    Convert model weights to numpy arrays.

    Used by Flower communication.
    """

    return [

        param.detach()
        .cpu()
        .numpy()

        for param in model.state_dict().values()
    ]


def set_model_parameters(
    model,
    parameters
):
    """
    Load global weights received
    from server.
    """

    params_dict = zip(

        model.state_dict().keys(),

        parameters
    )

    state_dict = {

        key: torch.tensor(value)

        for key, value in params_dict
    }

    model.load_state_dict(
        state_dict,
        strict=True
    )


def freeze_backbone(model):
    """
    Freeze all layers except
    classifier head.

    Used during personalization.
    """

    for name, param in model.named_parameters():

        if "fc" not in name:

            param.requires_grad = False


def unfreeze_all(model):
    """
    Unfreeze complete model.
    """

    for param in model.parameters():

        param.requires_grad = True


def is_opacus_ready(model):
    """
    Check whether model is valid
    for DP-SGD.
    """

    if not OPACUS_AVAILABLE:

        return False

    return ModuleValidator.is_valid(
        model
    )