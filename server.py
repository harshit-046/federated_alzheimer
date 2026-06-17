import copy

from model import get_model


class FederatedServer:

    def __init__(
        self,
        device
    ):

        self.device = device

        self.global_model = (
            get_model(
                pretrained=True
            )
        )

        self.global_model.to(
            self.device
        )

    # Get Global Model

    def get_global_model(self):

        return self.global_model

    # Get Global Weights

    def get_global_weights(self):

        return copy.deepcopy(
            self.global_model.state_dict()
        )

    # Set Global Weights

    def set_global_weights(
        self,
        weights
    ):

        self.global_model.load_state_dict(
            weights
        )

    # FedAvg Aggregation

    def aggregate(
        self,
        client_updates
    ):
        """
        client_updates:

        [
            {
                "weights": ...,
                "loss": ...,
                "num_samples": ...
            }
        ]
        """

        total_samples = sum(

            update["num_samples"]

            for update in client_updates
        )

        global_weights = copy.deepcopy(

            client_updates[0][
                "weights"
            ]
        )

        first_weight = (
            client_updates[0][
                "num_samples"
            ]
            / total_samples
        )

        for key in global_weights:

            global_weights[key] *= (
                first_weight
            )

        for update in (
            client_updates[1:]
        ):

            local_weight = (
                update["num_samples"]
                / total_samples
            )

            local_weights = (
                update["weights"]
            )

            for key in (
                global_weights
            ):

                global_weights[key] += (
                    local_weights[key]
                    * local_weight
                )

        self.set_global_weights(
            global_weights
        )

    # Average Client Loss

    def average_loss(
        self,
        client_updates
    ):

        losses = [

            update["loss"]

            for update in client_updates
        ]

        return (
            sum(losses)
            / len(losses)
        )