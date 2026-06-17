# dp_client.py

import copy

from client import (
    FederatedClient
)

from dp_training import (
    train_with_dp
)


class DPClient(
    FederatedClient
):

    def train(
        self,
        global_model
    ):

        model = self.get_local_model(
            global_model
        )

        results = (
            train_with_dp(
                model=model,
                train_loader=self.loader,
                device=self.device
            )
        )

        return {

            "weights":
                copy.deepcopy(
                    results["weights"]
                ),

            "loss":
                results["loss"],

            "epsilon":
                results["epsilon"],

            "num_samples":
                self.num_samples
        }