from dp_model import DPCNN


class DPServer:

    def __init__(self, device):

        self.device = device

        self.global_model = DPCNN()

        self.global_model.to(device)

    def get_global_model(self):

        return self.global_model

    def set_global_weights(
        self,
        weights
    ):

        self.global_model.load_state_dict(
            weights
        )

    def aggregate(
        self,
        client_updates
    ):

        total_samples = sum(
            update["num_samples"]
            for update in client_updates
        )

        global_weights = {

            k: v.clone()

            for k, v in

            client_updates[0][
                "weights"
            ].items()
        }

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

            for key in global_weights:

                global_weights[key] += (
                    update["weights"][key]
                    * local_weight
                )

        self.set_global_weights(
            global_weights
        )