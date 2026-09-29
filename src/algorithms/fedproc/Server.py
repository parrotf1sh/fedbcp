import copy

from ..BaseServer import BaseServer
from .ClientTrainer import ClientTrainer
from .config import resolve_config
from .model import ModelWithFeatures, ModelWithProjection
from .utils import aggregate_prototypes

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs
    ):
        super().__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        self.cfg = resolve_config(algo_params)
        if not isinstance(model, (ModelWithFeatures, ModelWithProjection)):
            raise ValueError("Use the FedProc model adapter before creating the optimizer")
        if self.cfg["use_project_head"] != isinstance(model, ModelWithProjection):
            raise ValueError("Configure the FedProc projection head before creating the optimizer")
        self.global_prototypes = None
        self.server_velocity = None
        self.client = ClientTrainer(
            alpha_rounds=self.cfg["alpha_rounds"],
            optimizer_class=type(optimizer),
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )

    def _clients_training(self, sampled_clients):
        """Preserve initialization, local training, then prototype refresh order."""
        round_idx = len(self.server_results["client_history"]) - 1
        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()
        initial_weights = [server_weights for _ in sampled_clients]

        # First-round prototypes use the broadcast model on selected clients only.
        if self.global_prototypes is None:
            initial_weights = self._collect_prototypes(sampled_clients, initial_weights)

        updated_local_weights, client_sizes = [], []
        round_results = {}
        for client_idx, weights in zip(sampled_clients, initial_weights):
            self._set_client_data(client_idx)
            self.client.download_global(
                weights, server_optimizer, self.global_prototypes, round_idx
            )
            local_results, local_size = self.client.train()
            updated_local_weights.append(self.client.upload_local())
            client_sizes.append(local_size)
            round_results = self._results_updater(round_results, local_results)
            self.client.reset()

        # Revisit each trained local model before aggregating model parameters.
        updated_local_weights = self._collect_prototypes(
            sampled_clients, updated_local_weights
        )
        return updated_local_weights, client_sizes, round_results

    def _collect_prototypes(self, sampled_clients, local_weights):
        payloads, updated_weights = [], []
        for client_idx, weights in zip(sampled_clients, local_weights):
            self._set_client_data(client_idx)
            self.client.model.load_state_dict(weights)
            payloads.append(self.client.upload_prototypes())
            # Prototype extraction also updates buffers in models with tracked BN.
            updated_weights.append(self.client.upload_local())
            self.client.reset()

        self.global_prototypes = aggregate_prototypes(
            payloads, self.num_classes, self.global_prototypes
        )
        return updated_weights

    def _aggregation(self, w, ns):
        if self.cfg["aggregation"] == "sampled":
            weights = super()._aggregation(w, ns)
        else:
            # Literal upstream behavior: positional client IDs and all-client total.
            # This intentionally retains its partial-participation weight mismatch.
            sizes = [
                self.data_distributed["local"][client_idx]["datasize"]
                for client_idx in range(self.n_clients)
            ]
            total_size = sum(sizes)
            weights = copy.deepcopy(w[0])
            for key in weights:
                weights[key] = w[0][key] * (sizes[0] / total_size)
                for client_idx in range(1, len(w)):
                    weights[key] += w[client_idx][key] * (sizes[client_idx] / total_size)

        momentum = self.cfg["server_momentum"]
        if momentum:
            old_weights = self.model.state_dict()
            if self.server_velocity is None:
                self.server_velocity = {key: 0 for key in weights}
            for key in weights:
                delta = old_weights[key] - weights[key]
                self.server_velocity[key] = (
                    momentum * self.server_velocity[key] + (1 - momentum) * delta
                )
                weights[key] = old_weights[key] - self.server_velocity[key]

        return weights
