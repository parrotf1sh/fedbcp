import copy

from ..BaseServer import BaseServer
from .ClientTrainer import ClientTrainer
from .config import resolve_config
from .model import ModelWithNormalizedHead
from .utils import aggregate_weights

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs
    ):
        super().__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        self.cfg = resolve_config(algo_params)
        if not isinstance(model, ModelWithNormalizedHead):
            raise ValueError("Use the FedNH model adapter before creating the optimizer")
        if model.cfg != self.cfg:
            raise ValueError("FedNH model and server settings must match")
        if self.cfg["client_lr_scheduler"] != "project" and scheduler is not None:
            raise ValueError("Disable the project scheduler when using a source FedNH schedule")
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if [id(parameter) for parameter in parameters] != [id(parameter) for parameter in optimizer_parameters]:
            raise ValueError("FedNH optimizer must contain the adapter's trainable parameters in order")
        if len(optimizer.param_groups) != 1:
            raise ValueError("FedNH expects the source optimizer's single parameter group")
        if model.prototype.shape[0] != self.num_classes:
            raise ValueError("FedNH prototype class count does not match the dataset")

        self.server_weights = copy.deepcopy(model.state_dict())
        self.exclude_layer_keys = {
            key for key in self.server_weights
            if any(excluded in key for excluded in self.cfg["exclude"])
        }
        self.initial_private_weights = {
            key: value.detach().cpu().clone() for key, value in self.server_weights.items()
            if key in self.exclude_layer_keys
        }
        self.client_private_weights = {}
        self.prototype_payloads = []
        self.client = ClientTrainer(
            optimizer_class=type(optimizer),
            initial_lr=optimizer.param_groups[0]["lr"],
            n_rounds=self.n_rounds,
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )

    def _set_client_data(self, client_idx):
        super()._set_client_data(client_idx)
        # Both project data pipelines already provide exact per-client counts.
        self.client.class_counts = list(self.data_distributed["data_map"][client_idx])
        if len(self.client.class_counts) != self.num_classes:
            raise ValueError("FedNH requires a count for every dataset class")

    def _clients_training(self, sampled_clients):
        round_idx = len(self.server_results["client_history"]) - 1
        self.server_weights = {
            key: value.to(self.device) for key, value in self.server_weights.items()
        }
        server_optimizer = self.optimizer.state_dict()
        updated_local_weights, client_sizes = [], []
        round_results = {}
        self.prototype_payloads = []
        for client_idx in sampled_clients:
            self._set_client_data(client_idx)
            weights = dict(self.server_weights)
            # Excluded tensors are private to each logical client, not the reused trainer.
            weights.update(self.client_private_weights.get(client_idx, self.initial_private_weights))
            self.client.download_global(weights, server_optimizer, round_idx)
            local_results, local_size = self.client.train()
            local_weights = self.client.upload_local()
            updated_local_weights.append(local_weights)
            self.prototype_payloads.append(self.client.upload_prototypes())
            if self.exclude_layer_keys:
                self.client_private_weights[client_idx] = {
                    key: local_weights[key].detach().cpu().clone()
                    for key in self.exclude_layer_keys
                }
            client_sizes.append(local_size)
            round_results = self._results_updater(round_results, local_results)
            self.client.reset()
        return updated_local_weights, client_sizes, round_results

    def _aggregation(self, w, ns):
        # FedNH weights participants equally, irrespective of ns.
        round_idx = len(self.server_results["client_history"]) - 1
        self.server_weights = aggregate_weights(
            self.server_weights, w, self.prototype_payloads,
            self.cfg, round_idx, self.exclude_layer_keys,
        )
        self.prototype_payloads = []
        return copy.deepcopy(self.server_weights)

    def _update_and_evaluate(self, ag_weights, round_results, round_idx, start_time):
        # Upstream's server-side evaluation client also skips excluded tensors
        # on download, even if the server updated an excluded prototype via EMA.
        evaluation_weights = dict(ag_weights)
        for key in self.exclude_layer_keys:
            evaluation_weights[key] = self.model.state_dict()[key]
        return super()._update_and_evaluate(
            evaluation_weights, round_results, round_idx, start_time
        )
