import copy
import inspect

import torch
import wandb

from ..BaseServer import BaseServer
from .ClientTrainer import ClientTrainer
from .config import resolve_config
from .utils import (
    get_relational_prototypes,
    get_distribution_distances,
    get_consistent_prototypes,
)

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs
    ):
        super().__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        if "get_features" not in inspect.signature(model.forward).parameters:
            raise NotImplementedError(
                "FedSC requires forward(..., get_features=True), as supported by "
                "fedavg_cifar, fedavg_tiny, and vgg11."
            )
        if type(optimizer) is not torch.optim.SGD:
            raise ValueError("The FedSC source uses SGD")
        if type(self.local_epochs) is not int or self.local_epochs < 1:
            raise ValueError("FedSC local_epochs must be a positive integer")
        self.cfg = resolve_config(algo_params)
        # These source placeholders are never read by the CE-only first round.
        self.relational_prototypes = {}
        self.consistent_prototypes = {}
        for label in range(self.num_classes):
            self.relational_prototypes[label] = torch.empty(
                self.num_classes, device=self.device, requires_grad=False
            )
            self.consistent_prototypes[label] = torch.empty(
                self.num_classes, device=self.device, requires_grad=False
            )
        self.client = ClientTrainer(
            cl_tau=self.cfg["cl_tau"],
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )

    def _clients_training(self, sampled_clients):
        round_idx = len(self.server_results["client_history"]) - 1
        if round_idx < 0:
            raise RuntimeError("Record sampled clients before FedSC local training")
        updated_local_weights, client_sizes = [], []
        local_prototypes, local_class_counts = [], []
        round_results = {}
        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        for client_idx in sampled_clients:
            self._set_client_data(client_idx)
            self.client.download_global(
                server_weights, server_optimizer, self.relational_prototypes,
                self.consistent_prototypes, round_idx,
            )
            local_results, local_size = self.client.train()
            updated_local_weights.append(self.client.upload_local())
            payload = self.client.upload_prototypes()
            local_prototypes.append(payload["prototypes"])
            local_class_counts.append(payload["class_counts"])
            client_sizes.append(local_size)
            round_results = self._results_updater(round_results, local_results)
            self.client.reset()

        # Refresh from this round only, before BaseServer aggregates the models.
        self.relational_prototypes = get_relational_prototypes(local_prototypes)
        distances = get_distribution_distances(local_class_counts, self.num_classes)
        self.consistent_prototypes = get_consistent_prototypes(
            self.relational_prototypes, distances, client_sizes, self.device
        )
        return updated_local_weights, client_sizes, round_results

    def _aggregation(self, w, ns):
        """Preserve source sum(weight * size) / total_size operation order."""
        if not w or sum(ns) == 0:
            raise ValueError("FedSC cannot aggregate empty client updates")
        weights = copy.deepcopy(w[0])
        for name in w[0]:
            values = []
            for local_weights, size in zip(w, ns):
                values.append(local_weights[name] * size)
            weights[name] = sum(values) / sum(ns)
        return weights

    def _wandb_logging(self, round_results, round_idx):
        super()._wandb_logging(round_results, round_idx)
        metrics = {
            "fedsc/" + name: sum(values) / len(values)
            for name, values in round_results.items()
            if name.endswith("_loss")
        }
        wandb.log(metrics, step=round_idx)
