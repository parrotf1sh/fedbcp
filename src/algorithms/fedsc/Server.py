import copy
import inspect
import time

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


def prototype_message_bytes(prototypes):
    """Logical payload: tensors + int64 class IDs and (for lists) lengths."""
    from ..fedavg.reporting import tensor_bytes
    return tensor_bytes(prototypes) + 8 * len(prototypes) + sum(
        8 for value in prototypes.values() if isinstance(value, (list, tuple))
    )


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
        self.prototype_total_bytes = 0
        self.prototype_total_seconds = 0.0
        self.total_fallbacks = 0
        self.total_lookups = 0
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

    def run(self):
        if self.cfg["metrics_only"]:
            from ..fedavg.reporting import run_metrics
            return run_metrics(self)
        return super().run()

    def _clients_training(self, sampled_clients):
        round_idx = len(self.server_results["client_history"]) - 1
        if round_idx < 0:
            raise RuntimeError("Record sampled clients before FedSC local training")
        updated_local_weights, client_sizes = [], []
        local_prototypes, local_class_counts = [], []
        round_results = {}
        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()
        if self.cfg["metrics_only"]:
            # Round-zero placeholders are copied, but their VALUES are never read.
            self.round_download_bytes = len(sampled_clients) * (
                prototype_message_bytes(self.relational_prototypes)
                + prototype_message_bytes(self.consistent_prototypes)
            )
            self.round_upload_bytes = 0

        for client_idx in sampled_clients:
            self._set_client_data(client_idx)
            self.client.download_global(
                server_weights, server_optimizer, self.relational_prototypes,
                self.consistent_prototypes, round_idx,
            )
            local_results, local_size = self.client.train()
            updated_local_weights.append(self.client.upload_local())
            payload = self.client.upload_prototypes()
            if self.cfg["metrics_only"]:
                self.round_upload_bytes += prototype_message_bytes(payload["prototypes"])
                self.round_upload_bytes += 8 * len(payload["class_counts"])
            local_prototypes.append(payload["prototypes"])
            local_class_counts.append(payload["class_counts"])
            client_sizes.append(local_size)
            round_results = self._results_updater(round_results, local_results)
            self.client.reset()

        # Refresh from this round only, before BaseServer aggregates the models.
        if self.cfg["metrics_only"]:
            from ..fedavg.reporting import synchronize
            synchronize(self.device)
            started = time.perf_counter()
        self.relational_prototypes = get_relational_prototypes(local_prototypes)
        distances = get_distribution_distances(local_class_counts, self.num_classes)
        self.consistent_prototypes = get_consistent_prototypes(
            self.relational_prototypes, distances, client_sizes, self.device
        )
        if self.cfg["metrics_only"]:
            synchronize(self.device)
            self.round_server_prototype_seconds = time.perf_counter() - started
            self.prototype_total_bytes += self.batch_communication_bytes()
            self.prototype_total_seconds += self.round_server_prototype_seconds + sum(round_results["prototype_seconds"])
            self.total_fallbacks += sum(round_results["fallback_count"])
            self.total_lookups += sum(round_results["lookup_count"])
        return updated_local_weights, client_sizes, round_results

    def batch_communication_bytes(self):
        return self.round_download_bytes + self.round_upload_bytes

    def batch_round_metrics(self, local):
        seen = sum(local["seen"])
        metrics = {"train/" + name: sum(local[name + "_sum"]) / seen
                   for name in ("ce_loss", "rpcl_loss", "cpdr_loss", "total_loss")}
        metrics.update({
            "fedsc/fallback_count": sum(local["fallback_count"]),
            "fedsc/lookup_count": sum(local["lookup_count"]),
            "fedsc/fallback_rate": sum(local["fallback_count"]) / max(1, sum(local["lookup_count"])),
            "fedsc/negative_pool_mean_per_batch": sum(local["negative_pool_sum"]) / sum(local["batches"]),
            "fedsc/negative_pool_max": max(local["negative_pool_max"]),
            "fedsc/prototype_covered_classes": len(self.relational_prototypes),
            "fedsc/prototype_missing_classes": self.num_classes - len(self.relational_prototypes),
            "fedsc/relational_prototype_count": sum(len(values) for values in self.relational_prototypes.values()),
            "fedsc/consistent_prototype_count": len(self.consistent_prototypes),
            "fedsc/prototype_samples": sum(local["prototype_samples"]),
            "fedsc/prototype_extra_forward_passes": 0,
            "fedsc/local_prototype_reduction_seconds": sum(local["prototype_seconds"]),
            "fedsc/server_prototype_seconds": self.round_server_prototype_seconds,
            "fedsc/prototype_upload_bytes": self.round_upload_bytes,
            "fedsc/prototype_download_bytes": self.round_download_bytes,
            "fedsc/prototype_cumulative_bytes": self.prototype_total_bytes,
        })
        return metrics

    def batch_summary_metrics(self, metrics):
        result = {"final_train_" + name: metrics["train/" + name]
                  for name in ("ce_loss", "rpcl_loss", "cpdr_loss", "total_loss")}
        result.update({
            "fedsc_implementation": "provided_source_behavior_v1",
            "cl_tau": self.cfg["cl_tau"],
            "fallback_count": self.total_fallbacks,
            "lookup_count": self.total_lookups,
            "fallback_rate": self.total_fallbacks / max(1, self.total_lookups),
            "prototype_communication_bytes": self.prototype_total_bytes,
            "prototype_reduction_and_server_seconds": self.prototype_total_seconds,
            "model_parameter_count": sum(p.numel() for p in self.model.parameters()),
            "communication_definition": (
                "model upload + model download + optimizer tensor download per participant; "
                "add local prototype/count upload and relational/consistent prototype download; "
                "prototype class IDs, list lengths and dense class counts encoded as int64; "
                "round-zero copied placeholders counted by size only; excludes transport overhead"),
        })
        return result

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
