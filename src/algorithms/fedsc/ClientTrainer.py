import copy
import time
import math

import torch

from ..BaseClientTrainer import BaseClientTrainer
from .criterion import SemanticCollaborationLoss
from .utils import average_prototypes
from .config import resolve_config

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, cl_tau, **kwargs):
        super().__init__(**kwargs)
        self.metrics_only = resolve_config(self.algo_params)["metrics_only"]
        self.criterion = SemanticCollaborationLoss(cl_tau, collect_metrics=self.metrics_only)
        self.relational_prototypes = None
        self.consistent_prototypes = None
        self.prototype_payload = None
        self.round_idx = None

    def download_global(
        self, server_weights, server_optimizer, relational_prototypes,
        consistent_prototypes, round_idx,
    ):
        self.model.load_state_dict(server_weights)
        self.model.to(self.device)
        # Source constructs a fresh Local (and SGD) for every participation.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.optimizer.load_state_dict(server_optimizer)
        self.optimizer.state.clear()
        self.optimizer.zero_grad()
        self.relational_prototypes = copy.deepcopy(relational_prototypes)
        self.consistent_prototypes = copy.deepcopy(consistent_prototypes)
        self.prototype_payload = None
        self.round_idx = round_idx

    def train(self):
        if self.round_idx is None:
            raise RuntimeError("Download the global model before FedSC training")
        self.model.train()
        self.model.to(self.device)
        local_size = self.datasize
        local_prototypes = {}
        loss_results = {
            "ce_loss": [], "rpcl_loss": [], "cpdr_loss": [], "total_loss": [],
        }

        totals = {"seen": 0, "correct": 0, "loss_sum": 0.0,
                  "fallback_count": 0, "lookup_count": 0, "negative_pool_sum": 0,
                  "negative_pool_max": 0, "batches": 0}
        totals.update({name + "_sum": 0.0 for name in loss_results})
        for epoch_idx in range(self.local_epochs):
            class_counts = [0 for _ in range(self.num_classes)]
            for data, targets in self.trainloader:
                data, targets = data.to(self.device), targets.to(self.device)
                # Project return order differs from the source's (features, logits).
                logits, features = self.model(data, get_features=True)
                for sample_idx in range(len(targets)):
                    class_counts[targets[sample_idx].item()] += 1
                loss, loss_dict = self.criterion(
                    logits, features, targets, self.relational_prototypes,
                    self.consistent_prototypes, self.round_idx,
                )
                if self.metrics_only and not math.isfinite(loss.detach().item()):
                    raise FloatingPointError("Non-finite FedSC training loss")
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Use this training forward's features, not a post-training pass.
                if epoch_idx == self.local_epochs - 1:
                    for sample_idx in range(len(targets)):
                        label = targets[sample_idx].item()
                        if label in local_prototypes:
                            local_prototypes[label].append(features[sample_idx, :])
                        else:
                            local_prototypes[label] = [features[sample_idx, :]]
                if self.metrics_only:
                    count = targets.numel()
                    totals["seen"] += count
                    totals["correct"] += (logits.detach().argmax(1) == targets).sum().item()
                    totals["loss_sum"] += loss.detach().item() * count
                    for name, value in loss_dict.items():
                        totals[name + "_sum"] += value.detach().item() * count
                    diagnostics = self.criterion.last_metrics
                    for name in ("fallback_count", "lookup_count"):
                        totals[name] += diagnostics[name]
                    totals["negative_pool_sum"] += diagnostics["negative_pool_size"]
                    totals["negative_pool_max"] = max(totals["negative_pool_max"], diagnostics["negative_pool_size"])
                    totals["batches"] += 1
                else:
                    for name, value in loss_dict.items():
                        loss_results[name].append(value.detach().item())
            if sum(class_counts) == 0:
                raise ValueError("Selected FedSC client has no training batches")

        if self.metrics_only:
            from ..fedavg.reporting import synchronize
            synchronize(self.device)
            prototype_started = time.perf_counter()
        self.prototype_payload = {
            "prototypes": average_prototypes(local_prototypes),
            "class_counts": class_counts,
        }
        if self.metrics_only:
            synchronize(self.device)
            totals["prototype_seconds"] = time.perf_counter() - prototype_started
            totals["prototype_samples"] = sum(class_counts)
            return totals, local_size
        # Standard project evaluation must not recompute uploaded prototypes.
        local_results = self._get_local_stats()
        for name, values in loss_results.items():
            local_results[name] = sum(values) / len(values)
        return local_results, local_size

    def upload_prototypes(self):
        return copy.deepcopy(self.prototype_payload)

    def reset(self):
        super().reset()
        self.relational_prototypes = None
        self.consistent_prototypes = None
        self.prototype_payload = None
        self.round_idx = None
