import copy

import torch

from ..BaseClientTrainer import BaseClientTrainer
from .criterion import SemanticCollaborationLoss
from .utils import average_prototypes

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, cl_tau, **kwargs):
        super().__init__(**kwargs)
        self.criterion = SemanticCollaborationLoss(cl_tau)
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
                for name, value in loss_dict.items():
                    loss_results[name].append(value.detach().item())
            if sum(class_counts) == 0:
                raise ValueError("Selected FedSC client has no training batches")

        self.prototype_payload = {
            "prototypes": average_prototypes(local_prototypes),
            "class_counts": class_counts,
        }
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
