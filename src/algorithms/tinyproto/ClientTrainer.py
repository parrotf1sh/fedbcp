import torch

from ..BaseClientTrainer import BaseClientTrainer
from .criterion import PrototypeLoss
from .utils import average_prototypes, cpu_snapshot, forward_with_features

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, cfg, mask_indices, learning_rate, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.mask_indices = {label: indices.to(self.device)
                             for label, indices in mask_indices.items()}
        self.learning_rate = learning_rate
        self.criterion = PrototypeLoss(cfg["lambda_proto"])
        self.global_prototypes = {}
        self.local_prototypes = {}
        self.class_counts = None

    def download_global(self, local_weights, global_prototypes, class_counts):
        """Restore this client's model; the only shared learning state is prototypes."""
        self.model.load_state_dict(local_weights, strict=True)
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.global_prototypes = {label: value.detach().to(self.device)
                                  for label, value in global_prototypes.items()}
        self.class_counts = class_counts

    def train(self):
        self.model.train()
        self.model.to(self.device)
        if self.trainloader is None or len(self.trainloader) == 0:
            raise ValueError("TinyProto requires a nonempty client training loader")
        prototypes = {}
        losses = {key: [] for key in ("total_loss", "ce_loss", "proto_loss")}
        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                data, targets = data.to(self.device), targets.to(self.device)
                logits, features = forward_with_features(self.model, data)
                loss, results = self.criterion(
                    logits, targets, features, self.global_prototypes, self.mask_indices
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite TinyProto local training loss")
                # Source accumulation spans ALL local epochs, before each update.
                for index, label in enumerate(targets):
                    prototypes.setdefault(label.item(), []).append(features[index].detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                last_batch_acc = logits.argmax(dim=1).eq(targets).float().mean().item()
                for key, value in results.items():
                    losses[key].append(value.item())
        if not losses["total_loss"]:
            raise ValueError("TinyProto client training produced no batches")
        self.local_prototypes = average_prototypes(
            prototypes, self.class_counts, self.mask_indices, self.cfg
        )
        local_results = {key: sum(values) / len(values) for key, values in losses.items()}
        local_results["last_batch_acc"] = last_batch_acc
        return local_results, self.datasize

    def upload_local(self):
        return cpu_snapshot(self.model.state_dict())

    def upload_prototypes(self):
        return {label: value.detach().clone() for label, value in self.local_prototypes.items()}

    def reset(self):
        self.datasize = None
        self.trainloader = None
        self.testloader = None
        self.optimizer = None
        self.class_counts = None
        self.global_prototypes = {}
        self.local_prototypes = {}
