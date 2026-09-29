import torch

from ..BaseClientTrainer import BaseClientTrainer
from .criterion import PrototypeContrastiveLoss

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, alpha_rounds, optimizer_class, **kwargs):
        super().__init__(**kwargs)
        self.alpha_rounds = alpha_rounds
        self.optimizer_class = optimizer_class
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=0)
        self.prototype_criterion = PrototypeContrastiveLoss()
        self.global_prototypes = None
        self.round_idx = None

    def train(self):
        """Train against the same global prototypes for the entire local run."""
        self.model.train()
        self.model.to(self.device)
        local_size = self.datasize

        if self.round_idx < self.alpha_rounds:
            alpha = self.round_idx / self.alpha_rounds
        else:
            alpha = 1

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()
                data, targets = data.to(self.device), targets.to(self.device)
                data.requires_grad = True
                targets.requires_grad = False
                targets = targets.long()

                logits, features = self.model(data, get_features=True)
                ce_loss = self.criterion(logits, targets)
                prototype_loss = self.prototype_criterion(
                    features, targets, self.global_prototypes
                )
                if self.round_idx == 0:
                    loss = ce_loss
                else:
                    loss = alpha * ce_loss + (1 - alpha) * prototype_loss

                loss.backward()
                self.optimizer.step()

        local_results = self._get_local_stats()
        return local_results, local_size

    def download_global(
        self, server_weights, server_optimizer, global_prototypes, round_idx
    ):
        super().download_global(server_weights, server_optimizer)
        self.global_prototypes = [
            prototype.detach().to(self.device) for prototype in global_prototypes
        ]
        self.round_idx = round_idx

    @torch.no_grad()
    def upload_prototypes(self):
        """Extract prototypes in training mode, as in get_global_class_center."""
        self.model.train()
        self.model.to(self.device)
        feature_sums, class_counts = {}, {}

        for data, targets in self.trainloader:
            data, targets = data.to(self.device), targets.to(self.device)
            _, features = self.model(data, get_features=True)
            for label in torch.unique(targets):
                feature = features[targets == label]
                class_idx = label.item()
                if class_idx not in feature_sums:
                    feature_sums[class_idx] = torch.sum(feature, dim=0)
                    class_counts[class_idx] = feature.shape[0]
                else:
                    feature_sums[class_idx] += torch.sum(feature, dim=0)
                    class_counts[class_idx] += feature.shape[0]

        return {"feature_sums": feature_sums, "class_counts": class_counts}

    def reset(self):
        super().reset()
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=0)
        self.global_prototypes = None
        self.round_idx = None
