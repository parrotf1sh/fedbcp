import torch

from ..BaseClientTrainer import BaseClientTrainer
from .criterion import PrototypeContrastiveLoss
from .config import loss_weights

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

        ce_weight, prototype_weight = loss_weights(self.round_idx, self.alpha_rounds)
        metrics_only = self.algo_params.get("metrics_only", False)
        totals = {"loss_sum": 0.0, "ce_loss_sum": 0.0, "prototype_loss_sum": 0.0,
                  "weighted_ce_loss_sum": 0.0, "weighted_prototype_loss_sum": 0.0,
                  "seen": 0, "correct": 0}

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
                    loss = ce_weight * ce_loss + prototype_weight * prototype_loss

                if metrics_only and not all(torch.isfinite(value).item()
                                            for value in (ce_loss, prototype_loss, loss)):
                    raise FloatingPointError("Non-finite FedProc loss in round {}".format(self.round_idx + 1))

                loss.backward()
                self.optimizer.step()

                if metrics_only:
                    count = targets.numel()
                    ce_value, proto_value = ce_loss.detach().item(), prototype_loss.detach().item()
                    totals["loss_sum"] += loss.detach().item() * count
                    totals["ce_loss_sum"] += ce_value * count
                    totals["prototype_loss_sum"] += proto_value * count
                    totals["weighted_ce_loss_sum"] += ce_weight * ce_value * count
                    totals["weighted_prototype_loss_sum"] += prototype_weight * proto_value * count
                    totals["seen"] += count
                    totals["correct"] += (logits.detach().argmax(1) == targets).sum().item()

        if metrics_only and not totals["seen"]:
            raise ValueError("Selected FedProc client has no training samples")
        local_results = totals if metrics_only else self._get_local_stats()
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
