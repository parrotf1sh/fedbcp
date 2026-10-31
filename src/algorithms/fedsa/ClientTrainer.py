import copy

import torch

from ..BaseClientTrainer import BaseClientTrainer
from .criterion import SemanticAnchorLoss, global_gap, local_gap, prototype_report_loss
from .utils import (
    average_prototypes,
    classifier_module,
    cpu_snapshot,
    forward_with_features,
    preserve_evaluation_rng,
)

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.criterion = SemanticAnchorLoss(
            cfg["lambda_anchor"], cfg["lambda_mcl"], cfg["lambda_cc"]
        )
        self.semantic_anchors = None
        self.local_prototypes = None

    def download_global(self, local_weights, server_optimizer, semantic_anchors,
                        local_prototypes):
        """Restore local weights/history; only semantic anchors are broadcast."""
        self.model.load_state_dict(local_weights, strict=True)
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.optimizer.load_state_dict(copy.deepcopy(server_optimizer))
        self.semantic_anchors = semantic_anchors.detach().to(self.device)
        self.local_prototypes = (
            None if local_prototypes is None else
            {label: value.detach().to(self.device) for label, value in local_prototypes.items()}
        )

    def train(self):
        self.model.train()
        self.model.to(self.device)
        if self.trainloader is None or len(self.trainloader) == 0:
            raise ValueError("FedSA requires a nonempty client training loader")
        local_margin = local_gap(self.local_prototypes)
        global_margin = global_gap(self.semantic_anchors)
        classifier = classifier_module(self.model)
        prototypes, loss_results = {}, {}
        anchor_class_losses = []

        # Unlike the project's FedProto, this collection spans ALL local epochs.
        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                data, targets = data.to(self.device), targets.to(self.device)
                logits, features = forward_with_features(self.model, data)
                loss, losses, class_losses = self.criterion(
                    logits, features, targets, classifier, self.semantic_anchors,
                    local_margin, global_margin,
                )
                for index, label in enumerate(targets):
                    prototypes.setdefault(label.item(), []).append(features[index].detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                for key, value in losses.items():
                    loss_results.setdefault(key, []).append(value.item())
                anchor_class_losses.append(class_losses)

        if not loss_results:
            raise ValueError("FedSA client training produced no batches")
        self.local_prototypes = average_prototypes(prototypes)
        local_results = {key: sum(values) / len(values) for key, values in loss_results.items()}
        local_results["global_gap"] = float(global_margin)
        local_results["effective_gap"] = float(max(local_margin, global_margin))
        # Keep -inf in the actual computation, but not in JSON/SwanLab metrics.
        local_results["local_gap_is_negative_infinity"] = float(local_margin == float("-inf"))
        if local_margin != float("-inf"):
            local_results["local_gap"] = float(local_margin)
        for label, value in enumerate(torch.stack(anchor_class_losses).mean(dim=0)):
            local_results["anchor_class_ce_{:03d}".format(label)] = value.item()
        return local_results, self.datasize

    @torch.no_grad()
    def train_metrics(self, global_prototypes):
        """Optional source report; never used as a replacement training loss."""
        previous_mode = self.model.training
        losses, samples = 0.0, 0
        self.model.to(self.device)
        try:
            self.model.eval()
            with preserve_evaluation_rng([self.trainloader]):
                for data, targets in self.trainloader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    logits, features = forward_with_features(self.model, data)
                    loss = prototype_report_loss(
                        logits, features, targets, global_prototypes, self.cfg["lambda_anchor"]
                    )
                    losses += loss.item() * targets.numel()
                    samples += targets.numel()
        finally:
            self.model.train(previous_mode)
        return losses, samples

    def upload_local(self):
        return cpu_snapshot(self.model.state_dict())

    def upload_prototypes(self):
        return {label: value.detach().clone() for label, value in self.local_prototypes.items()}

    def reset(self):
        self.datasize = None
        self.trainloader = None
        self.testloader = None
        self.optimizer = None
        self.semantic_anchors = None
        self.local_prototypes = None
