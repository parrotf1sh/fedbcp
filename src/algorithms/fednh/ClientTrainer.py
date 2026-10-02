import copy

import torch
import torch.nn.functional as F

from ..BaseClientTrainer import BaseClientTrainer
from .config import resolve_config
from .utils import client_learning_rate

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, optimizer_class, initial_lr, n_rounds, **kwargs):
        super().__init__(**kwargs)
        self.cfg = resolve_config(self.algo_params)
        self.optimizer_class = optimizer_class
        self.initial_lr = initial_lr
        self.n_rounds = n_rounds
        self.class_counts = None
        self.prototype_payload = None
        self.round_idx = None

    def download_global(self, server_weights, server_optimizer, round_idx):
        self.model.load_state_dict(server_weights)
        self.model.to(self.device)
        self.round_idx = round_idx
        self.optimizer = self.optimizer_class(
            filter(lambda parameter: parameter.requires_grad, self.model.parameters()), lr=0
        )
        # The upstream optimizer is recreated for every local training call.
        optimizer_state = copy.deepcopy(server_optimizer)
        optimizer_state["state"] = {}
        self.optimizer.load_state_dict(optimizer_state)
        for group in self.optimizer.param_groups:
            group["lr"] = client_learning_rate(
                self.cfg, self.initial_lr, round_idx, self.n_rounds, group["lr"]
            )

    def train(self):
        if self.trainloader is None or self.datasize is None or self.datasize <= 0:
            raise ValueError("FedNH requires a nonempty training loader")
        if self.class_counts is None or sum(self.class_counts) != self.datasize:
            raise ValueError("FedNH class counts must match the client's training dataset")
        self.model.train()
        self.model.to(self.device)
        self.criterion.to(self.device)
        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                data, targets = data.to(self.device), targets.to(self.device)
                logits = self.model(data)
                loss = self.criterion(logits, targets)
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda parameter: parameter.requires_grad, self.model.parameters()),
                    max_norm=self.cfg["max_grad_norm"],
                )
                self.optimizer.step()

        if self.cfg["client_adv_prototype_agg"]:
            self.prototype_payload = self._estimate_prototype_adv()
        else:
            self.prototype_payload = self._estimate_prototype()
        local_results = self._get_local_stats()
        return local_results, self.datasize

    def _count_by_class_full(self):
        return torch.tensor(
            [count if count > 0 else 1e-12 for count in self.class_counts],
            device=self.device,
        )

    @torch.no_grad()
    def _estimate_prototype(self):
        self.model.eval()
        prototype = torch.zeros_like(self.model.prototype)
        for data, targets in self.trainloader:
            data, targets = data.to(self.device), targets.to(self.device)
            _, features = self.model(data, get_features=True)
            for class_idx in torch.unique(targets).cpu().tolist():
                prototype[class_idx] += torch.sum(features[targets == class_idx], dim=0)
        for class_idx, count in enumerate(self.class_counts):
            if count > 0:
                prototype[class_idx] /= count
                prototype_norm = torch.norm(prototype[class_idx]).clamp(min=1e-12)
                prototype[class_idx] = torch.div(prototype[class_idx], prototype_norm)
                prototype[class_idx] *= count
        return {"scaled_prototype": prototype,
                "count_by_class_full": self._count_by_class_full()}

    @torch.no_grad()
    def _estimate_prototype_adv(self):
        self.model.eval()
        embeddings, labels, weights = [], [], []
        prototype = torch.zeros_like(self.model.prototype)
        for data, targets in self.trainloader:
            data, targets = data.to(self.device), targets.to(self.device)
            logits, features = self.model(data, get_features=True)
            probabilities = F.softmax(logits, dim=1)
            weights.append(torch.gather(probabilities, dim=1, index=targets.view(-1, 1)))
            embeddings.append(features)
            labels.append(targets)
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        weights = torch.cat(weights, dim=0).view(-1, 1)
        for class_idx, count in enumerate(self.class_counts):
            if count > 0:
                mask = labels == class_idx
                class_weights = weights[mask]
                class_features = embeddings[mask]
                # Keep the original denominator; do not introduce confidence filtering.
                prototype[class_idx] = torch.sum(
                    class_features * class_weights, dim=0
                ) / torch.sum(class_weights)
                prototype_norm = torch.norm(prototype[class_idx]).clamp(min=1e-12)
                prototype[class_idx] = torch.div(prototype[class_idx], prototype_norm)
        return {"adv_agg_prototype": prototype,
                "count_by_class_full": self._count_by_class_full()}

    def upload_prototypes(self):
        return copy.deepcopy(self.prototype_payload)

    def reset(self):
        super().reset()
        self.class_counts = None
        self.prototype_payload = None
        self.round_idx = None
