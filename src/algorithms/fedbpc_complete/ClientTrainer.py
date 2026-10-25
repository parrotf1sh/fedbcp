import copy

import torch

from ..BaseClientTrainer import BaseClientTrainer
from .config import resolve_config
from .criterion import PrototypeCompletionLoss, alignment_weights, completion_weights


class ClientTrainer(BaseClientTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = resolve_config(self.algo_params)
        self.criterion = PrototypeCompletionLoss()
        self.class_counts = None

    def download_global(self, server_weights, server_optimizer, bank, global_prior, round_idx):
        self.model.load_state_dict(server_weights)
        self.model.to(self.device)
        # Every logical client starts with a fresh optimizer; never reuse momentum.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        state = copy.deepcopy(server_optimizer)
        state["state"] = {}
        self.optimizer.load_state_dict(state)
        self.prototypes = bank.means.detach().to(self.device)
        self.variances = bank.variances.detach().to(self.device)
        reliability = bank.reliability.to(self.device)
        counts = torch.as_tensor(self.class_counts, dtype=torch.float32, device=self.device)
        active = round_idx >= self.cfg["warmup_rounds"]
        strength = self.cfg["completion_strength"] if active else 0.0
        self.real_weights, self.replay_weights = completion_weights(
            counts, global_prior.to(self.device), reliability, strength)
        self.align_weights = alignment_weights(counts, reliability)
        self.lambda_align = self.cfg["lambda_align"] if active else 0.0

    def train(self):
        if self.trainloader is None or self.datasize <= 0:
            raise ValueError("A nonempty local training loader is required")
        if self.class_counts is None or sum(self.class_counts) != self.datasize:
            raise ValueError("Client class counts must match its training sample count")
        self.model.train()
        totals = dict(loss_sum=0.0, correct=0, seen=0, real_ce_sum=0.0,
                      replay_ce_sum=0.0, align_sum=0.0, natural_ce_sum=0.0)
        for _ in range(self.local_epochs):
            for images, targets in self.trainloader:
                images, targets = images.to(self.device), targets.to(self.device)
                logits, features = self.model(images, get_features=True)
                loss, terms = self.criterion(
                    logits, targets, features, self.model.classifier, self.prototypes,
                    self.real_weights, self.replay_weights, self.align_weights, self.lambda_align,
                    self.variances, self.cfg["variance_scale"],
                    self.cfg["preserve_backbone_gradients"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite FedBPC-Complete loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                size = targets.numel()
                totals["seen"] += size
                totals["correct"] += (logits.detach().argmax(1) == targets).sum().item()
                totals["loss_sum"] += loss.detach().item() * size
                for key, value in terms.items():
                    totals[key + "_sum"] += value.detach().item() * size
        if not totals["seen"]:
            raise ValueError("No local training batches were processed")
        counts = torch.as_tensor(self.class_counts, device=self.device)
        totals["completion_mass"] = self.replay_weights.sum().item()
        totals["missing_completion_mass"] = self.replay_weights[counts == 0].sum().item()
        totals["minority_completion_mass"] = self.replay_weights[counts > 0].sum().item()
        totals["classification_mass"] = (
            (counts / counts.sum() * self.real_weights).sum()
            + self.replay_weights.sum()).item()
        return totals, self.datasize

    def reset(self):
        super().reset()
        self.class_counts = None
        self.prototypes = None
        self.variances = None
        self.real_weights = self.replay_weights = self.align_weights = None
