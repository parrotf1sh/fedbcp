import os
import sys
import copy
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, moon_criterion, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
        self.moon_criterion = moon_criterion

    def train(self):
        """Local training"""

        # Keep global model and prev local model
        self._keep_global()
        self._keep_prev_local()

        self.model.train()
        self.model.to(self.device)

        local_size = self.datasize
        metrics_only = self.algo_params.get("metrics_only", False)
        totals = {"loss_sum": 0.0, "ce_loss_sum": 0.0, "contrastive_loss_sum": 0.0,
                  "weighted_contrastive_loss_sum": 0.0, "correct": 0, "seen": 0}

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                output, z = self.model(data, get_features=True)

                # for moon contrast
                with torch.no_grad():
                    _, z_prev = self.prev_model(data, get_features=True)
                    _, z_g = self.dg_model(data, get_features=True)

                result = self.moon_criterion(output, targets, z, z_prev, z_g,
                                             return_components=metrics_only)
                loss, components = result if metrics_only else (result, None)

                # backward pass
                loss.backward()
                self.optimizer.step()
                if metrics_only:
                    count = targets.numel()
                    totals["loss_sum"] += loss.detach().item() * count
                    totals["correct"] += (output.detach().argmax(1) == targets).sum().item()
                    totals["seen"] += count
                    for key, value in components.items():
                        totals[key + "_sum"] += value.item() * count

        if metrics_only and not totals["seen"]:
            raise ValueError("Selected MOON client has no training samples")
        local_results = totals if metrics_only else self._get_local_stats()

        return local_results, local_size

    def download_global(self, server_weights, server_optimizer, prev_weights):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)
        self.prev_weights = prev_weights

    def _keep_prev_local(self):
        """Keep distributed global model's weight"""
        self.prev_model = copy.deepcopy(self.model)
        self.prev_model.load_state_dict(self.prev_weights)
        self.prev_model.to(self.device)

        for params in self.prev_model.parameters():
            params.requires_grad = False
        self.prev_model.eval()

    def _keep_global(self):
        super()._keep_global()
        self.dg_model.eval()

    def reset(self):
        super().reset()
        # References are rebuilt for each client; don't retain old GPU copies.
        self.prev_weights = None
        self.prev_model = None
        self.dg_model = None
