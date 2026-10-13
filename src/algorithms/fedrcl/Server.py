import copy

import torch
import wandb

from ..BaseServer import BaseServer
from .ClientTrainer import ClientTrainer
from .config import resolve_config
from .model import ModelWithLayerFeatures

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs
    ):
        super().__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        if not isinstance(model, ModelWithLayerFeatures):
            raise ValueError("Use the FedRCL model adapter before creating the optimizer")
        if type(optimizer) is not torch.optim.SGD:
            raise ValueError("The FedRCL source uses SGD")
        self.cfg = resolve_config(algo_params, model.num_layers)
        if type(self.local_epochs) is not int or self.local_epochs < 1:
            raise ValueError("FedRCL local_epochs must be a positive integer")
        self.client = ClientTrainer(
            cfg=self.cfg,
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )

    def run(self):
        if self.cfg["metrics_only"]:
            if not getattr(self, "experiment_config", {}).get("batch_protocol"):
                raise ValueError("FedRCL metrics_only requires the existing batch_protocol")
            from ..fedavg.reporting import run_metrics
            return run_metrics(self)
        return super().run()

    def _aggregation(self, w, ns):
        """Source aggregation: equal weight for each participating client."""
        if not w:
            raise ValueError("FedRCL cannot aggregate an empty client list")
        return {
            key: sum(weights[key] for weights in w) / len(w)
            for key in w[0]
        }

    def _wandb_logging(self, round_results, round_idx):
        super()._wandb_logging(round_results, round_idx)
        metrics = {
            "fedrcl/" + name: sum(values) / len(values)
            for name, values in round_results.items()
            if name.endswith("_loss")
        }
        wandb.log(metrics, step=round_idx)

    def batch_round_metrics(self, local):
        seen = sum(local["seen"])
        metrics = {"train/total_loss": sum(local["loss_sum"]) / seen}
        for name, values in local.items():
            if name.endswith("_loss_sum"):
                metrics["train/" + name[:-4]] = sum(values) / seen
        return metrics
