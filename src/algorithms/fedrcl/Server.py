import copy

import torch
import experiment_tracker as tracking

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

    def _tracking_logging(self, round_results, round_idx):
        super()._tracking_logging(round_results, round_idx)
        metrics = {
            "fedrcl/" + name: sum(values) / len(values)
            for name, values in round_results.items()
            if name.endswith("_loss")
        }
        tracking.log(metrics, step=round_idx)

    def batch_round_metrics(self, local):
        seen = sum(local["seen"])
        metrics = {"train/total_loss": sum(local["loss_sum"]) / seen}
        for name, values in local.items():
            if name.endswith("_loss_sum"):
                metrics["train/" + name[:-4]] = sum(values) / seen
        # cls is the source name; also expose the shared CE metric name.
        metrics["train/ce_loss"] = metrics["train/cls_loss"]
        initial_lr = self.optimizer.param_groups[0]["lr"]
        metrics.update({
            "fedrcl/local_first_epoch_lr": initial_lr,
            "fedrcl/local_last_epoch_lr": initial_lr * self.cfg["local_lr_decay"] ** (self.local_epochs - 1),
        })
        return metrics

    def batch_summary_metrics(self, metrics):
        result = {"final_" + name.replace("/", "_"): value
                  for name, value in metrics.items()
                  if name.startswith("train/") and name.endswith("_loss")}
        result.update({
            "aggregation": "equal_participant_full_state_mean",
            "fedrcl_implementation": "project_source_behavior_v1",
            "local_lr_decay": self.cfg["local_lr_decay"],
            "final_local_first_epoch_lr": metrics["fedrcl/local_first_epoch_lr"],
            "final_local_last_epoch_lr": metrics["fedrcl/local_last_epoch_lr"],
            "use_amp_requested": self.cfg["use_amp"],
            "autocast_enabled": torch.device(self.device).type == "cuda" and self.cfg["use_amp"],
            "grad_scaler_enabled": torch.device(self.device).type == "cuda",
            "rcl_branch_levels": self.cfg["rcl_loss"]["branch_level"],
            "model_parameter_count": sum(p.numel() for p in self.model.parameters()),
            "loss_component_definition": "sample-weighted raw losses before objective weights; cls is CE",
            "communication_definition": (
                "model upload + model download + optimizer tensor download per participant; "
                "reference model is copied locally from the downloaded model; "
                "layer features stay local; excludes transport overhead and CPU/GPU transfers"),
        })
        return result
