import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from ..BaseClientTrainer import BaseClientTrainer
from .criterion import CLLoss

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.dg_model = None
        self.local_scheduler = None
        # Retain source insertion order even if pair definitions are reordered.
        self.rcl_criterions = {"scl": None, "penalty": None}
        for pair in cfg["rcl_loss"]["pairs"]:
            self.rcl_criterions[pair["name"]] = CLLoss(pair=pair, **cfg["rcl_loss"])

    def download_global(self, server_weights, server_optimizer):
        self.model.load_state_dict(server_weights)
        # Every selected client starts a fresh SGD, as in RCLClient.setup.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.optimizer.load_state_dict(server_optimizer)
        self.optimizer.state.clear()
        for group in self.optimizer.param_groups:
            # The global scheduler's initial_lr is not the current round's lr.
            group["initial_lr"] = group["lr"]
        self.local_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: self.cfg["local_lr_decay"] ** epoch
        )

    def get_weights(self):
        weights = {
            "cls": self.cfg["ce_loss"]["weight"],
            "cossim": self.cfg["feature_align_loss"]["weight"],
        }
        if self.cfg["prox_loss"]:
            weights["prox"] = self.cfg["prox_loss"]["weight"]
        for pair in self.cfg["rcl_loss"]["pairs"]:
            weights[pair["name"]] = pair["weight"]
        return weights

    def _algorithm_rcl(self, local_results, global_results, targets):
        losses = {"cossim": []}
        rcl = self.cfg["rcl_loss"]
        for layer_idx in range(self.model.num_layers):
            train_layer = rcl["branch_level"] is False or layer_idx in rcl["branch_level"]
            local_feature = local_results["layer{}".format(layer_idx)]
            global_feature = global_results["layer{}".format(layer_idx)]
            if local_feature.dim() == 4:
                local_feature = F.adaptive_avg_pool2d(local_feature, 1)
                global_feature = F.adaptive_avg_pool2d(global_feature, 1)

            if self.cfg["feature_align_loss"]["align_type"] == "l2":
                loss_align = F.mse_loss(
                    local_feature.squeeze(-1).squeeze(-1),
                    global_feature.squeeze(-1).squeeze(-1),
                )
            else:
                loss_align = F.cosine_embedding_loss(
                    local_feature.squeeze(-1).squeeze(-1),
                    global_feature.squeeze(-1).squeeze(-1), torch.ones_like(targets),
                )
            losses["cossim"].append(loss_align)

            if train_layer:
                for name, criterion in self.rcl_criterions.items():
                    if criterion is not None:
                        # Preserve the source's ordered train_layer updates between pairs.
                        if criterion.pair.get("branch_level"):
                            train_layer = layer_idx in criterion.pair["branch_level"]
                        if train_layer:
                            loss_rcl = criterion(
                                old_feat=global_feature, new_feat=local_feature,
                                target=targets, reduction=False, topk_neg=rcl["topk_neg"],
                            )
                            losses.setdefault(name, []).append(loss_rcl.mean())

        return {
            name: torch.mean(torch.stack(values)) if len(values) > 0 else 0
            for name, values in losses.items()
        }

    def _algorithm(self, data, targets):
        no_relu = not self.cfg["rcl_loss"]["feature_relu"]
        results = self.model(data, get_layer_features=True, no_relu=no_relu)
        with torch.no_grad():
            global_results = self.dg_model(data, get_layer_features=True, no_relu=no_relu)
        losses = {"cls": self.criterion(results["logit"], targets)}

        prox_loss = 0
        fixed_params = dict(self.dg_model.named_parameters())
        for name, parameter in self.model.named_parameters():
            prox_loss += ((parameter - fixed_params[name].detach()) ** 2).sum()
        losses["prox"] = prox_loss
        losses.update(self._algorithm_rcl(results, global_results, targets))
        return losses, results["logit"]

    def train(self):
        if self.local_scheduler is None:
            raise RuntimeError("Download the global model before FedRCL training")
        self.model.to(self.device)
        self.model.train()
        self._keep_global()
        # Freezing parameters does not imply eval mode in the source.
        self.dg_model.train()
        use_amp = torch.device(self.device).type == "cuda" and self.cfg["use_amp"]
        # Source scaling is enabled on CUDA even when autocast is disabled.
        scaler = GradScaler(enabled=torch.device(self.device).type == "cuda")
        weights = self.get_weights()
        totals = {"loss_sum": 0.0, "correct": 0, "seen": 0}
        loss_names = list(dict.fromkeys(["cls", "prox", "cossim"] + list(self.rcl_criterions)))
        for name in loss_names:
            totals[name + "_loss_sum"] = 0.0
        local_size = self.datasize

        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                data, targets = data.to(self.device), targets.to(self.device)
                self.model.zero_grad()
                with autocast(enabled=use_amp):
                    losses, logits = self._algorithm(data, targets)
                    for name in losses:
                        weights.setdefault(name, 0)
                    loss = sum(weights[name] * value for name, value in losses.items())

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                scaler.step(self.optimizer)
                scaler.update()

                count = targets.numel()
                totals["loss_sum"] += loss.detach().item() * count
                totals["correct"] += (logits.detach().argmax(1) == targets).sum().item()
                totals["seen"] += count
                for name, value in losses.items():
                    value = value.detach().item() if torch.is_tensor(value) else value
                    totals[name + "_loss_sum"] += value * count
            self.local_scheduler.step()

        if not totals["seen"]:
            raise ValueError("Selected FedRCL client has no training batches")
        if self.cfg["metrics_only"]:
            local_results = totals
        else:
            local_results = self._get_local_stats()
            local_results["total_loss"] = totals["loss_sum"] / totals["seen"]
            for name in loss_names:
                local_results[name + "_loss"] = totals[name + "_loss_sum"] / totals["seen"]
        self.model.to("cpu")
        self.dg_model.to("cpu")
        return local_results, local_size

    def reset(self):
        super().reset()
        self.dg_model = None
        self.local_scheduler = None
