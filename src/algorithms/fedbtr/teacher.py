"""A separate balanced readout trained on frozen global features.

Each client evaluates one full local mean gradient at the SAME initial head.
The uploaded update is combined with inverse-participation weighting. Thus,
before optional clipping, its expectation is the full-federation weighted
gradient under uniform sampling without replacement. No per-class aggregation
or per-client label distribution is required at the server.
"""

import copy

import torch
import torch.nn.functional as F

from .criterion import global_class_weights
from .utils import classifier_module, tensor_bytes


class BalancedReadoutTeacher:
    def __init__(self, global_model, global_counts, cfg, device):
        self.model = copy.deepcopy(global_model).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.head_name, frozen_head = classifier_module(self.model)
        self.head = copy.deepcopy(frozen_head).to(device)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)
        self.cfg = cfg
        self.device = device
        self.weights = global_class_weights(global_counts, cfg["teacher_balance_power"]).to(device)

    def calibrate(self, clients, total_clients, total_samples):
        if not clients:
            raise ValueError("Teacher calibration requires participating clients")
        if self.cfg["teacher_mode"] != "balanced":
            return dict(teacher_gradient_norm=0.0, teacher_examples=0,
                        teacher_uplink_bytes=0, teacher_downlink_bytes=0)
        probability = len(clients) / total_clients
        accumulated = {name: torch.zeros_like(p) for name, p in self.head.named_parameters()}
        seen_total = 0
        frozen_head = classifier_module(self.model)[1]
        captured = []

        def capture_inputs(module, inputs):
            captured.append(inputs[0].detach())

        handle = frozen_head.register_forward_pre_hook(capture_inputs)
        try:
            for client in clients:
                self.head.zero_grad(set_to_none=True)
                seen = 0
                for images, targets in client["calibration"]:
                    images, targets = images.to(self.device), targets.to(self.device)
                    captured.clear()
                    with torch.no_grad():
                        self.model(images)
                    if len(captured) != 1 or captured[0].ndim != 2:
                        raise ValueError("Expected a single final linear classifier invocation")
                    logits = self.head(captured[0])
                    losses = F.cross_entropy(logits, targets, reduction="none") * self.weights[targets]
                    # Full local empirical mean, invariant to the last batch size.
                    (losses.sum() / client["datasize"]).backward()
                    seen += targets.numel()
                if seen != client["datasize"]:
                    raise ValueError("Calibration loader must cover each local training sample once")
                coefficient = client["datasize"] / (probability * total_samples)
                for name, parameter in self.head.named_parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Non-finite teacher gradient")
                    accumulated[name].add_(parameter.grad, alpha=coefficient)
                seen_total += seen
        finally:
            handle.remove()
        norm = torch.sqrt(sum(gradient.square().sum() for gradient in accumulated.values()))
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite aggregated teacher gradient")
        factor = 1.0
        if self.cfg["teacher_grad_clip"] > 0:
            factor = min(1.0, self.cfg["teacher_grad_clip"] / max(norm.item(), 1e-12))
        with torch.no_grad():
            for name, parameter in self.head.named_parameters():
                parameter.add_(accumulated[name], alpha=-self.cfg["teacher_lr"] * factor)
        frozen_head.load_state_dict(self.head.state_dict())
        size = tensor_bytes(self.head.state_dict()) * len(clients)
        return dict(teacher_gradient_norm=norm.item(), teacher_gradient_scale=factor,
                    teacher_examples=seen_total, teacher_uplink_bytes=size, teacher_downlink_bytes=size)
