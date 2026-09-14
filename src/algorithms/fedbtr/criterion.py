"""Full-class tolerance-band retention; no NTD or label-masking loss is used."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def global_class_weights(counts, power=1.0):
    counts = torch.as_tensor(counts, dtype=torch.float32)
    if counts.ndim != 1 or not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Global class counts must be finite, positive and one-dimensional")
    prior = counts / counts.sum()
    weights = prior.pow(-power)
    # E_{global training prior}[weight] = 1. Do NOT renormalize per minibatch.
    return weights / (prior * weights).sum()


def weighted_classification(logits, targets, weights):
    return (F.cross_entropy(logits, targets, reduction="none") * weights[targets]).mean()


class RetentionLoss(nn.Module):
    def __init__(self, global_counts, cfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("class_weights", global_class_weights(
            global_counts, cfg["student_balance_power"]))

    def forward(self, logits, targets, teacher_logits, local_counts, strength=1.0):
        classification = weighted_classification(logits, targets, self.class_weights)
        zero = logits.new_zeros(())
        retention, active, teacher_correct = zero, zero, zero
        if teacher_logits is not None and self.cfg["retention_mode"] != "none" and strength > 0:
            teacher_logits = teacher_logits.detach()
            counts = torch.as_tensor(local_counts, dtype=logits.dtype, device=logits.device)
            if counts.shape != self.class_weights.shape or (counts < 0).any():
                raise ValueError("Local class counts have an invalid shape or value")
            support = self.cfg["support_scale"] / (counts + self.cfg["support_scale"])
            if self.cfg["uniform_support"]:
                # Keep the mean strength fixed while removing class dependence.
                support = support.mean().expand_as(support)
            correct = teacher_logits.argmax(dim=1).eq(targets)
            teacher_correct = correct.float().mean()
            # Sample-level label-conflict relaxation, not a mask on class outputs.
            sample_weight = torch.where(correct, logits.new_ones(targets.shape),
                                        logits.new_full(targets.shape, self.cfg["teacher_error_weight"]))
            if self.cfg["retention_mode"] == "band":
                student = logits - logits.mean(dim=1, keepdim=True)
                teacher = teacher_logits - teacher_logits.mean(dim=1, keepdim=True)
                tolerance = self.cfg["band_min"] + (
                    self.cfg["band_max"] - self.cfg["band_min"]) * (1 - support)
                excess = (torch.abs(student - teacher) - tolerance).clamp_min(0)
                # Divide by C, not support.sum(): absolute support strength must survive.
                per_sample = (support * excess.square()).mean(dim=1)
                active = (excess > 0).float().mean()
            else:
                temperature = self.cfg["kl_temperature"]
                per_sample = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="none").sum(dim=1) * temperature ** 2
                active = logits.new_ones(())
            retention = (per_sample * sample_weight).mean()
        total = classification + strength * self.cfg["retention_weight"] * retention
        return total, dict(classification=classification.detach(), retention=retention.detach(),
                          band_active=active.detach(), teacher_correct=teacher_correct.detach())
