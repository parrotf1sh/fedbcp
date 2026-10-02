import math

import torch
import torch.nn as nn

from .config import resolve_config

__all__ = ["ModelWithNormalizedHead"]


class ModelWithNormalizedHead(nn.Module):
    """Project feature extractor with the original FedNH prototype head."""

    def __init__(self, model, algo_params):
        super().__init__()
        self.cfg = resolve_config(algo_params)
        for name in ("classifier", "linear", "linear_2"):
            classifier = getattr(model, name, None)
            if isinstance(classifier, nn.Linear):
                break
        else:
            raise NotImplementedError(
                "FedNH requires a final Linear named classifier, linear or linear_2"
            )

        num_classes, feature_dim = classifier.out_features, classifier.in_features
        # Calling the project forward with an Identity preserves every backbone op.
        setattr(model, name, nn.Identity())
        self.features = model
        self.prototype = nn.Parameter(classifier.weight.detach().clone(), requires_grad=False)
        scaling = self.cfg["scaling_init"]
        if scaling is None:
            # Upstream Conv2CifarNH starts at 1; ResNetModNH starts at 20.
            scaling = 20.0 if name == "linear" else 1.0
        self.scaling = nn.Parameter(self.prototype.new_tensor([float(scaling)]))
        self._initialize_head(num_classes, feature_dim)

    @torch.no_grad()
    def _initialize_head(self, num_classes, feature_dim):
        if self.cfg["head_init"] == "orthogonal":
            # Keep the upstream rand -> orthogonal_ initialization, without
            # an additional row normalization (also when classes > features).
            prototype = nn.init.orthogonal_(torch.rand(num_classes, feature_dim))
        elif self.cfg["head_init"] == "uniform" and feature_dim == 2:
            prototype = torch.zeros(num_classes, 2)
            for class_idx in range(num_classes):
                theta = class_idx * 2 * torch.pi / num_classes
                prototype[class_idx] = torch.tensor([math.cos(theta), math.sin(theta)])
        else:
            raise NotImplementedError("FedNH uniform initialization requires 2D features")
        self.prototype.copy_(prototype.to(self.prototype))
        if self.cfg["fix_scaling"]:
            # Upstream uses a scalar tensor for fixed scaling.
            self.scaling = nn.Parameter(
                self.prototype.new_tensor(float(self.cfg["fixed_scaling"])),
                requires_grad=False,
            )

    def forward(self, data, get_features=False):
        features = self.features(data)
        feature_norm = torch.norm(features, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        features = torch.div(features, feature_norm)
        if not self.prototype.requires_grad:
            prototype = self.prototype
        else:
            prototype_norm = torch.norm(
                self.prototype, p=2, dim=1, keepdim=True
            ).clamp(min=1e-12)
            prototype = torch.div(self.prototype, prototype_norm)
        logits = torch.matmul(features, prototype.T)
        logits = self.scaling * logits
        if get_features:
            return logits, features
        return logits
