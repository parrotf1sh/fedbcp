import inspect

import torch
import torch.nn as nn

# These helpers implement the same sequential FedProto averages as FedSA's
# agg_func/proto_aggregation, or only copy state/preserve evaluation randomness.
# Do not reuse FedProto's feature location or prototype candidate filtering.
from ..fedproto.utils import (
    aggregate_prototypes,
    average_prototypes,
    cpu_snapshot,
    preserve_evaluation_rng,
)


def classifier_module(model):
    """Use the existing FedBPC-compatible model and its unmodified linear head."""
    classifier = getattr(model, "classifier", None)
    if (not isinstance(classifier, nn.Linear)
            or "get_features" not in inspect.signature(model.forward).parameters):
        raise ValueError(
            "FedSA requires forward(data, get_features=True) and a linear classifier; "
            "use the same fedavg_cifar, fedavg_tiny, or vgg11 model as FedBPC"
        )
    return classifier


def forward_with_features(model, data):
    logits, features = model(data, get_features=True)
    classifier = classifier_module(model)
    if (features.ndim != 2 or features.shape[1] != classifier.in_features
            or logits.shape != (features.shape[0], classifier.out_features)):
        raise ValueError("FedSA requires classifier-input features with shape [batch, feature_dim]")
    return logits, features


def initialize_anchors(num_classes, feature_dim, seed, device):
    """Upstream CPU Linear/SyncBN/ReLU initialization, with an isolated CPU seed."""
    with torch.random.fork_rng(devices=[]):
        # Unlike torch.manual_seed, this does not reset CUDA training streams.
        torch.random.default_generator.manual_seed(seed)
        projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, device="cpu"),
            nn.SyncBatchNorm(feature_dim, eps=1e-5, device="cpu"),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim, device="cpu"),
            nn.SyncBatchNorm(feature_dim, eps=1e-5, device="cpu"),
            nn.ReLU(inplace=True),
        )
        random_centers = torch.randn(num_classes, feature_dim, device="cpu")
        anchors = projector(random_centers).detach()
    return anchors.to(device)


@torch.no_grad()
def update_anchors(semantic_anchors, global_prototypes, momentum):
    # Preserve both literal upstream coefficients at the default setting.
    update_weight = 0.0001 if momentum == 0.9999 else 1.0 - momentum
    for label, prototype in global_prototypes.items():
        semantic_anchors[label] = (
            momentum * semantic_anchors[label] + update_weight * prototype
        )


@torch.no_grad()
def class_margins(anchors):
    distances = torch.cdist(anchors, anchors, p=2)
    distances.fill_diagonal_(float("inf"))
    margins = torch.min(distances, dim=1).values
    return dict(min=margins.min().item(), mean=margins.mean().item(),
                max=margins.max().item())
