import torch
import torch.nn as nn


def create_optimizer(model, name="sgd", **params):
    """Keep all three optimizer branches provided by the FedProc source."""
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **params)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **params)
    if name == "amsgrad":
        params = dict(params, amsgrad=True)
        return torch.optim.Adam(model.parameters(), **params)
    raise ValueError("Unknown FedProc optimizer: {}".format(name))


def classifier_module(model):
    """Locate the existing classifier without changing the model forward pass."""
    for name in ("classifier", "linear", "linear_2"):
        head = getattr(model, name, None)
        if isinstance(head, nn.Linear):
            return name, head
    raise ValueError("FedProc requires a final Linear named classifier, linear or linear_2")


def forward_with_features(model, data):
    features = []

    def collect_features(module, inputs):
        features.append(inputs[0])

    handle = classifier_module(model)[1].register_forward_pre_hook(collect_features)
    try:
        logits = model(data)
    finally:
        handle.remove()
    return logits, features[0]


def aggregate_prototypes(payloads, num_classes, global_prototypes=None):
    """Divide per-class feature sums by sample counts across selected clients."""
    prototypes = []
    template = next(
        (feature for payload in payloads for feature in payload["feature_sums"].values()),
        None,
    )
    if template is None and global_prototypes is None:
        raise ValueError("FedProc cannot initialize prototypes from empty training loaders")

    for class_idx in range(num_classes):
        feature_sum = None
        class_count = 0
        for payload in payloads:
            if class_idx in payload["feature_sums"]:
                feature = payload["feature_sums"][class_idx]
                if feature_sum is None:
                    feature_sum = feature.clone()
                else:
                    feature_sum += feature
                class_count += payload["class_counts"][class_idx]

        if feature_sum is None:
            if global_prototypes is None:
                prototypes.append(torch.zeros_like(template))
            else:
                prototypes.append(global_prototypes[class_idx].clone())
        else:
            prototypes.append(feature_sum / class_count)

    return prototypes
