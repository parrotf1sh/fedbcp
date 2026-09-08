import torch
import torch.nn.functional as F

__all__ = ["aggregate_prototypes"]


@torch.no_grad()
def aggregate_prototypes(
    prototype_payloads,
    num_classes,
    global_prototypes=None,
    prototype_valid=None,
    prototype_age=None,
    proto_momentum=0.5,
    count_shrinkage=5.0,
    min_proto_contributors=1,
    max_prototype_age=None,
):
    """Aggregate class prototypes with confidence and small-sample safeguards."""
    if len(prototype_payloads) == 0:
        return global_prototypes, prototype_valid, prototype_age, _empty_metrics(num_classes)

    feature_dim = _infer_feature_dim(prototype_payloads)
    if feature_dim is None:
        return global_prototypes, prototype_valid, prototype_age, _empty_metrics(num_classes)

    device = torch.device("cpu")
    if global_prototypes is None:
        global_prototypes = torch.zeros(num_classes, feature_dim, device=device)
    else:
        global_prototypes = global_prototypes.detach().cpu()

    if prototype_valid is None:
        prototype_valid = torch.zeros(num_classes, dtype=torch.bool, device=device)
    else:
        prototype_valid = prototype_valid.detach().cpu().bool()

    if prototype_age is None:
        prototype_age = torch.zeros(num_classes, dtype=torch.long, device=device)
    else:
        prototype_age = prototype_age.detach().cpu().long()

    prototype_age[prototype_valid] += 1

    weighted_sums = torch.zeros(num_classes, feature_dim, device=device)
    confidence_sums = torch.zeros(num_classes, device=device)
    contributor_counts = torch.zeros(num_classes, device=device)
    uploaded_confidences = []

    for payload in prototype_payloads:
        local_prototypes = payload["prototypes"].detach().cpu()
        local_valid = payload["prototype_valid"].detach().cpu().bool()
        local_confidence = payload["confidence"].detach().cpu()
        local_counts = payload.get(
            "class_counts",
            torch.ones(num_classes, device=device),
        ).detach().cpu().float()

        _validate_payload(
            local_prototypes,
            local_valid,
            local_confidence,
            local_counts,
            num_classes,
            feature_dim,
        )

        valid_confidence = local_confidence[local_valid]
        uploaded_confidences.append(valid_confidence)

        weights = (
            local_confidence.clamp_min(0.0)
            * local_valid.float()
            * _count_shrinkage(local_counts, count_shrinkage)
        )
        weighted_sums += local_prototypes * weights.view(-1, 1)
        confidence_sums += weights
        contributor_counts += (weights > 0).float()

    class_has_update = (confidence_sums > 0) & (
        contributor_counts >= min_proto_contributors
    )
    if class_has_update.any():
        round_prototypes = weighted_sums[class_has_update] / confidence_sums[
            class_has_update
        ].view(-1, 1)
        round_prototypes = F.normalize(round_prototypes, dim=1)

        previously_valid = prototype_valid[class_has_update]
        old_prototypes = global_prototypes[class_has_update]
        merged = torch.where(
            previously_valid.view(-1, 1),
            proto_momentum * old_prototypes + (1.0 - proto_momentum) * round_prototypes,
            round_prototypes,
        )
        global_prototypes[class_has_update] = F.normalize(merged, dim=1)
        prototype_valid[class_has_update] = True
        prototype_age[class_has_update] = 0

    expired = torch.zeros(num_classes, dtype=torch.bool, device=device)
    if max_prototype_age is not None:
        expired = prototype_valid & (prototype_age > max_prototype_age)
        if expired.any():
            global_prototypes[expired] = 0
            prototype_valid[expired] = False
            prototype_age[expired] = 0

    metrics = {
        "prototype_coverage": prototype_valid.float().mean().item(),
        "prototype_confidence_mean": _confidence_mean(uploaded_confidences),
        "prototype_contributors_mean": contributor_counts.mean().item(),
        "prototype_contributors_min_updated": _min_updated_contributors(
            contributor_counts, class_has_update
        ),
        "prototype_expired": expired.float().sum().item(),
        "prototype_age_mean": prototype_age[prototype_valid].float().mean().item()
        if prototype_valid.any()
        else 0.0,
    }

    return global_prototypes, prototype_valid, prototype_age, metrics


def _infer_feature_dim(prototype_payloads):
    for payload in prototype_payloads:
        prototypes = payload["prototypes"]
        if prototypes.ndim == 2:
            return prototypes.size(1)

    return None


def _confidence_mean(uploaded_confidences):
    if len(uploaded_confidences) == 0:
        return 0.0

    non_empty = [item for item in uploaded_confidences if item.numel() > 0]
    if len(non_empty) == 0:
        return 0.0

    return torch.cat(non_empty).mean().item()


def _count_shrinkage(class_counts, count_shrinkage):
    if count_shrinkage <= 0:
        return torch.ones_like(class_counts)

    return class_counts / (class_counts + count_shrinkage)


def _validate_payload(
    local_prototypes,
    local_valid,
    local_confidence,
    local_counts,
    num_classes,
    feature_dim,
):
    if local_prototypes.shape != (num_classes, feature_dim):
        raise ValueError(
            "FedBPC prototype payload has invalid prototype shape: "
            "{} != ({}, {})".format(local_prototypes.shape, num_classes, feature_dim)
        )
    if local_valid.numel() != num_classes:
        raise ValueError("FedBPC prototype_valid length does not match num_classes.")
    if local_confidence.numel() != num_classes:
        raise ValueError("FedBPC confidence length does not match num_classes.")
    if local_counts.numel() != num_classes:
        raise ValueError("FedBPC class_counts length does not match num_classes.")
    if not torch.isfinite(local_prototypes).all():
        raise ValueError("FedBPC prototype payload contains non-finite prototypes.")
    if not torch.isfinite(local_confidence).all():
        raise ValueError("FedBPC prototype payload contains non-finite confidence.")
    if not torch.isfinite(local_counts).all():
        raise ValueError("FedBPC prototype payload contains non-finite class counts.")


def _min_updated_contributors(contributor_counts, class_has_update):
    if not class_has_update.any():
        return 0.0

    return contributor_counts[class_has_update].min().item()


def _empty_metrics(num_classes):
    return {
        "prototype_coverage": 0.0,
        "prototype_confidence_mean": 0.0,
        "prototype_contributors_mean": 0.0,
        "prototype_contributors_min_updated": 0.0,
        "prototype_expired": 0.0,
        "prototype_age_mean": 0.0,
    }
