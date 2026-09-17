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
    singleton_update_enabled=False,
    singleton_update_scale=2.0 / 3.0,
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
    raw_confidence_sums = torch.zeros(num_classes, device=device)
    contributor_counts = torch.zeros(num_classes, device=device)
    uploaded_confidences = []
    local_prototypes_by_class = [[] for _ in range(num_classes)]

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
        raw_confidence_sums += local_confidence * local_valid.float()
        contributor_counts += (weights > 0).float()
        for class_idx in torch.nonzero(local_valid, as_tuple=False).view(-1).tolist():
            local_prototypes_by_class[class_idx].append(local_prototypes[class_idx])

    class_has_update = (confidence_sums > 0) & (
        contributor_counts >= min_proto_contributors
    )
    update_cosines = torch.zeros(num_classes, device=device)
    update_cosine_valid = torch.zeros(num_classes, dtype=torch.bool, device=device)
    applied_update_cosines = torch.zeros(num_classes, device=device)
    applied_update_cosine_valid = torch.zeros(
        num_classes, dtype=torch.bool, device=device
    )
    effective_momenta = torch.zeros(num_classes, device=device)
    singleton_damped = torch.zeros(num_classes, dtype=torch.bool, device=device)
    if class_has_update.any():
        updated_class_indices = torch.nonzero(
            class_has_update, as_tuple=False
        ).view(-1)
        round_prototypes = weighted_sums[class_has_update] / confidence_sums[
            class_has_update
        ].view(-1, 1)
        round_prototypes = F.normalize(round_prototypes, dim=1)

        previously_valid = prototype_valid[class_has_update]
        old_prototypes = global_prototypes[class_has_update]
        if previously_valid.any():
            previous_class_indices = updated_class_indices[previously_valid]
            update_cosines[previous_class_indices] = F.cosine_similarity(
                old_prototypes[previously_valid],
                round_prototypes[previously_valid],
                dim=1,
            )
            update_cosine_valid[previous_class_indices] = True

        updated_momenta = torch.full(
            (updated_class_indices.numel(),),
            float(proto_momentum),
            device=device,
        )
        singleton_in_update = contributor_counts[class_has_update].eq(1)
        singleton_damping_active = (
            singleton_update_enabled
            and float(singleton_update_scale) < 1.0
            and float(proto_momentum) < 1.0
        )
        damped_in_update = (
            previously_valid & singleton_in_update & singleton_damping_active
        )
        if damped_in_update.any():
            base_update_weight = 1.0 - float(proto_momentum)
            singleton_momentum = 1.0 - (
                base_update_weight * float(singleton_update_scale)
            )
            updated_momenta[damped_in_update] = singleton_momentum

        # A class without an existing global prototype must be initialized from
        # the available round prototype; there is no prior estimate to retain.
        updated_momenta[~previously_valid] = 0.0
        effective_momenta[updated_class_indices] = updated_momenta
        singleton_damped[updated_class_indices] = damped_in_update

        merged = torch.where(
            previously_valid.view(-1, 1),
            updated_momenta.view(-1, 1) * old_prototypes
            + (1.0 - updated_momenta).view(-1, 1) * round_prototypes,
            round_prototypes,
        )
        merged = F.normalize(merged, dim=1)
        global_prototypes[class_has_update] = merged

        if previously_valid.any():
            previous_class_indices = updated_class_indices[previously_valid]
            applied_update_cosines[previous_class_indices] = F.cosine_similarity(
                old_prototypes[previously_valid],
                merged[previously_valid],
                dim=1,
            )
            applied_update_cosine_valid[previous_class_indices] = True

        prototype_valid[class_has_update] = True
        prototype_age[class_has_update] = 0

    expired = torch.zeros(num_classes, dtype=torch.bool, device=device)
    if max_prototype_age is not None:
        expired = prototype_valid & (prototype_age > max_prototype_age)
        if expired.any():
            global_prototypes[expired] = 0
            prototype_valid[expired] = False
            prototype_age[expired] = 0

    pair_cosines, pair_cosine_mins, pair_counts = _same_class_pair_cosines(
        local_prototypes_by_class, num_classes
    )
    interclass_mean, interclass_max = _interclass_cosines(
        global_prototypes, prototype_valid
    )
    raw_confidence_mean = torch.zeros(num_classes, device=device)
    has_contributors = contributor_counts > 0
    raw_confidence_mean[has_contributors] = (
        raw_confidence_sums[has_contributors]
        / contributor_counts[has_contributors]
    )
    singleton_updates = class_has_update & contributor_counts.eq(1)
    multi_updates = class_has_update & contributor_counts.ge(2)
    singleton_update_valid = update_cosine_valid & contributor_counts.eq(1)
    multi_update_valid = update_cosine_valid & contributor_counts.ge(2)

    metrics = {
        "prototype_coverage": prototype_valid.float().mean().item(),
        "prototype_memory_coverage": prototype_valid.float().mean().item(),
        "prototype_current_round_coverage": class_has_update.float().mean().item(),
        "prototype_stale_class_count": (
            prototype_valid & ~class_has_update
        ).float().sum().item(),
        "prototype_confidence_mean": _confidence_mean(uploaded_confidences),
        "prototype_contributors_mean": contributor_counts.mean().item(),
        "prototype_contributors_min_updated": _min_updated_contributors(
            contributor_counts, class_has_update
        ),
        "prototype_expired": expired.float().sum().item(),
        "prototype_age_mean": prototype_age[prototype_valid].float().mean().item()
        if prototype_valid.any()
        else 0.0,
        "prototype_update_cosine_mean": update_cosines[
            update_cosine_valid
        ].mean().item()
        if update_cosine_valid.any()
        else 0.0,
        "prototype_update_cosine_min": update_cosines[
            update_cosine_valid
        ].min().item()
        if update_cosine_valid.any()
        else 0.0,
        "prototype_update_cosine_count": update_cosine_valid.float().sum().item(),
        "prototype_applied_update_cosine_mean": _masked_mean(
            applied_update_cosines, applied_update_cosine_valid
        ),
        "prototype_applied_update_cosine_min": _masked_min(
            applied_update_cosines, applied_update_cosine_valid
        ),
        "prototype_applied_update_cosine_count": (
            applied_update_cosine_valid.float().sum().item()
        ),
        "prototype_singleton_update_count": singleton_updates.float().sum().item(),
        "prototype_singleton_damped_count": singleton_damped.float().sum().item(),
        "prototype_multi_update_count": multi_updates.float().sum().item(),
        "prototype_singleton_raw_update_cosine_mean": _masked_mean(
            update_cosines, singleton_update_valid
        ),
        "prototype_singleton_applied_update_cosine_mean": _masked_mean(
            applied_update_cosines,
            applied_update_cosine_valid & contributor_counts.eq(1),
        ),
        "prototype_multi_raw_update_cosine_mean": _masked_mean(
            update_cosines, multi_update_valid
        ),
        "prototype_multi_applied_update_cosine_mean": _masked_mean(
            applied_update_cosines,
            applied_update_cosine_valid & contributor_counts.ge(2),
        ),
        "prototype_effective_momentum_mean": _masked_mean(
            effective_momenta, applied_update_cosine_valid
        ),
        "prototype_singleton_effective_momentum_mean": _masked_mean(
            effective_momenta,
            applied_update_cosine_valid & contributor_counts.eq(1),
        ),
        "prototype_multi_effective_momentum_mean": _masked_mean(
            effective_momenta,
            applied_update_cosine_valid & contributor_counts.ge(2),
        ),
        "prototype_same_class_client_cosine_mean": (
            torch.sum(pair_cosines * pair_counts)
            / pair_counts.sum().clamp_min(1.0)
        ).item(),
        "prototype_same_class_client_cosine_min": pair_cosine_mins[
            pair_counts > 0
        ].min().item()
        if (pair_counts > 0).any()
        else 0.0,
        "prototype_same_class_pair_count": pair_counts.sum().item(),
        "prototype_interclass_cosine_mean": interclass_mean,
        "prototype_interclass_cosine_max": interclass_max,
    }

    for class_idx in range(num_classes):
        prefix = "diag_proto_class_{:03d}".format(class_idx)
        metrics[prefix + "_contributors"] = contributor_counts[class_idx].item()
        metrics[prefix + "_age"] = prototype_age[class_idx].item()
        metrics[prefix + "_updated"] = class_has_update[class_idx].float().item()
        metrics[prefix + "_confidence"] = raw_confidence_mean[class_idx].item()
        metrics[prefix + "_pair_cosine"] = pair_cosines[class_idx].item()
        metrics[prefix + "_pair_cosine_min"] = pair_cosine_mins[
            class_idx
        ].item()
        metrics[prefix + "_pair_count"] = pair_counts[class_idx].item()
        metrics[prefix + "_update_cosine"] = update_cosines[class_idx].item()
        metrics[prefix + "_update_cosine_valid"] = update_cosine_valid[
            class_idx
        ].float().item()
        metrics[prefix + "_applied_update_cosine"] = applied_update_cosines[
            class_idx
        ].item()
        metrics[prefix + "_applied_update_cosine_valid"] = (
            applied_update_cosine_valid[class_idx].float().item()
        )
        metrics[prefix + "_effective_momentum"] = effective_momenta[
            class_idx
        ].item()
        metrics[prefix + "_singleton_damped"] = singleton_damped[
            class_idx
        ].float().item()

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


def _masked_mean(values, mask):
    if not mask.any():
        return 0.0

    return values[mask].mean().item()


def _masked_min(values, mask):
    if not mask.any():
        return 0.0

    return values[mask].min().item()


def _same_class_pair_cosines(local_prototypes_by_class, num_classes):
    pair_cosines = torch.zeros(num_classes)
    pair_cosine_mins = torch.zeros(num_classes)
    pair_counts = torch.zeros(num_classes)

    for class_idx, prototype_list in enumerate(local_prototypes_by_class):
        if len(prototype_list) < 2:
            continue
        prototypes = F.normalize(torch.stack(prototype_list), dim=1)
        similarity = torch.matmul(prototypes, prototypes.t())
        upper_triangle = torch.triu(
            torch.ones_like(similarity, dtype=torch.bool), diagonal=1
        )
        values = similarity[upper_triangle]
        pair_cosines[class_idx] = values.mean()
        pair_cosine_mins[class_idx] = values.min()
        pair_counts[class_idx] = values.numel()

    return pair_cosines, pair_cosine_mins, pair_counts


def _interclass_cosines(global_prototypes, prototype_valid):
    valid_prototypes = global_prototypes[prototype_valid]
    if valid_prototypes.size(0) < 2:
        return 0.0, 0.0

    valid_prototypes = F.normalize(valid_prototypes, dim=1)
    similarity = torch.matmul(valid_prototypes, valid_prototypes.t())
    off_diagonal = ~torch.eye(
        similarity.size(0), dtype=torch.bool, device=similarity.device
    )
    values = similarity[off_diagonal]
    return values.mean().item(), values.max().item()


def _empty_metrics(num_classes):
    metrics = {
        "prototype_coverage": 0.0,
        "prototype_memory_coverage": 0.0,
        "prototype_current_round_coverage": 0.0,
        "prototype_stale_class_count": 0.0,
        "prototype_confidence_mean": 0.0,
        "prototype_contributors_mean": 0.0,
        "prototype_contributors_min_updated": 0.0,
        "prototype_expired": 0.0,
        "prototype_age_mean": 0.0,
        "prototype_update_cosine_mean": 0.0,
        "prototype_update_cosine_min": 0.0,
        "prototype_update_cosine_count": 0.0,
        "prototype_applied_update_cosine_mean": 0.0,
        "prototype_applied_update_cosine_min": 0.0,
        "prototype_applied_update_cosine_count": 0.0,
        "prototype_singleton_update_count": 0.0,
        "prototype_singleton_damped_count": 0.0,
        "prototype_multi_update_count": 0.0,
        "prototype_singleton_raw_update_cosine_mean": 0.0,
        "prototype_singleton_applied_update_cosine_mean": 0.0,
        "prototype_multi_raw_update_cosine_mean": 0.0,
        "prototype_multi_applied_update_cosine_mean": 0.0,
        "prototype_effective_momentum_mean": 0.0,
        "prototype_singleton_effective_momentum_mean": 0.0,
        "prototype_multi_effective_momentum_mean": 0.0,
        "prototype_same_class_client_cosine_mean": 0.0,
        "prototype_same_class_client_cosine_min": 0.0,
        "prototype_same_class_pair_count": 0.0,
        "prototype_interclass_cosine_mean": 0.0,
        "prototype_interclass_cosine_max": 0.0,
    }
    for class_idx in range(num_classes):
        prefix = "diag_proto_class_{:03d}".format(class_idx)
        metrics[prefix + "_contributors"] = 0.0
        metrics[prefix + "_age"] = 0.0
        metrics[prefix + "_updated"] = 0.0
        metrics[prefix + "_confidence"] = 0.0
        metrics[prefix + "_pair_cosine"] = 0.0
        metrics[prefix + "_pair_cosine_min"] = 0.0
        metrics[prefix + "_pair_count"] = 0.0
        metrics[prefix + "_update_cosine"] = 0.0
        metrics[prefix + "_update_cosine_valid"] = 0.0
        metrics[prefix + "_applied_update_cosine"] = 0.0
        metrics[prefix + "_applied_update_cosine_valid"] = 0.0
        metrics[prefix + "_effective_momentum"] = 0.0
        metrics[prefix + "_singleton_damped"] = 0.0

    return metrics
