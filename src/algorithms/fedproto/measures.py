import math

import torch
import torch.nn.functional as F

from .criterion import prototype_loss
from .utils import forward_with_features


def _classification_metrics(count, correct, predicted, groups):
    counts, hits = count.tolist(), correct.tolist()
    total = sum(counts)
    if total == 0:
        raise ValueError("FedProto evaluation dataset is empty")
    class_accuracy = [hit / size if size else None for hit, size in zip(hits, counts)]
    present = [value for value in class_accuracy if value is not None]
    f1 = 2 * correct.double() / (count + predicted).clamp_min(1)
    worst_count = max(1, math.ceil(len(present) * 0.2))
    result = dict(
        accuracy=sum(hits) / total,
        macro_accuracy=sum(present) / len(present),
        macro_f1=f1[count > 0].mean().item(),
        worst20_recall=sum(sorted(present)[:worst_count]) / worst_count,
        class_accuracy=class_accuracy,
        class_count=counts,
        evaluated_classes=len(present),
        samples=total,
    )
    for name, indices in groups.items():
        values = [class_accuracy[index] for index in indices
                  if class_accuracy[index] is not None]
        result[name] = sum(values) / len(values) if values else None
    return result


@torch.no_grad()
def evaluate_client(model, loader, global_prototypes, local_classes,
                    num_classes, groups, device, missing_class_distance=100.0):
    """Separate head/prototype counters; keep upstream distance predictions."""
    if loader is None:
        return None
    previous_mode = model.training
    model.to(device)
    model.eval()
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    head_correct, proto_correct = torch.zeros_like(counts), torch.zeros_like(counts)
    head_predicted, proto_predicted = torch.zeros_like(counts), torch.zeros_like(counts)
    ce_sum, proto_sum = 0.0, 0.0
    eligible_classes = set(global_prototypes).intersection(local_classes)
    try:
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            logits, features = forward_with_features(model, data)
            if logits.ndim != 2 or logits.shape[1] != num_classes:
                raise ValueError("FedProto classifier size does not match dataset classes")
            if not torch.isfinite(logits).all() or not torch.isfinite(features).all():
                raise FloatingPointError("Non-finite FedProto evaluation output")
            log_probs = F.log_softmax(logits, dim=1)
            ce_sum += F.nll_loss(log_probs, targets, reduction="sum").item()
            predictions = log_probs.argmax(dim=1)
            counts += torch.bincount(targets, minlength=num_classes)
            head_correct += torch.bincount(targets[predictions.eq(targets)], minlength=num_classes)
            head_predicted += torch.bincount(predictions, minlength=num_classes)

            if global_prototypes:
                # Keep the finite sentinel, original nested loops, and argmin ties.
                distances = missing_class_distance * torch.ones(
                    (data.size(0), num_classes), device=device
                )
                for index in range(data.size(0)):
                    for class_idx in range(num_classes):
                        if class_idx in global_prototypes and class_idx in local_classes:
                            distances[index, class_idx] = F.mse_loss(
                                features[index], global_prototypes[class_idx]
                            )
                predictions = distances.argmin(dim=1)
                proto_correct += torch.bincount(targets[predictions.eq(targets)], minlength=num_classes)
                proto_predicted += torch.bincount(predictions, minlength=num_classes)
                proto_sum += prototype_loss(features, targets, global_prototypes).item() * data.size(0)
    finally:
        model.train(previous_mode)

    counts = counts.cpu()
    head = _classification_metrics(counts, head_correct.cpu(), head_predicted.cpu(), groups)
    head["cross_entropy"] = ce_sum / head["samples"]
    proto = None
    if global_prototypes:
        proto = _classification_metrics(counts, proto_correct.cpu(), proto_predicted.cpu(), groups)
        proto["prototype_mse"] = proto_sum / proto["samples"]
    return dict(classifier=head, prototype=proto, eligible_classes=len(eligible_classes))


def average_client_metrics(records, prediction):
    """Equal mean across ALL client predictors on the same complete dataset."""
    values = [record[prediction] for record in records]
    if not values or any(value is None for value in values):
        return None
    counts = values[0]["class_count"]
    if any(value["class_count"] != counts for value in values):
        raise ValueError("Overall FedProto metrics require the same test set for every client")
    result = {}
    for key in values[0]:
        if key in ("class_accuracy", "class_count", "evaluated_classes", "samples"):
            continue
        items = [value[key] for value in values]
        result[key] = None if items[0] is None else sum(items) / len(items)
    result["class_accuracy"] = [
        None if count == 0 else sum(value["class_accuracy"][index] for value in values) / len(values)
        for index, count in enumerate(counts)
    ]
    result["class_count"] = counts
    result["samples_per_client"] = sum(counts)
    result["evaluated_classes"] = values[0]["evaluated_classes"]
    result["evaluated_clients"] = len(values)
    result["client_accuracy_std"] = math.sqrt(
        sum((value["accuracy"] - result["accuracy"]) ** 2 for value in values) / len(values)
    )
    return result
