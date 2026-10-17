import math

import torch
import torch.nn.functional as F

from .utils import forward_with_features


def _classification_metrics(count, correct, predicted, groups):
    counts, hits = count.tolist(), correct.tolist()
    total = sum(counts)
    if total == 0:
        raise ValueError("FedSA evaluation dataset is empty")
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
        values = [class_accuracy[index] for index in indices if class_accuracy[index] is not None]
        result[name] = sum(values) / len(values) if values else None
    return result


@torch.no_grad()
def evaluate_client(model, loader, global_prototypes, num_classes, groups, device):
    """clientSA.test_metrics: all uploaded classes, infinite sentinel, MSE argmin."""
    if loader is None:
        return None
    previous_mode = model.training
    model.to(device)
    model.eval()
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    head_correct, proto_correct = torch.zeros_like(counts), torch.zeros_like(counts)
    head_predicted, proto_predicted = torch.zeros_like(counts), torch.zeros_like(counts)
    ce_sum = 0.0
    prototypes = (None if global_prototypes is None else {
        label: value.detach().to(device) for label, value in global_prototypes.items()
        if not isinstance(value, list)
    })
    try:
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            logits, features = forward_with_features(model, data)
            if logits.shape[1] != num_classes:
                raise ValueError("FedSA classifier size does not match dataset classes")
            if not torch.isfinite(logits).all() or not torch.isfinite(features).all():
                raise FloatingPointError("Non-finite FedSA evaluation output")
            counts += torch.bincount(targets, minlength=num_classes)
            predictions = logits.argmax(dim=1)
            head_correct += torch.bincount(targets[predictions.eq(targets)], minlength=num_classes)
            head_predicted += torch.bincount(predictions, minlength=num_classes)
            ce_sum += F.cross_entropy(logits, targets, reduction="sum").item()

            if prototypes is not None:
                distances = float("inf") * torch.ones(
                    targets.shape[0], num_classes, device=device
                )
                for index, feature in enumerate(features):
                    for label, prototype in prototypes.items():
                        distances[index, label] = F.mse_loss(feature, prototype)
                predictions = torch.argmin(distances, dim=1)
                proto_correct += torch.bincount(targets[predictions.eq(targets)], minlength=num_classes)
                proto_predicted += torch.bincount(predictions, minlength=num_classes)
    finally:
        model.train(previous_mode)
    counts = counts.cpu()
    classifier = _classification_metrics(counts, head_correct.cpu(), head_predicted.cpu(), groups)
    classifier["cross_entropy"] = ce_sum / classifier["samples"]
    prototype = (None if prototypes is None else
                 _classification_metrics(counts, proto_correct.cpu(), proto_predicted.cpu(), groups))
    return dict(classifier=classifier, prototype=prototype,
                eligible_classes=0 if prototypes is None else len(prototypes))


def average_client_metrics(records, prediction):
    """Equal mean of all persistent client predictors on one complete test set."""
    values = [record[prediction] for record in records]
    if not values or any(value is None for value in values):
        return None
    counts = values[0]["class_count"]
    if any(value["class_count"] != counts for value in values):
        raise ValueError("Overall FedSA metrics require the same test set for every client")
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


def local_test_summary(records):
    """Original server's sample-weighted accuracy on existing local test splits."""
    if not records:
        return None
    summary = dict(evaluated_clients=len(records))
    for prediction in ("prototype", "classifier"):
        values = [record[prediction] for record in records]
        if any(value is None for value in values):
            summary[prediction] = None
            continue
        samples = sum(value["samples"] for value in values)
        summary[prediction] = dict(
            accuracy=sum(value["accuracy"] * value["samples"] for value in values) / samples,
            samples=samples,
        )
    return summary
