"""Shared evaluation only; no algorithm-specific training or model selection."""

import torch


def groups_from_counts(counts):
    order = sorted(range(len(counts)), key=lambda c: (-int(counts[c]), c))
    quotient, remainder = divmod(len(order), 3)
    groups, offset = {}, 0
    for i, name in enumerate(("head", "middle", "tail")):
        size = quotient + int(i < remainder)
        groups[name] = order[offset:offset + size]
        offset += size
    return groups


@torch.no_grad()
def evaluate(model, loader, num_classes, groups, device):
    if loader is None:
        return None
    previous_mode = model.training
    model.to(device)
    model.eval()
    count = torch.zeros(num_classes, dtype=torch.long, device=device)
    correct = torch.zeros_like(count)
    loss_sum = 0.0
    try:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            if logits.ndim != 2 or logits.shape[1] != num_classes or not torch.isfinite(logits).all():
                raise FloatingPointError("Invalid or non-finite logits during evaluation")
            predictions = logits.argmax(dim=1)
            count += torch.bincount(labels, minlength=num_classes)
            correct += torch.bincount(labels[predictions.eq(labels)], minlength=num_classes)
            loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
    finally:
        model.train(previous_mode)
    counts, hits = count.cpu().tolist(), correct.cpu().tolist()
    per_class = [hit / n if n else None for hit, n in zip(hits, counts)]
    present = [value for value in per_class if value is not None]
    if not present:
        raise ValueError("Empty evaluation dataset")
    result = dict(accuracy=sum(hits) / sum(counts), macro_accuracy=sum(present) / len(present),
                  cross_entropy=loss_sum / sum(counts), class_accuracy=per_class,
                  class_count=counts, evaluated_classes=len(present))
    for name, indices in groups.items():
        values = [per_class[c] for c in indices if per_class[c] is not None]
        result[name] = sum(values) / len(values) if values else None
    return result


