import torch
import torch.nn as nn

from longtail_metrics import evaluate, groups_from_counts


def classifier_module(model):
    for name in ("classifier", "linear", "linear_2"):
        head = getattr(model, name, None)
        if isinstance(head, nn.Linear):
            return name, head
    raise ValueError("FedBTR requires a final Linear named classifier, linear or linear_2")


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def aggregate_states(states, sizes):
    if not states or len(states) != len(sizes) or min(sizes) <= 0:
        raise ValueError("Cannot aggregate empty or invalid client updates")
    total = float(sum(sizes))
    result = {}
    for name, value in states[0].items():
        if value.is_floating_point():
            result[name] = sum(state[name] * (size / total) for state, size in zip(states, sizes))
        else:
            # BatchNorm counters are integer metadata, not trainable weights.
            result[name] = torch.stack([state[name] for state in states]).max(dim=0).values
    return result


def tensor_bytes(state):
    return sum(t.numel() * t.element_size() for t in state.values())


def sync_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
