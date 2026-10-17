import math
import os
import random
from contextlib import contextmanager
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn


def create_optimizer(model, name="sgd", **params):
    """Source TinyProto uses fresh, plain SGD and a fixed learning rate."""
    if name != "sgd":
        raise ValueError("TinyProto supports only the source SGD optimizer")
    unknown = set(params) - {"lr", "momentum", "weight_decay", "dampening", "nesterov"}
    if unknown:
        raise ValueError("Unknown TinyProto optimizer settings: {}".format(sorted(unknown)))
    if any(params.get(key, 0) != 0 for key in ("momentum", "weight_decay", "dampening")):
        raise ValueError("TinyProto requires zero momentum, weight_decay and dampening")
    if params.get("nesterov", False) is not False:
        raise ValueError("TinyProto does not use Nesterov SGD")
    learning_rate = params.get("lr", 0.01)
    if (type(learning_rate) not in (int, float)
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("TinyProto learning rate must be finite and positive")
    return torch.optim.SGD(model.parameters(), lr=learning_rate)


def classifier_module(model):
    for name in ("classifier", "linear", "linear_2"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Linear):
            return module
    raise ValueError("TinyProto requires a project CNN, VGG or ResNet classifier")


def forward_with_features(model, data):
    """Observe the classifier input in one unchanged project model forward."""
    features = []

    def collect_features(module, inputs):
        features.append(inputs[0])

    handle = classifier_module(model).register_forward_pre_hook(collect_features)
    try:
        logits = model(data)
    finally:
        handle.remove()
    if len(features) != 1 or features[0].ndim != 2:
        raise ValueError("TinyProto expects one batch of classifier-input vectors")
    return logits, features[0]


def hamming_distance(vector1, vector2):
    return np.count_nonzero(vector1 != vector2)


def generate_random_vector(n, ones_count):
    vector = np.zeros(n, dtype=int)
    vector[:ones_count] = 1
    np.random.shuffle(vector)
    return vector


def create_start_vector(n, ones_count):
    vector = np.zeros(n, dtype=int)
    vector[-ones_count:] = 1
    return vector


def find_optimal_vectors_greedy(n, ones_count, max_attempts=10000):
    """Retain upstream candidate order, uniqueness, strict ties and attempt limit."""
    if n < 2 or not 0 < ones_count < n or math.comb(n, ones_count) < n:
        raise ValueError("TinyProto cannot generate enough distinct nonempty class masks")
    vectors = [create_start_vector(n, ones_count)]
    for _ in range(n - 1):
        best_vector = None
        max_total_distance = -1
        attempts = 0
        while attempts < max_attempts:
            candidate = generate_random_vector(n, ones_count)
            if any(np.array_equal(candidate, vector) for vector in vectors):
                attempts += 1
                continue
            total_distance = sum(hamming_distance(candidate, vector) for vector in vectors)
            if total_distance > max_total_distance:
                max_total_distance = total_distance
                best_vector = candidate
            attempts += 1
        if best_vector is None:
            raise ValueError("Failed to find a unique vector after {} attempts".format(max_attempts))
        vectors.append(best_vector)
    vectors = np.array(vectors)
    total_distance = sum(hamming_distance(v1, v2) for v1, v2 in combinations(vectors, 2))
    return vectors, total_distance / (n * (n - 1) / 2)


def create_mask_indices(feature_dim, num_classes, cfg):
    if not cfg["add_cps"]:
        return {}
    length = feature_dim // num_classes
    per_mask = int(num_classes * (cfg["cps_ratio"] / 100))
    if (length < 1 or num_classes < 2 or not 0 < per_mask < num_classes
            or math.comb(num_classes, per_mask) < num_classes):
        raise ValueError("TinyProto feature dimension/classes/cps_ratio cannot form valid masks")
    identity = dict(feature_dim=feature_dim, num_classes=num_classes,
                    cps_ratio=cfg["cps_ratio"], mask_max_attempts=cfg["mask_max_attempts"])
    path = cfg["mask_cache_path"]
    if path is not None and os.path.exists(path):
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(cached, dict) or cached.get("identity") != identity:
            raise ValueError("TinyProto mask cache does not match the current configuration")
        indices = cached.get("indices")
    else:
        # Mask construction must not consume the subsequent training RNG stream.
        with preserve_evaluation_rng([]):
            vectors, _ = find_optimal_vectors_greedy(
                num_classes, per_mask, cfg["mask_max_attempts"]
            )
        indices = {}
        for label in range(num_classes):
            masked_proto = np.zeros(feature_dim)
            for index, value in enumerate(vectors[label]):
                start = index * length
                masked_proto[start:start + length] = value
            indices[label] = torch.as_tensor(np.nonzero(masked_proto)[0], dtype=torch.long)
    if not isinstance(indices, dict) or set(indices) != set(range(num_classes)):
        raise ValueError("TinyProto mask cache must contain every class")
    seen = set()
    for value in indices.values():
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.long
                or value.ndim != 1 or value.numel() != per_mask * length):
            raise ValueError("Invalid TinyProto mask indices")
        values = value.tolist()
        blocks = sorted(set(index // length for index in values))
        expected = [index for block in blocks for index in range(block * length, (block + 1) * length)]
        if (values != expected or len(blocks) != per_mask or blocks[0] < 0
                or blocks[-1] >= num_classes or tuple(values) in seen):
            raise ValueError("TinyProto masks must be distinct complete blocks within the feature dimension")
        seen.add(tuple(values))
    if path is not None and not os.path.exists(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(dict(identity=identity, indices=indices), path)
    return indices


def average_prototypes(prototypes, class_counts, mask_indices, cfg):
    """Source agg_func: sequential local mean, optional count scaling, then CPS."""
    result = {}
    for label, values in prototypes.items():
        if not values:
            raise ValueError("Cannot average an empty TinyProto prototype list")
        if len(values) > 1:
            prototype = 0 * values[0].detach()
            for value in values:
                prototype += value.detach()
            prototype = prototype / len(values)
        else:
            prototype = values[0].detach()
        if cfg["simple_scale"]:
            prototype = prototype * class_counts[label]
        if cfg["add_cps"]:
            prototype = prototype[mask_indices[label].to(prototype.device)]
        result[label] = prototype
    return result


def aggregate_prototypes(local_prototypes, cfg):
    by_class = {}
    for prototypes in local_prototypes:
        for label, prototype in prototypes.items():
            by_class.setdefault(label, []).append(prototype)
    for label, values in by_class.items():
        prototypes = torch.stack(values)
        if cfg["simple_scale"]:
            by_class[label] = (len(prototypes) * cfg["constant_scale_factor"]
                               * torch.mean(prototypes, dim=0).detach())
        else:
            by_class[label] = torch.mean(prototypes, dim=0).detach()
    # Approved source repair: return AFTER every class has been aggregated.
    return by_class


def restore_prototypes(global_prototypes, mask_indices, feature_dim, device):
    result = {}
    for label, prototype in global_prototypes.items():
        prototype = prototype.detach().to(device)
        if mask_indices:
            indices = mask_indices[label].to(device)
            if prototype.shape != indices.shape:
                raise ValueError("TinyProto compressed prototype and mask sizes differ")
            restored = prototype.new_zeros(feature_dim)
            restored[indices] = prototype
        else:
            if prototype.shape != (feature_dim,):
                raise ValueError("TinyProto prototype and feature dimensions differ")
            restored = prototype
        if not torch.isfinite(restored).all():
            raise FloatingPointError("Non-finite TinyProto global prototype")
        result[label] = restored
    return result


def cpu_snapshot(weights):
    return {key: value.detach().cpu().clone() for key, value in weights.items()}


@contextmanager
def preserve_evaluation_rng(loaders):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    generators = {}
    for loader in loaders:
        generator = getattr(loader, "generator", None)
        if generator is not None and id(generator) not in generators:
            generators[id(generator)] = (generator, generator.get_state())
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        for generator, state in generators.values():
            generator.set_state(state)
