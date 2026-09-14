"""Deterministic, dependency-free long-tail manifests shared by all algorithms.

Indices refer to the ORIGINAL training dataset. Validation is held out before
long-tail subsampling. The manifest stores indices/statistics, not images;
held-out validation and official test samples are excluded from training.
"""

import hashlib
import json
import math
import os
import random
import tempfile


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def split_spec(n_clients, partition, imbalance_factor=100.0,
               validation_fraction=0.04, split_seed=19940817,
               class_order_seed=2022, min_client_samples=1, max_attempts=200):
    method = partition.get("method", "iid")
    method = {"stratified_iid": "iid", "dirichlet": "lda"}.get(method, method)
    if method not in ("iid", "lda"):
        raise ValueError("longtail pipeline supports iid/stratified_iid and lda/dirichlet")
    if int(n_clients) != n_clients or n_clients < 1:
        raise ValueError("n_clients must be a positive integer")
    if not math.isfinite(imbalance_factor) or imbalance_factor < 1:
        raise ValueError("imbalance_factor must be finite and >= 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if int(min_client_samples) != min_client_samples or min_client_samples < 1:
        raise ValueError("min_client_samples must be a positive integer")
    if int(max_attempts) != max_attempts or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    alpha = float(partition.get("alpha", 0.5)) if method == "lda" else None
    if alpha is not None and (not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("Dirichlet alpha must be finite and positive")
    return dict(n_clients=int(n_clients), method=method, alpha=alpha,
                imbalance_factor=float(imbalance_factor),
                validation_fraction=float(validation_fraction),
                split_seed=int(split_seed), class_order_seed=int(class_order_seed),
                min_client_samples=int(min_client_samples), max_attempts=int(max_attempts))


def build_manifest(targets, **kwargs):
    labels = [int(x) for x in targets]
    classes = sorted(set(labels))
    if len(classes) < 2 or classes != list(range(len(classes))):
        raise ValueError("Expected at least two contiguous classes numbered from zero")
    spec = split_spec(**kwargs)
    rng = random.Random(spec["split_seed"])
    by_class = [[] for _ in classes]
    for index, label in enumerate(labels):
        by_class[label].append(index)
    validation, pools = [], []
    for indices in by_class:
        rng.shuffle(indices)
        count = int(len(indices) * spec["validation_fraction"])
        if spec["validation_fraction"] > 0:
            if len(indices) < 2:
                raise ValueError("Cannot hold out validation while preserving every training class")
            count = max(1, min(count, len(indices) - 1))
        validation.extend(indices[:count])
        pools.append(indices[count:])

    order = classes[:]
    random.Random(spec["class_order_seed"]).shuffle(order)
    maximum = min(map(len, pools))
    train_by_class = [[] for _ in classes]
    for rank, label in enumerate(order):
        desired = max(1, int(maximum * spec["imbalance_factor"] **
                             (-rank / (len(classes) - 1))))
        train_by_class[label] = pools[label][:desired]
    train = sorted(index for indices in train_by_class for index in indices)
    if len(train) < spec["n_clients"] * spec["min_client_samples"]:
        raise ValueError("Too few long-tail samples for the requested client minimum")

    if spec["method"] == "iid":
        clients = [[] for _ in range(spec["n_clients"])]
        for indices in train_by_class:
            quotient, remainder = divmod(len(indices), len(clients))
            allocation = list(range(len(clients)))
            rng.shuffle(allocation)
            # Random tie breaking, then fill smaller clients with remainders.
            allocation.sort(key=lambda k: len(clients[k]))
            position = 0
            for slot, client in enumerate(allocation):
                count = quotient + int(slot < remainder)
                clients[client].extend(indices[position:position + count])
                position += count
        if min(map(len, clients)) < spec["min_client_samples"]:
            raise ValueError("Class-wise equal allocation cannot meet min_client_samples")
    else:
        clients = None
        for _ in range(spec["max_attempts"]):
            proposal = [[] for _ in range(spec["n_clients"])]
            valid_draw = True
            for indices in train_by_class:
                weights = [rng.gammavariate(spec["alpha"], 1.0) for _ in proposal]
                if sum(weights) == 0:
                    valid_draw = False
                    break
                assigned = rng.choices(range(len(proposal)), weights=weights, k=len(indices))
                for index, client in zip(indices, assigned):
                    proposal[client].append(index)
            if valid_draw and min(map(len, proposal)) >= spec["min_client_samples"]:
                clients = proposal
                break
        if clients is None:
            raise ValueError("Dirichlet allocation failed: increase alpha/max_attempts or reduce clients; "
                             "no silent sample relocation is performed")

    data_map = []
    for indices in clients:
        indices.sort()
        counts = [0] * len(classes)
        for index in indices:
            counts[labels[index]] += 1
        data_map.append(counts)
    global_counts = [len(indices) for indices in train_by_class]
    manifest = dict(version=1, spec=spec, targets_sha256=_digest(labels),
                    train_indices=train, validation_indices=sorted(validation),
                    client_indices=clients, data_map=data_map,
                    global_counts=global_counts, class_order=order,
                    class_coverage=[sum(row[c] > 0 for row in data_map) for c in classes],
                    actual_imbalance_factor=max(global_counts) / min(global_counts),
                    dropped_training_pool_count=len(labels) - len(train) - len(validation))
    manifest["sha256"] = _digest(manifest)
    validate_manifest(manifest, labels, spec)
    return manifest


def validate_manifest(manifest, targets, expected_spec=None):
    labels = [int(x) for x in targets]
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("version") != 1 or manifest.get("sha256") != _digest(payload):
        raise ValueError("Manifest version or checksum mismatch")
    if manifest["targets_sha256"] != _digest(labels):
        raise ValueError("Manifest does not match this dataset's ordered training labels")
    if expected_spec is not None and manifest["spec"] != expected_spec:
        raise ValueError("Manifest settings differ; choose another manifest path")
    train, validation = manifest["train_indices"], manifest["validation_indices"]
    flattened = [i for client in manifest["client_indices"] for i in client]
    for indices in (train, validation, flattened):
        if any(type(i) is not int or i < 0 or i >= len(labels) for i in indices):
            raise ValueError("Invalid sample index in manifest")
        if len(indices) != len(set(indices)):
            raise ValueError("Repeated sample index in manifest")
    if set(train) & set(validation) or sorted(flattened) != sorted(train):
        raise ValueError("Training/validation leakage or incomplete client allocation")
    if len(manifest["client_indices"]) != manifest["spec"]["n_clients"]:
        raise ValueError("Wrong number of clients")
    counts = [[0] * len(set(labels)) for _ in manifest["client_indices"]]
    for client, indices in enumerate(manifest["client_indices"]):
        if len(indices) < manifest["spec"]["min_client_samples"]:
            raise ValueError("Client does not meet minimum sample count")
        for index in indices:
            counts[client][labels[index]] += 1
    global_counts = [sum(row[c] for row in counts) for c in range(len(counts[0]))]
    if counts != manifest["data_map"] or global_counts != manifest["global_counts"]:
        raise ValueError("Incorrect class statistics in manifest")
    if min(global_counts) <= 0:
        raise ValueError("Every class must exist in global training data")
    coverage = [sum(row[c] > 0 for row in counts) for c in range(len(global_counts))]
    if coverage != manifest["class_coverage"]:
        raise ValueError("Incorrect class coverage in manifest")
    if manifest["actual_imbalance_factor"] != max(global_counts) / min(global_counts):
        raise ValueError("Incorrect actual imbalance factor in manifest")
    if manifest["dropped_training_pool_count"] != len(labels) - len(train) - len(validation):
        raise ValueError("Incorrect dropped sample count in manifest")


def load_or_create_manifest(path, targets, **kwargs):
    """Never overwrite an incompatible existing split."""
    spec = split_spec(**kwargs)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        validate_manifest(manifest, targets, spec)
        return manifest
    manifest = build_manifest(targets, **kwargs)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".split-", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        # link is atomic and refuses to replace another process's manifest.
        try:
            os.link(temporary, path)
        except FileExistsError:
            with open(path, encoding="utf-8") as handle:
                existing = json.load(handle)
            validate_manifest(existing, targets, spec)
            if existing["sha256"] != manifest["sha256"]:
                raise ValueError("Concurrent split generation produced different manifests")
    finally:
        os.unlink(temporary)
    return manifest
