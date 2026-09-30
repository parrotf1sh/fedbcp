"""Deterministic, bounded label partitioning with an algorithm-independent cache.

Only sample indices and metadata are persisted; never images or model weights.
This module imports NumPy, but not torch, so independent splits can use CPU workers.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
import zipfile

import numpy as np


DEFAULT_SEED = 19940817
DEFAULT_MAX_ATTEMPTS = 128
DEFAULT_MIN_SAMPLES = 10
DEFAULT_POLICY = "repair"
FORMAT_VERSION = 1


def canonical_partition(partition):
    method = partition["method"]
    if method == "lda":
        result = {"method": method, "alpha": float(partition["alpha"]),
                  "min_samples": int(partition.get("min_samples", DEFAULT_MIN_SAMPLES)),
                  "max_attempts": int(partition.get("max_attempts", DEFAULT_MAX_ATTEMPTS)),
                  "insufficient_policy": partition.get("insufficient_policy", DEFAULT_POLICY)}
        if not np.isfinite(result["alpha"]) or result["alpha"] <= 0:
            raise ValueError("LDA alpha must be finite and positive")
        if result["min_samples"] < 1 or result["max_attempts"] < 1:
            raise ValueError("LDA min_samples/max_attempts must be positive")
        if result["insufficient_policy"] not in ("strict", "repair"):
            raise ValueError("insufficient_policy must be strict or repair")
        return result
    if method == "sharding":
        return {"method": method, "shard_per_user": int(partition["shard_per_user"])}
    raise ValueError("Shared partition cache supports lda and sharding, got " + str(method))


def label_digest(targets):
    return hashlib.sha256(np.asarray(targets, dtype="<i8").tobytes()).hexdigest()


def cache_identity(dataset_name, targets, n_clients, partition, seed=DEFAULT_SEED, test_targets=None):
    partition = canonical_partition(partition)
    identity = {"format": FORMAT_VERSION, "dataset": dataset_name,
                "partition": partition, "seed": int(seed), "n_clients": int(n_clients),
                "train_count": len(targets), "train_labels_sha256": label_digest(targets),
                "test_count": len(test_targets) if test_targets is not None else 0,
                "test_labels_sha256": label_digest(test_targets) if test_targets is not None else None,
                "implementation": "bounded_lda_v1" if partition["method"] == "lda" else "legacy_sharding_v1",
                "rng": "numpy.RandomState-MT19937"}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return key, identity


def _validate_labels(targets, n_clients):
    targets = np.asarray(targets, dtype=np.int64)
    if targets.ndim != 1 or not len(targets) or n_clients < 1:
        raise ValueError("Expected nonempty 1-D targets and a positive client count")
    classes = np.unique(targets)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Class labels must be contiguous integers starting at zero")
    return targets, classes


def _candidate(class_indices, n_clients, alpha, rng, total):
    chunks = [[] for _ in range(n_clients)]
    sizes = np.zeros(n_clients, dtype=np.int64)
    for source in class_indices:
        indices = source.copy()
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        proportions *= sizes < total / n_clients
        denominator = proportions.sum()
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Numerically invalid Dirichlet draw; increase alpha or check client count")
        cuts = (np.cumsum(proportions / denominator) * len(indices)).astype(int)[:-1]
        parts = np.split(indices, cuts)
        for client, part in enumerate(parts):
            chunks[client].append(part)
            sizes[client] += len(part)
    return chunks, sizes


def lda_indices(targets, n_clients, partition, seed=DEFAULT_SEED, progress=True):
    targets, classes = _validate_labels(targets, n_clients)
    settings = canonical_partition(partition)
    minimum = settings["min_samples"]
    if len(targets) < n_clients * minimum:
        raise ValueError("Not enough samples to give every client min_samples={}".format(minimum))
    rng = np.random.RandomState(seed)
    class_indices = [np.flatnonzero(targets == c) for c in classes]
    best, best_sizes, best_deficit, selected = None, None, len(targets) + 1, 0
    for attempt in range(1, settings["max_attempts"] + 1):
        chunks, sizes = _candidate(class_indices, n_clients, settings["alpha"], rng, len(targets))
        deficit = int(np.maximum(minimum - sizes, 0).sum())
        if deficit < best_deficit:
            best = [np.concatenate(row).tolist() for row in chunks]
            best_sizes, best_deficit, selected = sizes.copy(), deficit, attempt
        if deficit == 0:
            break
        if progress and (attempt == 1 or attempt % 32 == 0 or attempt == settings["max_attempts"]):
            print("[partition] LDA attempt {}/{}: min={}, below_min={}, best_deficit={}".format(
                attempt, settings["max_attempts"], sizes.min(), int((sizes < minimum).sum()), best_deficit), flush=True)
    metadata = {"attempts": attempt, "selected_attempt": selected,
                "minimum_before_repair": int(best_sizes.min()),
                "clients_below_minimum_before_repair": int((best_sizes < minimum).sum()),
                "repaired_samples": 0}
    if best_deficit:
        if settings["insufficient_policy"] == "strict":
            raise RuntimeError("LDA rejected all {} attempts (alpha={}, clients={}, min_samples={}); "
                               "best candidate needs {} transferred samples. No split was cached. "
                               "Use insufficient_policy='repair' only if all compared algorithms adopt it.".format(
                                   attempt, settings["alpha"], n_clients, minimum, best_deficit))
        # Each transfer fills exactly one missing sample and never empties a donor.
        # Global class counts, sample uniqueness and total dataset size are preserved.
        for recipient in np.flatnonzero(best_sizes < minimum):
            while best_sizes[recipient] < minimum:
                donor = int(best_sizes.argmax())
                if best_sizes[donor] <= minimum:
                    raise RuntimeError("Cannot repair an infeasible partition")
                position = rng.randint(len(best[donor]))
                index = best[donor][position]
                best[donor][position] = best[donor][-1]
                best[donor].pop()
                best[recipient].append(index)
                best_sizes[donor] -= 1
                best_sizes[recipient] += 1
        metadata["repaired_samples"] = best_deficit
        if progress:
            print("[partition] Explicit minimum-size repair: moved {}/{} samples ({:.4%}); "
                  "use this same cached split for every algorithm.".format(
                      best_deficit, len(targets), best_deficit / len(targets)), flush=True)
    result = {}
    for client, indices in enumerate(best):
        rng.shuffle(indices)
        result[client] = np.asarray(indices, dtype=np.int64)
    metadata["minimum_after_repair"] = int(best_sizes.min())
    metadata["repaired_fraction"] = best_deficit / len(targets)
    return result, metadata


def _shard_indices(targets, n_clients, shards, rng, python_rng, assignment=None):
    targets, classes = _validate_labels(targets, n_clients)
    per_class, remainder = divmod(shards * n_clients, len(classes))
    if shards < 1 or remainder or per_class < 1:
        raise ValueError("n_clients * shard_per_user must be divisible by class count")
    shards_by_class = {}
    for cls in classes:
        indices = np.flatnonzero(targets == cls)
        if len(indices) < per_class:
            raise ValueError("Sharding would create empty class shards")
        # Exactly the original remainder assignment, not np.array_split.
        leftover_count = len(indices) % per_class
        main = indices[:-leftover_count] if leftover_count else indices
        parts = list(main.reshape(per_class, -1))
        if leftover_count:
            for i, extra in enumerate(indices[-leftover_count:]):
                parts[i] = np.concatenate((parts[i], [extra]))
        shards_by_class[int(cls)] = parts
    if assignment is None:
        assignment = list(range(len(classes))) * per_class
        python_rng.shuffle(assignment)
        assignment = np.asarray(assignment).reshape(n_clients, shards)
    result = {}
    for client in range(n_clients):
        selected = []
        for cls in assignment[client]:
            candidates = shards_by_class[int(cls)]
            position = rng.choice(len(candidates), replace=False)
            selected.append(candidates.pop(position))
        result[client] = np.concatenate(selected).astype(np.int64)
    return result, assignment


def _pack(mapping, n_clients):
    if mapping is None:
        return np.empty(0, dtype=np.int64), np.zeros(1, dtype=np.int64)
    offsets = np.r_[0, np.cumsum([len(mapping[i]) for i in range(n_clients)])]
    return np.concatenate([mapping[i] for i in range(n_clients)]).astype(np.int64), offsets


def _unpack(indices, offsets, total, n_clients, minimum=1):
    if indices.dtype.kind not in "iu" or offsets.dtype.kind not in "iu":
        raise ValueError("Cached indices must be integers")
    if (indices.ndim != 1 or offsets.shape != (n_clients + 1,) or offsets[0] != 0
            or offsets[-1] != total or len(indices) != total or np.any(np.diff(offsets) < minimum)
            or not np.array_equal(np.sort(indices), np.arange(total))):
        raise ValueError("Cached split has missing/duplicate/out-of-range samples or invalid client sizes")
    return {i: indices[offsets[i]:offsets[i + 1]].copy() for i in range(n_clients)}


def _payload_digest(arrays):
    hasher = hashlib.sha256()
    for array in arrays:
        array = np.asarray(array, dtype="<i8")
        hasher.update(np.asarray(array.shape, dtype="<i8").tobytes())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def _read(path, identity, key):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        if metadata["identity"] != identity or metadata["cache_key"] != key:
            raise ValueError("Cache identity mismatch")
        arrays = [data[name] for name in ("train_indices", "train_offsets", "test_indices", "test_offsets")]
    if _payload_digest(arrays) != metadata["indices_sha256"]:
        raise ValueError("Cache index checksum mismatch")
    n_clients = identity["n_clients"]
    minimum = identity["partition"].get("min_samples", 1)
    train = _unpack(arrays[0], arrays[1], identity["train_count"], n_clients, minimum)
    test = _unpack(arrays[2], arrays[3], identity["test_count"], n_clients) if identity["test_count"] else None
    return train, test, metadata


def load_or_create_partition(targets, n_clients, partition, dataset_name, cache_dir,
                             seed=DEFAULT_SEED, test_targets=None, progress=True):
    """Lock, validate and reuse one shared split, independent of training seed/model."""
    import fcntl
    started = time.perf_counter()
    partition = canonical_partition(partition)
    targets, _ = _validate_labels(targets, n_clients)
    if partition["method"] == "lda":
        test_targets = None  # Global test set is never repartitioned for LDA.
    elif test_targets is None:
        raise ValueError("Sharding cache must include the paired test labels")
    key, identity = cache_identity(dataset_name, targets, n_clients, partition, seed, test_targets)
    directory = Path(cache_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (dataset_name + "_" + partition["method"] + "_" + key[:24] + ".npz")
    with path.with_suffix(".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if progress:
                print("[partition] Waiting for another process to prepare " + str(path), flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            try:
                train, test, metadata = _read(path, identity, key)
            except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
                quarantined = path.with_suffix(".invalid-" + str(time.time_ns()) + ".npz")
                path.replace(quarantined)
                if progress:
                    print("[partition] Invalid cache quarantined: {} ({})".format(quarantined, exc), flush=True)
            else:
                metadata.update(cache_hit=True, cache_path=str(path), elapsed_seconds=time.perf_counter() - started)
                if progress:
                    print("[partition] HIT {} ({:.3f}s)".format(path, metadata["elapsed_seconds"]), flush=True)
                return train, test, metadata
        if progress:
            print("[partition] MISS {} seed={} settings={}".format(path, seed, partition), flush=True)
        if partition["method"] == "lda":
            train, details = lda_indices(targets, n_clients, partition, seed, progress)
            test = None
        else:
            rng, python_rng = np.random.RandomState(seed), random.Random(seed)
            train, assignment = _shard_indices(targets, n_clients, partition["shard_per_user"], rng, python_rng)
            test, _ = _shard_indices(test_targets, n_clients, partition["shard_per_user"], rng, python_rng, assignment)
            details = {"attempts": 1, "repaired_samples": 0, "repaired_fraction": 0.0}
        train_indices, train_offsets = _pack(train, n_clients)
        test_indices, test_offsets = _pack(test, n_clients)
        arrays = [train_indices, train_offsets, test_indices, test_offsets]
        _unpack(train_indices, train_offsets, len(targets), n_clients, partition.get("min_samples", 1))
        if test is not None:
            _unpack(test_indices, test_offsets, len(test_targets), n_clients)
        metadata = {"identity": identity, "cache_key": key, "indices_sha256": _payload_digest(arrays),
                    "numpy_version": np.__version__, "details": details}
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".partition-", suffix=".tmp", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(temporary, train_indices=train_indices, train_offsets=train_offsets,
                                    test_indices=test_indices, test_offsets=test_offsets,
                                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        metadata.update(cache_hit=False, cache_path=str(path), elapsed_seconds=time.perf_counter() - started)
        if progress:
            print("[partition] SAVED {} ({:.3f}s)".format(path, metadata["elapsed_seconds"]), flush=True)
        return train, test, metadata
