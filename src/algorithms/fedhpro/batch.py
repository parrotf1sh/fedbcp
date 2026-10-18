"""Deterministic DSA seed derivation and metrics-only helpers."""
from contextlib import contextmanager
import hashlib
import json
import time


def derive_dsa_seed(train_seed, round_idx, client_idx, repeat_idx, label):
    """Stable v1 mapping; indices are zero-based, independent of wall clock/hash()."""
    values = [train_seed, round_idx, client_idx, repeat_idx, label]
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("DSA seed context must contain nonnegative integers")
    payload = json.dumps(["fedhpro-dsa-v1"] + values, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big") % (2 ** 31)


def sparse_payload_bytes(payload):
    from ..fedavg.reporting import tensor_bytes
    return tensor_bytes(payload) + 8 * len(payload)  # int64 class IDs.


@contextmanager
def stage_timer(metrics, name, device, enabled):
    if not enabled:
        yield
        return
    from ..fedavg.reporting import synchronize
    synchronize(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        synchronize(device)
        metrics[name] = metrics.get(name, 0.0) + time.perf_counter() - started
