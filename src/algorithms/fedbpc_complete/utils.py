"""Versioned raw-feature moments and deterministic training-data views."""
import copy
import random
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler


@contextmanager
def isolated_randomness(loader, device):
    py_state, np_state = random.getstate(), np.random.get_state()
    generators = []
    for owner in (loader, getattr(loader, "sampler", None)):
        generator = getattr(owner, "generator", None)
        if generator is not None and all(generator is not item[0] for item in generators):
            generators.append((generator, generator.get_state()))
    if getattr(loader, "persistent_workers", False):
        raise ValueError("Statistics loaders must not use persistent random workers")
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            for generator, state in generators:
                generator.set_state(state)


def make_statistics_loader(train_loader, test_loader, seed, num_workers=0):
    """Reuse TRAIN images with the test transform; never read test samples."""
    dataset = copy.copy(train_loader.dataset)
    if not hasattr(dataset, "transform") or not hasattr(test_loader.dataset, "transform"):
        raise ValueError(
            "Supply an explicit calibration loader for datasets without a transform attribute")
    dataset.transform = copy.deepcopy(test_loader.dataset.transform)
    return DataLoader(
        dataset, batch_size=train_loader.batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, collate_fn=train_loader.collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def collect_moments(model, loader, num_classes, device, encoder_version):
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("Statistics must visit every training sample exactly once in order")
    modes = [(module, module.training) for module in model.modules()]
    sums = torch.zeros(num_classes, model.classifier.in_features, dtype=torch.float64)
    square_sums = torch.zeros_like(sums)
    counts = torch.zeros(num_classes, dtype=torch.long)
    try:
        with isolated_randomness(loader, device):
            model.eval()
            model.to(device)
            for images, targets in loader:
                targets = targets.to(device, dtype=torch.long)
                _, features = model(images.to(device), get_features=True)
                if features.ndim != 2 or features.shape[1] != sums.shape[1]:
                    raise ValueError("get_features must return the input to the linear classifier")
                if not torch.isfinite(features).all():
                    raise FloatingPointError("Non-finite feature moments")
                if targets.numel() and (targets.min() < 0 or targets.max() >= num_classes):
                    raise ValueError("Invalid class index")
                labels = targets.cpu()
                raw = features.detach().cpu().double()
                sums.index_add_(0, labels, raw)
                square_sums.index_add_(0, labels, raw.square())
                counts += torch.bincount(labels, minlength=num_classes)
    finally:
        for module, training in modes:
            module.training = training
    # Keep accumulation precision through transmission: subtracting raw moments
    # in float32 can erase small within-class variances of large feature values.
    return {"feature_sums": sums, "feature_square_sums": square_sums, "class_counts": counts,
            "encoder_version": int(encoder_version)}


class PrototypeBank:
    def __init__(self, num_classes, feature_dim, max_age):
        self.means = torch.zeros(num_classes, feature_dim)
        self.variances = torch.zeros(num_classes, feature_dim)
        self.support = torch.zeros(num_classes, dtype=torch.long)
        self.valid = torch.zeros(num_classes, dtype=torch.bool)
        self.age = torch.zeros(num_classes, dtype=torch.long)
        self.max_age = max_age

    @property
    def reliability(self):
        return self.valid.float() / (1.0 + self.age.float())

    @torch.no_grad()
    def update(self, payloads, encoder_version):
        sums = torch.zeros_like(self.means, dtype=torch.float64)
        square_sums = torch.zeros_like(sums)
        counts = torch.zeros_like(self.age)
        contributors = torch.zeros_like(self.age)
        for payload in payloads:
            local_sum = payload["feature_sums"].detach().cpu()
            local_count = payload["class_counts"].detach().cpu()
            if payload["encoder_version"] != encoder_version:
                raise ValueError("Cannot aggregate moments from different encoder versions")
            local_squares = payload["feature_square_sums"].detach().cpu()
            if (local_sum.shape != self.means.shape or local_count.shape != self.age.shape
                    or local_squares.shape != self.means.shape):
                raise ValueError("Invalid prototype payload shape")
            if not all(torch.isfinite(x).all() for x in (local_sum, local_count, local_squares)):
                raise ValueError("Non-finite prototype payload")
            if (local_count < 0).any() or not torch.equal(local_count, local_count.long()):
                raise ValueError("Prototype counts must be nonnegative integers")
            if (local_sum[local_count == 0].count_nonzero()
                    or local_squares[local_count == 0].count_nonzero()):
                raise ValueError("Empty classes must have zero feature moments")
            if (local_squares < 0).any():
                raise ValueError("Squared feature sums must be nonnegative")
            sums += local_sum.double()
            square_sums += local_squares.double()
            counts += local_count.long()
            contributors += (local_count > 0).long()
        self.age[self.valid] += 1
        observed = counts > 0
        current = (sums / counts.clamp_min(1).unsqueeze(1)).float()
        # Pool raw second moments, including BETWEEN-client mean differences.
        centered = square_sums - sums.square() / counts.clamp_min(1).unsqueeze(1)
        variance = (centered.clamp_min(0) / (counts - 1).clamp_min(1).unsqueeze(1)).float()
        variance[counts <= 1] = 0  # Unknown scatter, not evidence of zero population variance.
        # Zero mean is a legitimate distribution prototype (e.g. symmetric data).
        self.valid[observed] = True
        self.means[observed] = current[observed]
        self.variances[observed] = variance[observed]
        self.support[observed] = counts[observed]
        self.age[observed] = 0
        expired = self.valid & (self.age > self.max_age)
        self.valid[expired] = False
        self.means[~self.valid] = 0
        self.variances[~self.valid] = 0
        self.support[~self.valid] = 0
        self.age[~self.valid] = 0
        return {"counts": counts, "contributors": contributors,
                "updated": observed, "expired": expired}

    def state_dict(self):
        return {key: getattr(self, key).clone()
                for key in ("means", "variances", "support", "valid", "age")}
