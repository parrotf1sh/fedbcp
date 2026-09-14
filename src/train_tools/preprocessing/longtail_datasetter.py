"""Opt-in image loaders for the shared longtail_split manifest protocol."""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .longtail_split import load_or_create_manifest, split_spec, _digest
from .cifar10.datasets import CIFAR10_truncated
from .cifar100.datasets import CIFAR100_truncated
from .tinyimagenet.datasets import TinyImageNet_Truncated
from .cifar10.loader import _data_transforms_cifar10
from .cifar100.loader import _data_transforms_cifar100
from .tinyimagenet.loader import _data_transforms_tinyimagenet

__all__ = ["longtail_data_distributer"]


class IndexedView(Dataset):
    """Share underlying images; keep subset labels aligned (also for TinyImageNet)."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.targets = np.asarray(dataset.targets)[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, label = self.dataset[self.indices[position]]
        return self.transform(image), int(label)


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def longtail_data_distributer(root, dataset_name, batch_size, n_clients, partition,
                             longtail=None, oracle_size=0, oracle_batch_size=None):
    if oracle_size:
        raise ValueError("The longtail protocol does not provide extra oracle training data")
    options = dict(longtail or {})
    allowed = {"imbalance_factor", "validation_fraction", "split_seed", "class_order_seed",
               "min_client_samples", "max_attempts", "manifest_path", "manifest_dir",
               "num_workers", "download"}
    if set(options) - allowed:
        raise ValueError("Unknown longtail options: {}".format(sorted(set(options) - allowed)))
    if dataset_name not in ("cifar10", "cifar100", "tinyimagenet"):
        raise ValueError("longtail supports cifar10, cifar100, tinyimagenet")
    if batch_size < 1 or int(batch_size) != batch_size:
        raise ValueError("batch_size must be a positive integer")
    workers = int(options.get("num_workers", 0))
    if workers < 0:
        raise ValueError("num_workers must be nonnegative")
    dataset_root = os.path.join(root, dataset_name)
    if dataset_name == "tinyimagenet":
        source = TinyImageNet_Truncated(dataset_root, train=True)
        test_source = TinyImageNet_Truncated(dataset_root, train=False)
        transforms = _data_transforms_tinyimagenet()
    else:
        cls = CIFAR10_truncated if dataset_name == "cifar10" else CIFAR100_truncated
        source = cls(dataset_root, train=True, download=options.get("download", False))
        test_source = cls(dataset_root, train=False, download=options.get("download", False))
        transforms = (_data_transforms_cifar10() if dataset_name == "cifar10"
                      else _data_transforms_cifar100())
    train_transform, evaluation_transform = transforms
    split_options = {key: options[key] for key in (
        "imbalance_factor", "validation_fraction", "split_seed", "class_order_seed",
        "min_client_samples", "max_attempts") if key in options}
    split_options.update(n_clients=n_clients, partition=dict(partition))
    spec = split_spec(**split_options)
    manifest_path = options.get("manifest_path") or os.path.join(
        options.get("manifest_dir", "./splits"),
        "{}-{}.json".format(dataset_name, _digest(spec)[:16]))
    manifest = load_or_create_manifest(manifest_path, source.targets.tolist(), **split_options)
    seed = spec["split_seed"]

    def loader(dataset, indices, transform, shuffle, offset):
        generator = torch.Generator()
        generator.manual_seed(seed + offset)
        return DataLoader(IndexedView(dataset, indices, transform), batch_size=batch_size,
                          shuffle=shuffle, num_workers=workers, drop_last=False,
                          worker_init_fn=_seed_worker, generator=generator)

    local = {}
    counts = np.asarray(manifest["data_map"], dtype=np.int64)
    for client, indices in enumerate(manifest["client_indices"]):
        local[client] = dict(
            datasize=len(indices),
            train=loader(source, indices, train_transform, True, 10 + client),
            calibration=loader(source, indices, evaluation_transform, False, 100000 + client),
            train_eval=loader(source, indices, evaluation_transform, False, 200000 + client),
            test=None, dist=counts[client] / counts[client].sum(),
            class_counts=counts[client].copy())
    validation = (loader(source, manifest["validation_indices"], evaluation_transform,
                         False, 300001) if manifest["validation_indices"] else None)
    global_loaders = dict(
        train=loader(source, manifest["train_indices"], train_transform, True, 300000),
        validation=validation,
        test=loader(test_source, range(len(test_source)), evaluation_transform, False, 300002))
    print(">>> Longtail split {}: train={}, validation={}, IF={:.3f}, missing client-class pairs={}".format(
        manifest["sha256"][:12], len(manifest["train_indices"]),
        len(manifest["validation_indices"]), manifest["actual_imbalance_factor"],
        int((counts == 0).sum())))
    return {
        "global": global_loaders, "local": local, "data_map": counts,
        "num_classes": len(manifest["global_counts"]),
        "global_class_counts": np.asarray(manifest["global_counts"], dtype=np.int64),
        "split_manifest": manifest, "split_manifest_path": os.path.abspath(manifest_path)}
