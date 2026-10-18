import copy
import time
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..BaseClientTrainer import BaseClientTrainer
from .augmentation import AugmentParams, diff_augment
from .criterion import HyperPrototypeLoss
from .batch import derive_dsa_seed
from .utils import classifier_module, forward_with_features, preserve_evaluation_rng

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.criterion = HyperPrototypeLoss(cfg["hp_info_t"])
        self.augmentation = AugmentParams(cfg["augmentation"])
        self.feature_dim = classifier_module(self.model).in_features
        self.hyper_prototypes = None
        self.k_margin = 0.0
        self.dsa_context = None

    def download_global(self, server_weights, server_optimizer, hyper_prototypes):
        self.model.load_state_dict(server_weights)
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.optimizer.load_state_dict(copy.deepcopy(server_optimizer))
        # Source clears state for every client participation.
        self.optimizer.state.clear()
        self.hyper_prototypes = hyper_prototypes.detach().clone().to(self.device)
        self.k_margin = 0.0

    def _dsa_seed(self, repeat_idx, label):
        if self.cfg["dsa_seed_mode"] == "wall_clock":
            return int(time.time() * 1000) % 100000
        if self.dsa_context is None:
            raise RuntimeError("Set FedHPro training seed, round and client before deterministic DSA")
        return derive_dsa_seed(*self.dsa_context, repeat_idx, label)

    def compute_gradient(self, global_meta_params, class_counts):
        """Collect real feature/head gradients BEFORE any local optimization."""
        dataset = self.trainloader.dataset
        classes = [index for index, count in enumerate(class_counts) if count > 0]
        if len(dataset) == 0 or not classes:
            raise ValueError("FedHPro requires nonempty client data and class counts")
        # Keep source materialization and repeat sampling semantics, including
        # the separate dataset traversal for labels. Never resplit client data.
        images_all = [dataset[index][0].unsqueeze(0) for index in range(len(dataset))]
        labels_all = [dataset[index][1] for index in range(len(dataset))]
        images_all = torch.cat(images_all, dim=0).to(self.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=self.device)
        observed = torch.bincount(labels_all, minlength=self.num_classes)
        expected = torch.as_tensor(class_counts, device=self.device, dtype=torch.long)
        if observed.shape != expected.shape or not torch.equal(observed, expected):
            raise ValueError("FedHPro client labels do not match the shared data_map")
        indices_class = {
            label: (labels_all == label).nonzero(as_tuple=True)[0] for label in classes
        }
        self.model.load_state_dict(global_meta_params)
        self.model.to(self.device)
        self.model.eval()
        classifier = classifier_module(self.model)
        classifier.train()
        classifier_params = list(classifier.parameters())
        feature_gradients = {label: [] for label in classes}
        meta_gradients = {label: [] for label in classes}

        for repeat_idx in range(self.cfg["repeat_num"]):
            batch_images, batch_ranges = [], []
            start = 0
            for label in classes:
                indices = indices_class[label]
                shuffled = torch.randperm(indices.numel(), device=self.device)
                images = images_all[indices[shuffled[:self.cfg["batch_real"]]]]
                if self.cfg["method"] == "DSA":
                    # Only seed provenance changes; Siamese DSA and RNG resets stay intact.
                    seed = self._dsa_seed(repeat_idx, label)
                    images = diff_augment(
                        images, self.cfg["dsa_strategy"], seed=seed,
                        param=self.augmentation,
                    )
                end = start + images.shape[0]
                batch_images.append(images)
                batch_ranges.append((label, start, end))
                start = end
            with torch.no_grad():
                _, feature_batch = forward_with_features(
                    self.model, torch.cat(batch_images, dim=0)
                )
            for label, start, end in batch_ranges:
                targets = torch.full(
                    (end - start,), label, device=self.device, dtype=torch.long
                )
                features = feature_batch[start:end].detach().requires_grad_(True)
                logits = classifier(features)
                loss = F.cross_entropy(logits, targets)
                gradients = torch.autograd.grad(loss, [features] + classifier_params)
                feature_gradients[label].append(gradients[0].detach().clone())
                meta_gradients[label].append([
                    gradient.detach().clone() for gradient in gradients[1:]
                ])

        feature_average, meta_average = {}, {}
        for label in classes:
            feature_average[label] = torch.stack(feature_gradients[label], dim=0).mean(dim=(0, 1))
            gradients = meta_gradients[label]
            weight = 1.0 / len(gradients)
            meta_average[label] = []
            for param_idx in range(len(gradients[0])):
                values = [repeat[param_idx] * weight for repeat in gradients]
                meta_average[label].append(sum(values))
        return feature_average, meta_average

    @torch.no_grad()
    def compute_margin(self):
        if self.cfg["use_margin"]:
            sums = torch.zeros(self.num_classes, self.feature_dim, device=self.device)
            counts = torch.zeros(self.num_classes, device=self.device)
            loader = DataLoader(
                self.trainloader.dataset, batch_size=self.cfg["margin_batch_size"],
                shuffle=False, num_workers=0,
            )
            # Deliberately retain train mode and recompute inside EVERY batch.
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)
                _, features = forward_with_features(self.model, data)
                sums.index_add_(0, targets, features)
                counts.index_add_(0, targets, torch.ones_like(targets, dtype=torch.float))
            valid = counts > 0
            prototypes = sums[valid] / counts[valid].unsqueeze(1)
            products = torch.matmul(prototypes, prototypes.T)
            off_diagonal = products.clone()
            off_diagonal.fill_diagonal_(0)
            self.k_margin = torch.sum(off_diagonal / ((self.num_classes - 1) ** 2))

    def train(self):
        if self.hyper_prototypes is None:
            raise RuntimeError("Download FedHPro global weights and prototypes before training")
        self.model.train()
        self.model.to(self.device)
        hp_stack = self.hyper_prototypes.reshape(
            self.num_classes, self.cfg["length_of_hp"], self.feature_dim
        )
        hp_class_norm_mean = F.normalize(hp_stack, dim=2).mean(dim=1)
        hp_avg = hp_stack.mean(dim=1)
        loss_results = {name: [] for name in ("ce_loss", "hpcl_loss", "hpal_loss", "total_loss")}
        totals = {"seen": 0, "correct": 0, "loss_sum": 0.0}
        totals.update({name + "_sum": 0.0 for name in loss_results})
        for _ in range(self.local_epochs):
            for data, targets in self.trainloader:
                data, targets = data.to(self.device), targets.to(self.device)
                # Shared loader already supplies the project's train transforms.
                logits, features = forward_with_features(self.model, data)
                loss_ce = F.cross_entropy(logits, targets)
                if self.cfg["use_margin"]:
                    self.compute_margin()
                loss, losses = self.criterion(
                    loss_ce, features, targets, hp_class_norm_mean, hp_avg, self.k_margin
                )
                if self.cfg["metrics_only"] and not math.isfinite(loss.detach().item()):
                    raise FloatingPointError("Non-finite FedHPro local loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                if self.cfg["metrics_only"]:
                    count = targets.numel()
                    totals["seen"] += count
                    totals["correct"] += (logits.detach().argmax(1) == targets).sum().item()
                    totals["loss_sum"] += loss.detach().item() * count
                    for name, value in losses.items():
                        totals[name + "_sum"] += value.detach().item() * count
                else:
                    for name, value in losses.items():
                        loss_results[name].append(value.detach().item())
        if self.cfg["metrics_only"]:
            if not totals["seen"]:
                raise ValueError("FedHPro client training produced no batches")
            return totals, self.datasize
        if not loss_results["total_loss"]:
            raise ValueError("FedHPro client training produced no batches")
        with preserve_evaluation_rng([self.trainloader, self.testloader]):
            local_results = self._get_local_stats()
        local_results.update({name: sum(values) / len(values) for name, values in loss_results.items()})
        return local_results, self.datasize

    def reset(self):
        super().reset()
        self.dsa_context = None
        self.hyper_prototypes = None
        self.k_margin = 0.0
