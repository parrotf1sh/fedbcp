"""Opt-in observational diagnostics. No probe state enters model aggregation.

Validation cohorts are fixed by ordered sample positions at their origin round.
Shadow clients replay the actual client's pre-update RNG and loader states.
"""

import copy
import hashlib
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler

from .ClientTrainer import ClientTrainer
from .utils import classifier_module, sync_device


class RandomSnapshot:
    def __init__(self, *loaders, device="cpu"):
        self.python = random.getstate()
        self.numpy = np.random.get_state()
        self.cpu = torch.get_rng_state()
        device = torch.device(device)
        self.cuda = None
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            self.cuda = (index, torch.cuda.get_rng_state(index))
        self.generators = []
        seen = set()
        for loader in loaders:
            if loader is None:
                continue
            if getattr(loader, "persistent_workers", False):
                raise ValueError("Transfer diagnostics cannot replay persistent DataLoader workers")
            for owner in (loader, getattr(loader, "sampler", None),
                          getattr(getattr(loader, "batch_sampler", None), "sampler", None)):
                generator = getattr(owner, "generator", None)
                if generator is not None and id(generator) not in seen:
                    self.generators.append((generator, generator.get_state()))
                    seen.add(id(generator))

    def restore(self):
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            index, state = self.cuda
            torch.cuda.set_rng_state(state, index)
        for generator, state in self.generators:
            generator.set_state(state)


@contextmanager
def isolated_randomness(*loaders, device="cpu"):
    snapshot = RandomSnapshot(*loaders, device=device)
    try:
        yield
    finally:
        snapshot.restore()


def _margin(logits, labels):
    other = logits.clone()
    other.scatter_(1, labels[:, None], float("-inf"))
    return logits.gather(1, labels[:, None]).squeeze(1) - other.max(dim=1).values


def _band_excess(student, teacher, counts, cfg):
    # Same centered, full-output band as RetentionLoss; used without gradients.
    support = cfg["support_scale"] / (counts.to(student) + cfg["support_scale"])
    if cfg["uniform_support"]:
        support = support.mean().expand_as(support)
    tolerance = cfg["band_min"] + (cfg["band_max"] - cfg["band_min"]) * (1 - support)
    student = student - student.mean(dim=1, keepdim=True)
    teacher = teacher - teacher.mean(dim=1, keepdim=True)
    return (torch.abs(student - teacher) - tolerance).clamp_min(0)


def _band_counts(excess, labels, corrected):
    active = excess > 0
    return dict(count=int(corrected.sum().item()),
                all_inactive_count=int((~active.any(dim=1) & corrected).sum().item()),
                true_class_inactive_count=int((~active.gather(1, labels[:, None]).squeeze(1)
                                               & corrected).sum().item()),
                active_output_count=int(active[corrected].sum().item()))


def _band_rates(counts, num_classes):
    n = counts["count"]
    return dict(counts,
                all_inactive_fraction=counts["all_inactive_count"] / n if n else None,
                true_class_inactive_fraction=counts["true_class_inactive_count"] / n if n else None,
                active_output_fraction=counts["active_output_count"] / (n * num_classes) if n else None)


class BatchTrace:
    """Verify actual augmented inputs/order, without saving or uploading inputs."""
    def __init__(self):
        self.digest = hashlib.sha256()
        self.batches = self.examples = 0
        self.seconds = 0.0

    def observe(self, images, labels):
        start = time.perf_counter()
        for tensor in (images, labels):
            self.digest.update(str((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
            self.digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        self.batches += 1
        self.examples += labels.numel()
        self.seconds += time.perf_counter() - start


class TrainingProbe:
    """Use the teacher's existing forward features; no extra backbone forward.

Counts describe augmented batch exposures, including repeated local epochs.
Only diagnostic clients use this hook. No validation samples enter training.
"""
    def __init__(self, teacher, global_model, counts, cfg, device):
        raw_head = classifier_module(global_model)[1]
        self.weight = raw_head.weight.detach().to(device).clone()
        self.bias = None if raw_head.bias is None else raw_head.bias.detach().to(device).clone()
        self.counts = torch.as_tensor(counts, device=device)
        self.cfg, self.device = cfg, device
        self.features = []
        self.handle = classifier_module(teacher)[1].register_forward_pre_hook(self._capture)
        self.examples = 0
        self.seconds = 0.0
        self.totals = {name: dict(count=0, all_inactive_count=0,
                                 true_class_inactive_count=0, active_output_count=0)
                       for name in ("initial_gap", "student_gap")}

    def _capture(self, module, inputs):
        self.features.append(inputs[0].detach())

    @torch.no_grad()
    def observe(self, student, teacher, labels):
        sync_device(self.device)
        start = time.perf_counter()
        if len(self.features) != 1:
            raise ValueError("Transfer probe expects one final linear classifier invocation")
        raw = F.linear(self.features.pop(), self.weight, self.bias)
        corrected = raw.argmax(1).ne(labels) & teacher.argmax(1).eq(labels)
        self.examples += labels.numel()
        for name, logits in (("initial_gap", raw), ("student_gap", student)):
            counts = _band_counts(_band_excess(logits, teacher, self.counts, self.cfg), labels, corrected)
            for key, value in counts.items():
                self.totals[name][key] += value
        sync_device(self.device)
        self.seconds += time.perf_counter() - start

    def close(self):
        self.handle.remove()
        self.features.clear()

    def result(self):
        return dict(examples=self.examples,
                    **{name: _band_rates(counts, self.weight.shape[0])
                       for name, counts in self.totals.items()})


class CorrectionCohort:
    def __init__(self, round_number, before, teacher, labels, groups, cfg):
        self.round = round_number
        self.before, self.teacher, self.labels = before, teacher, labels
        self.groups, self.cfg = groups, cfg
        self.before_correct = before.argmax(1).eq(labels)
        self.teacher_correct = teacher.argmax(1).eq(labels)
        self.corrected = ~self.before_correct & self.teacher_correct
        self.harmed = self.before_correct & ~self.teacher_correct
        self.before_margin = _margin(before, labels)
        self.teacher_margin = _margin(teacher, labels)
        self.aggregate_correct = None
        self.local_logits_sum = torch.zeros_like(before)
        self.local_correct_sum = torch.zeros(len(labels))
        self.local_weight = self.local_clients = 0

    def add_local(self, logits, size):
        self.local_logits_sum.add_(logits, alpha=float(size))
        self.local_correct_sum.add_(logits.argmax(1).eq(self.labels).float(), alpha=float(size))
        self.local_weight += size
        self.local_clients += 1

    def aggregation_gap(self, logits):
        # Compare the actual FedAvg model with ALL participating local models.
        # A logit ensemble is a diagnostic comparator, never a deployed model.
        mean_correct = self.local_correct_sum / self.local_weight
        ensemble_correct = self.local_logits_sum.argmax(1).eq(self.labels).float()
        aggregate_correct = logits.argmax(1).eq(self.labels).float()
        result = dict(clients=self.local_clients, training_samples=self.local_weight)
        for name, mask in (("all", torch.ones_like(self.corrected)), ("corrected", self.corrected)):
            n = int(mask.sum())
            result[name] = dict(
                count=n,
                weighted_local_accuracy=mean_correct[mask].mean().item() if n else None,
                logit_ensemble_accuracy=ensemble_correct[mask].mean().item() if n else None,
                aggregate_accuracy=aggregate_correct[mask].mean().item() if n else None,
                aggregate_minus_weighted_local=(aggregate_correct - mean_correct)[mask].mean().item() if n else None)
        self.local_logits_sum = self.local_correct_sum = None
        return result

    def _summary(self, logits, margin, mask):
        n = int(mask.sum())
        correct = logits.argmax(1).eq(self.labels)
        return dict(count=n,
                    before_accuracy=self.before_correct[mask].float().mean().item() if n else None,
                    teacher_accuracy=self.teacher_correct[mask].float().mean().item() if n else None,
                    accuracy=correct[mask].float().mean().item() if n else None,
                    mean_margin_gain=(margin - self.before_margin)[mask].mean().item() if n else None,
                    mean_teacher_margin_gain=(self.teacher_margin - self.before_margin)[mask].mean().item() if n else None)

    def score(self, logits, local_counts=None):
        margin = _margin(logits, self.labels)
        masks = dict(all=torch.ones_like(self.corrected), corrected=self.corrected, harmed=self.harmed)
        for group, indices in self.groups.items():
            member = torch.zeros_like(self.corrected)
            for label in indices:
                member |= self.labels.eq(label)
            masks["corrected_" + group] = self.corrected & member
        if local_counts is not None:
            n = torch.as_tensor(local_counts)[self.labels]
            for name, mask in (("missing", n == 0),
                               ("low_support", (n > 0) & (n < self.cfg["support_scale"])),
                               ("supported", n >= self.cfg["support_scale"])):
                masks[name] = mask
                masks["corrected_" + name] = self.corrected & mask
        return {name: self._summary(logits, margin, mask) for name, mask in masks.items()}

    def band(self, local_counts):
        counts = _band_counts(_band_excess(self.before, self.teacher,
                                         torch.as_tensor(local_counts), self.cfg),
                              self.labels, self.corrected)
        return _band_rates(counts, self.before.shape[1])

    def followup(self, logits):
        result = self.score(logits)
        mask = self.corrected & self.aggregate_correct
        n = int(mask.sum())
        result["aggregate_correct_endpoint"] = dict(
            count=n, accuracy=logits.argmax(1).eq(self.labels)[mask].float().mean().item() if n else None)
        return result


class TransferDiagnostics:
    def __init__(self, loader, groups, cfg, global_counts, local_epochs, device):
        if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
            raise ValueError("Transfer diagnostics require ordered, complete validation batches")
        if getattr(loader, "persistent_workers", False):
            raise ValueError("Transfer diagnostics require non-persistent validation workers")
        self.loader, self.groups, self.cfg = loader, groups, cfg
        self.global_counts, self.local_epochs = global_counts, local_epochs
        self.num_classes, self.device = len(global_counts), device
        self.labels = None
        self.pending = []
        self.started = self.completed_followups = 0

    @torch.no_grad()
    def predict(self, model):
        # Preserve submodule modes and loader RNG, including dropout and workers.
        modes = [(module, module.training) for module in model.modules()]
        previous_device = next(model.parameters()).device
        outputs, targets = [], []
        with isolated_randomness(self.loader, device=self.device):
            try:
                model.to(self.device)
                model.eval()
                for images, labels in self.loader:
                    logits = model(images.to(self.device))
                    if (logits.ndim != 2 or logits.shape[1] != self.num_classes
                            or not torch.isfinite(logits).all()):
                        raise FloatingPointError("Invalid logits in transfer diagnostics")
                    outputs.append(logits.detach().cpu())
                    targets.append(labels.detach().cpu().long())
            finally:
                model.to(previous_device)
                for module, mode in modes:
                    module.training = mode
        if not targets:
            raise ValueError("Empty transfer validation loader")
        labels = torch.cat(targets)
        if self.labels is None:
            self.labels = labels
        elif not torch.equal(self.labels, labels):
            raise ValueError("Validation sample order changed during transfer tracking")
        return torch.cat(outputs)

    def start(self, round_number, model, teacher):
        before, calibrated = self.predict(model), self.predict(teacher)
        self.started += 1
        return CorrectionCohort(round_number, before, calibrated, self.labels, self.groups, self.cfg)

    def shadows(self, cohort, global_model, actual_logits, initial_state, optimizer_state,
                data, round_idx, replay, actual_trace):
        result = dict(full=cohort.score(actual_logits, data["class_counts"]))
        if not self.cfg["transfer_shadow_branches"]:
            return result
        # One transient student and one raw teacher; neither can modify actual state.
        with isolated_randomness(data["train"], self.loader, device=self.device):
            raw_teacher = copy.deepcopy(global_model).to(self.device)
            raw_teacher.eval()
            try:
                for name in ("balanced_only", "raw_teacher_band"):
                    cfg = dict(self.cfg)
                    cfg["teacher_mode"] = "none" if name == "balanced_only" else "global"
                    if name == "balanced_only":
                        cfg["retention_mode"] = "none"
                    student = ClientTrainer(global_counts=self.global_counts, cfg=cfg,
                                            algo_params=cfg, model=copy.deepcopy(global_model).cpu(),
                                            local_epochs=self.local_epochs, device=self.device,
                                            num_classes=self.num_classes)
                    try:
                        trace = BatchTrace()
                        # Restore AFTER constructing models/optimizers, just before SGD.
                        replay.restore()
                        student.train_client(initial_state, optimizer_state, data,
                                             None if name == "balanced_only" else raw_teacher, round_idx,
                                             batch_trace=trace)
                        if trace.digest.digest() != actual_trace.digest.digest():
                            raise RuntimeError("Transfer shadow batch replay differs from actual training. "
                                               "Use the longtail loaders with num_workers=0 and deterministic transforms' RNGs.")
                        result[name] = cohort.score(self.predict(student.model), data["class_counts"])
                        result[name]["replay_batches_match"] = 1
                    finally:
                        student.model.cpu()
                        del student
            finally:
                raw_teacher.cpu()
                del raw_teacher
        return result

    def needs_prediction(self, round_number):
        return any(round_number - cohort.round in self.cfg["transfer_followup_rounds"]
                   for cohort in self.pending)

    def aggregate(self, cohort, logits):
        cohort.aggregate_correct = logits.argmax(1).eq(cohort.labels)
        if self.cfg["transfer_followup_rounds"]:
            self.pending.append(cohort)
        result = cohort.score(logits)
        result["aggregation_gap"] = cohort.aggregation_gap(logits)
        return result

    def followups(self, round_number, logits):
        results = {}
        lags = self.cfg["transfer_followup_rounds"]
        for cohort in self.pending:
            lag = round_number - cohort.round
            if lag in lags:
                results["lag_{:03d}".format(lag)] = dict(
                    origin_round=cohort.round, elapsed_rounds=lag, **cohort.followup(logits))
                self.completed_followups += 1
        self.pending = [cohort for cohort in self.pending if round_number - cohort.round < max(lags)] if lags else []
        return results

    def final(self, last_round):
        incomplete = sum(sum(last_round - cohort.round < lag for lag in self.cfg["transfer_followup_rounds"])
                         for cohort in self.pending)
        return dict(cohorts_started=self.started, completed_followups=self.completed_followups,
                    unobserved_followups_at_end=incomplete)
