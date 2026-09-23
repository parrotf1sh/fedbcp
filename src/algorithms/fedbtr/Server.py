"""FedBTR: balanced readout construction followed by tolerance-band retention."""

import copy
import json
import os
import time
import uuid
from contextlib import nullcontext

import numpy as np
import torch
import wandb

from experiment_logging import configure_tracking, log_final, log_round, tracking_enabled

from ..BaseServer import BaseServer
from .ClientTrainer import ClientTrainer
from .config import resolve_config
from .diagnostics import BatchTrace, RandomSnapshot, TrainingProbe, TransferDiagnostics
from .teacher import BalancedReadoutTeacher
from .utils import (aggregate_states, classifier_module, cpu_state, evaluate,
                    groups_from_counts, sync_device, tensor_bytes)

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(self, algo_params, model, data_distributed, optimizer, scheduler=None, **kwargs):
        super().__init__(algo_params, model, data_distributed, optimizer, scheduler, **kwargs)
        self.cfg = resolve_config(algo_params)
        if not 0 < self.sample_ratio <= 1 or self.n_rounds < 1 or self.local_epochs < 1:
            raise ValueError("FedBTR requires positive rounds/epochs and sample_ratio in (0, 1]")
        if int(self.n_rounds) != self.n_rounds or int(self.local_epochs) != self.local_epochs:
            raise ValueError("rounds and local_epochs must be integers")
        self.n_rounds, self.local_epochs = int(self.n_rounds), int(self.local_epochs)
        self.validation = data_distributed["global"].get("validation")
        if self.validation is None:
            raise ValueError("FedBTR requires held-out validation: use data_setups.pipeline=longtail")
        if "global_class_counts" not in data_distributed:
            raise ValueError("Global training counts are required; use the shared longtail pipeline")
        self.counts = torch.as_tensor(data_distributed["global_class_counts"], dtype=torch.long)
        if self.counts.ndim != 1 or self.counts.numel() != self.num_classes or (self.counts <= 0).any():
            raise ValueError("Invalid global training counts")
        if classifier_module(model)[1].out_features != self.num_classes:
            raise ValueError("Classifier output does not match dataset class count")
        self.groups = groups_from_counts(self.counts.tolist())
        self.total_samples = int(self.counts.sum())
        observed = torch.zeros_like(self.counts)
        for index, data in data_distributed["local"].items():
            if index not in range(self.n_clients) or "calibration" not in data or "class_counts" not in data:
                raise ValueError("Client data must follow the longtail pipeline contract")
            local_counts = torch.as_tensor(data["class_counts"], dtype=torch.long)
            if local_counts.shape != self.counts.shape or (local_counts < 0).any():
                raise ValueError("Invalid local training counts")
            if int(local_counts.sum()) != data["datasize"] or data["datasize"] <= 0:
                raise ValueError("Local count/sample size mismatch")
            observed += local_counts
        if not torch.equal(observed, self.counts):
            raise ValueError("Global counts must equal the actual client training counts")
        self.model.cpu()
        self.client = ClientTrainer(global_counts=self.counts, cfg=self.cfg,
                                    algo_params=self.cfg, model=copy.deepcopy(model),
                                    local_epochs=self.local_epochs, device=self.device,
                                    num_classes=self.num_classes)
        self.transfer = None
        if self.cfg["transfer_diagnostics"]:
            for data in data_distributed["local"].values():
                if getattr(data["train"], "persistent_workers", False):
                    raise ValueError("Transfer shadow replay requires non-persistent train workers")
            self.transfer = TransferDiagnostics(self.validation, self.groups, self.cfg,
                                                self.counts, self.local_epochs, self.device)
        run_name = "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:8])
        self.output_dir = os.path.abspath(os.path.join(self.cfg["output_dir"], run_name))
        os.makedirs(self.output_dir, exist_ok=False)
        self.best_state = None
        self.best_accuracy = -1.0
        self.best_round = None
        self.last_teacher = None
        self.run_metadata = dict(
            algorithm="FedBTR", config=self.cfg, groups=self.groups,
            global_class_counts=self.counts.tolist(),
            n_clients=self.n_clients, n_rounds=self.n_rounds,
            local_epochs=self.local_epochs, sample_ratio=self.sample_ratio,
            optimizer={key: value for key, value in self.optimizer.param_groups[0].items() if key != "params"},
            model=type(model).__name__,
            split_sha256=data_distributed.get("split_manifest", {}).get("sha256"),
            split_manifest_path=data_distributed.get("split_manifest_path"),
            one_time_count_uplink_bytes=self.n_clients * self.num_classes * 8,
            one_time_global_count_downlink_bytes=self.n_clients * self.num_classes * 8,
            selection="validation macro accuracy; no test-based selection",
            privacy="Simulation holds all data; deployed teacher needs global counts and client sizes. Local support is computed client-side; optional diagnostics expose support-bin summaries.",
        )
        if self.transfer is not None:
            self.run_metadata["transfer_diagnostics"] = dict(
                version=1, data="ordered held-out validation; augmented training exposures for band probe",
                cohort="global wrong and calibrated teacher correct, fixed at origin round",
                shadows="same global/optimizer state and replayed RNG; selected clients only; never aggregated",
                followups="endpoint observations, not proof of uninterrupted correctness",
                privacy="Simulation-only central evaluation of local/shadow models; no raw records uploaded to W&B",
            )
        self.set_experiment_config(None)
        configure_tracking(wandb.run, self.run_metadata)
        print(">>> FedBTR results: {}".format(self.output_dir))

    def set_experiment_config(self, experiment):
        # One local metadata file instead of separate experiment/run JSONs.
        self._write_json("experiment.json", dict(experiment=experiment, runtime=self.run_metadata))

    def _write_json(self, name, value):
        path = os.path.join(self.output_dir, name)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
        os.replace(temporary, path)

    def _checkpoint(self, name, state, round_idx, validation):
        if not self.cfg["save_checkpoints"]:
            return
        path = os.path.join(self.output_dir, name)
        torch.save(dict(model=state, round=round_idx, validation=validation,
                        config=self.cfg, groups=self.groups,
                        split_sha256=self.data_distributed.get("split_manifest", {}).get("sha256")),
                   path + ".tmp")
        os.replace(path + ".tmp", path)

    def _evaluate(self, model, loader=None):
        return evaluate(model, self.validation if loader is None else loader,
                        self.num_classes, self.groups, self.device)

    def _log(self, record, handle):
        if handle is not None:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
        log_round(wandb.run, record)

    def run(self):
        model_bytes = tensor_bytes(self.model.state_dict())
        total_uplink = total_downlink = 0
        metrics_path = os.path.join(self.output_dir, "rounds.jsonl")
        # W&B offline mode already keeps its own local history. Only fall back
        # to JSONL when tracking is absent/disabled, or explicitly requested.
        local_history = self.cfg["save_local_history"] or not tracking_enabled(wandb.run)
        history_context = open(metrics_path, "w", encoding="utf-8") if local_history else nullcontext(None)
        with history_context as handle:
            for round_idx in range(self.n_rounds):
                sync_device(self.device)
                round_start = time.perf_counter()
                self.model.cpu()
                original_state = cpu_state(self.model)
                optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                count = max(1, int(self.n_clients * self.sample_ratio))
                # Same schedule as BaseServer, without reseeding training RNGs.
                sampled = np.random.RandomState(round_idx).choice(self.n_clients, count, replace=False).tolist()
                self.server_results["client_history"].append(sampled)
                clients = [self.data_distributed["local"][index] for index in sampled]
                record = dict(round=round_idx + 1, clients=sampled,
                              learning_rate=optimizer_state["param_groups"][0]["lr"])
                teacher = None
                teacher_stats = dict(teacher_examples=0, teacher_uplink_bytes=0, teacher_downlink_bytes=0)
                stage_start = time.perf_counter()
                if self.cfg["teacher_mode"] != "none":
                    construction = BalancedReadoutTeacher(self.model, self.counts, self.cfg, self.device)
                    teacher_stats = construction.calibrate(clients, self.n_clients, self.total_samples)
                    teacher = construction.model
                    del construction
                sync_device(self.device)
                record["teacher_construction_seconds"] = time.perf_counter() - stage_start
                record.update(teacher_stats)

                diagnostic = (self.cfg["diagnostic_interval"] > 0 and
                              (round_idx == 0 or (round_idx + 1) % self.cfg["diagnostic_interval"] == 0))
                evaluate_round = ((round_idx + 1) % self.cfg["eval_interval"] == 0 or
                                  round_idx == self.n_rounds - 1)
                evaluation_seconds = 0.0
                transfer_seconds = 0.0
                cohort = None
                if diagnostic:
                    start = time.perf_counter()
                    record["global_before_validation"] = self._evaluate(self.model)
                    self.model.cpu()
                    if teacher is not None:
                        record["teacher_validation"] = self._evaluate(teacher)
                    sync_device(self.device)
                    evaluation_seconds += time.perf_counter() - start

                if diagnostic and self.transfer is not None:
                    sync_device(self.device)
                    stage_start = time.perf_counter()
                    cohort = self.transfer.start(round_idx + 1, self.model, teacher)
                    record["transfer"] = dict(origin_round=round_idx + 1,
                                              reference=cohort.score(cohort.teacher), clients=[])
                    sync_device(self.device)
                    transfer_seconds += time.perf_counter() - stage_start

                states, sizes, training = [], [], []
                record["local_diagnostics"] = []
                start = time.perf_counter()
                local_evaluation_seconds = 0.0
                local_transfer_seconds = 0.0
                for slot, (index, data) in enumerate(zip(sampled, clients)):
                    probe = replay = entry = batch_trace = None
                    if cohort is not None and slot < self.cfg["diagnostic_clients"]:
                        sync_device(self.device)
                        stage_start = time.perf_counter()
                        entry = dict(client=index, retention_enabled=int(round_idx >= self.cfg["warmup_rounds"]),
                                     validation_band=cohort.band(data["class_counts"]))
                        probe = TrainingProbe(teacher, self.model, data["class_counts"], self.cfg, self.device)
                        batch_trace = BatchTrace()
                        replay = RandomSnapshot(data["train"], device=self.device)
                        sync_device(self.device)
                        local_transfer_seconds += time.perf_counter() - stage_start
                    try:
                        state, metrics = self.client.train_client(original_state, optimizer_state,
                                                                 data, teacher, round_idx, probe=probe,
                                                                 batch_trace=batch_trace)
                    finally:
                        if probe is not None:
                            probe.close()
                    states.append(state)
                    sizes.append(data["datasize"])
                    training.append(metrics)
                    if cohort is not None:
                        sync_device(self.device)
                        stage_start = time.perf_counter()
                        actual_logits = self.transfer.predict(self.client.model)
                        cohort.add_local(actual_logits, data["datasize"])
                        if entry is not None:
                            self.client.model.cpu()
                            entry["training_band"] = probe.result()
                            entry["replayed_batches"] = batch_trace.batches
                            entry["replayed_examples"] = batch_trace.examples
                            entry["branches"] = self.transfer.shadows(
                                cohort, self.model, actual_logits, original_state, optimizer_state,
                                data, round_idx, replay, batch_trace)
                            record["transfer"]["clients"].append(entry)
                        sync_device(self.device)
                        local_transfer_seconds += time.perf_counter() - stage_start
                        if probe is not None:
                            local_transfer_seconds += probe.seconds + batch_trace.seconds
                        del probe, replay, actual_logits, batch_trace
                    if diagnostic and slot < self.cfg["diagnostic_clients"]:
                        evaluation_start = time.perf_counter()
                        after = self._evaluate(self.client.model)
                        initial = (record["teacher_validation"] if self.cfg["student_init"] == "teacher"
                                   else record["global_before_validation"])
                        delta = [a - b if a is not None and b is not None else None
                                 for a, b in zip(after["class_accuracy"], initial["class_accuracy"])]
                        local_counts = data["class_counts"].tolist()
                        summaries = {}
                        for name, indices in (
                            ("missing", [c for c, n in enumerate(local_counts) if n == 0]),
                            ("low_support", [c for c, n in enumerate(local_counts) if 0 < n < self.cfg["support_scale"]]),
                            ("supported", [c for c, n in enumerate(local_counts) if n >= self.cfg["support_scale"]]),
                        ):
                            values = [delta[c] for c in indices if delta[c] is not None]
                            summaries[name + "_mean_delta"] = sum(values) / len(values) if values else None
                        record["local_diagnostics"].append(dict(client=index, validation=after,
                                                               class_delta=delta, **summaries))
                        sync_device(self.device)
                        local_evaluation_seconds += time.perf_counter() - evaluation_start
                sync_device(self.device)
                record["local_training_seconds"] = (time.perf_counter() - start
                                                    - local_evaluation_seconds - local_transfer_seconds)
                transfer_seconds += local_transfer_seconds
                evaluation_seconds += local_evaluation_seconds
                self.client.model.cpu()
                if teacher is not None:
                    teacher.cpu()
                self.last_teacher = teacher
                weights = aggregate_states(states, sizes)
                self.model.load_state_dict(weights)
                total_examples = sum(item["training_examples"] for item in training)
                record["training"] = {
                    key: sum(item[key] * item["training_examples"] for item in training) / total_examples
                    for key in ("classification", "retention", "band_active", "teacher_correct", "train_accuracy")}
                record["training"].update(examples=total_examples,
                    teacher_forward_examples=sum(item["teacher_forward_examples"] for item in training))
                record["base_uplink_bytes"] = model_bytes * count
                record["base_downlink_bytes"] = model_bytes * count
                total_uplink += record["base_uplink_bytes"] + teacher_stats["teacher_uplink_bytes"]
                total_downlink += record["base_downlink_bytes"] + teacher_stats["teacher_downlink_bytes"]
                record.update(cumulative_uplink_bytes=total_uplink, cumulative_downlink_bytes=total_downlink)

                start = time.perf_counter()
                validation = None
                if evaluate_round or diagnostic:
                    validation = self._evaluate(self.model)
                    record["global_validation"] = validation
                    if validation["evaluated_classes"] != self.num_classes:
                        raise ValueError("Validation must cover all classes for macro selection")
                    # Diagnostic frequency must not change the selection budget.
                    if evaluate_round and validation["macro_accuracy"] > self.best_accuracy:
                        self.best_accuracy = validation["macro_accuracy"]
                        self.best_state = cpu_state(self.model)
                        self.best_round = round_idx + 1
                        self._checkpoint("best.pt", self.best_state, self.best_round, validation)
                if self.cfg["test_interval"] > 0 and (round_idx + 1) % self.cfg["test_interval"] == 0:
                    record["global_test_reporting_only"] = self._evaluate(self.model, self.testloader)
                sync_device(self.device)
                evaluation_seconds += time.perf_counter() - start
                self.model.cpu()
                record["evaluation_seconds"] = evaluation_seconds
                if self.transfer is not None:
                    sync_device(self.device)
                    stage_start = time.perf_counter()
                    if cohort is not None or self.transfer.needs_prediction(round_idx + 1):
                        logits = self.transfer.predict(self.model)
                        # Finish older origins before registering this round's cohort.
                        followups = self.transfer.followups(round_idx + 1, logits)
                        if followups:
                            record["transfer_followup"] = followups
                        if cohort is not None:
                            record["transfer"]["aggregate"] = self.transfer.aggregate(cohort, logits)
                        del logits
                    sync_device(self.device)
                    transfer_seconds += time.perf_counter() - stage_start
                    record["transfer_diagnostic_seconds"] = transfer_seconds
                if self.scheduler is not None:
                    self.scheduler.step()
                record["round_seconds"] = time.perf_counter() - round_start
                self._log(record, handle)
                score = "not evaluated" if validation is None else "{:.4f}".format(validation["macro_accuracy"])
                print("[FedBTR {}/{}] validation={} elapsed={:.2f}s".format(
                    round_idx + 1, self.n_rounds, score, record["round_seconds"]))

        last_state = cpu_state(self.model)
        self._checkpoint("last.pt", last_state, self.n_rounds, validation)
        final = dict(best_round=self.best_round, best_validation_macro=self.best_accuracy,
                     last_global_test=self._evaluate(self.model, self.testloader),
                     cumulative_uplink_bytes=total_uplink, cumulative_downlink_bytes=total_downlink)
        if self.transfer is not None:
            final["transfer_diagnostics"] = self.transfer.final(self.n_rounds)
        self.model.load_state_dict(self.best_state)
        final["best_validation_selected_test"] = self._evaluate(self.model, self.testloader)
        self.model.load_state_dict(last_state)
        self.model.cpu()
        if self.last_teacher is not None:
            final["last_pre_local_teacher_test"] = self._evaluate(self.last_teacher, self.testloader)
            self.last_teacher.cpu()
            self._checkpoint("last_teacher.pt", cpu_state(self.last_teacher), self.n_rounds, None)
        self._write_json("final.json", final)
        log_final(wandb.run, final)
        print(">>> FedBTR complete: {}".format(self.output_dir))
