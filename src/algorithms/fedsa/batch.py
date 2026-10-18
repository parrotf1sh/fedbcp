"""Opt-in queue telemetry; no model serialization or changes to FedSA math."""
import csv
import json
from pathlib import Path


def evaluation_rounds(n_rounds, interval=10, final_rounds=10):
    for name, value in (("n_rounds", n_rounds), ("eval_interval", interval),
                        ("final_eval_rounds", final_rounds)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    if final_rounds < 10:
        raise ValueError("final_eval_rounds must be at least 10 for the last10 metric")
    return sorted({1, *range(interval, n_rounds + 1, interval),
                   *range(max(1, n_rounds - final_rounds + 1), n_rounds + 1)})


def prototype_bytes(prototypes):
    from ..fedavg.reporting import tensor_bytes
    # Tensor payload plus one int64 class ID per sparse dictionary entry.
    return tensor_bytes(prototypes) + 8 * len(prototypes)


class BatchReporter:
    def __init__(self, server):
        from ..fedavg.reporting import log_partition, tensor_bytes, write_json
        self.server = server
        protocol = server.experiment_config["batch_protocol"]
        if protocol.get("save_checkpoints", False):
            raise ValueError("FedSA batch experiments prohibit checkpoints")
        if protocol.get("write_metrics_files") is not True:
            raise ValueError("FedSA batch queue requires metric/status files")
        self.evaluation_rounds = evaluation_rounds(
            server.n_rounds, protocol["eval_interval"], protocol["final_eval_rounds"])
        if protocol.get("evaluation_rounds") != self.evaluation_rounds:
            raise ValueError("FedSA evaluation schedule disagrees with the task manifest")
        self.output = Path(protocol["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output / "metrics.jsonl"
        self.history_path.write_text("")
        (self.output / "clients.jsonl").write_text("")
        write_json(self.output / "evaluation_rounds.json", self.evaluation_rounds)
        log_partition(server, self.output)
        self.train_seconds = self.eval_seconds = 0.0
        self.train_bytes = 0
        self.completed_rounds = 0
        self.top1_by_round = {}
        self.initial_model_bytes = server.n_clients * tensor_bytes(server.initial_weights)
        server.batch_tracking = True
        server.batch_eval_bytes = 0

    def record(self, metrics, sampled, train_seconds, eval_seconds):
        from ..fedavg.reporting import tensor_bytes
        from experiment_logging import flatten_metrics
        self.train_seconds += train_seconds
        self.eval_seconds += eval_seconds
        down = len(sampled) * tensor_bytes(self.server.semantic_anchors)
        up = self.server.batch_round_upload_bytes
        self.train_bytes += down + up
        metrics.update({
            "time/train_round_seconds": train_seconds,
            "time/eval_round_seconds": eval_seconds,
            "time/cumulative_train_seconds": self.train_seconds,
            "time/cumulative_eval_seconds": self.eval_seconds,
            "communication/anchor_download_bytes": down,
            "communication/prototype_upload_bytes": up,
            "communication/round_bytes": down + up,
            "communication/cumulative_bytes": self.train_bytes,
            "communication/evaluation_cumulative_bytes": self.server.batch_eval_bytes,
        })
        # Validate before writing; sparse fields stay absent on non-evaluation rounds.
        metrics = flatten_metrics(metrics)
        with self.history_path.open("a") as handle:
            handle.write(json.dumps(metrics, allow_nan=False) + "\n")
        with (self.output / "clients.jsonl").open("a") as handle:
            handle.write(json.dumps({"round": metrics["round"],
                                     "clients": [int(i) for i in sampled]}) + "\n")
        self.completed_rounds += 1
        if "global/top1" in metrics:
            self.top1_by_round[metrics["round"]] = metrics["global/top1"]

    def finish(self, summary, final_report_seconds):
        expected = list(range(max(1, self.server.n_rounds - 9), self.server.n_rounds + 1))
        if (self.completed_rounds != self.server.n_rounds
                or sorted(self.top1_by_round) != self.evaluation_rounds
                or any(index not in self.top1_by_round for index in expected)):
            raise ValueError("FedSA run lacks required training/evaluation rounds")
        summary.update({
            "completed_rounds": self.completed_rounds,
            "evaluation_rounds": self.evaluation_rounds,
            "evaluated_rounds": len(self.top1_by_round),
            "last10_top1_mean": sum(self.top1_by_round[index] for index in expected) / len(expected),
            "last10_training_rounds": expected,
            "total_train_seconds": self.train_seconds,
            "total_eval_seconds": self.eval_seconds,
            "final_additional_report_seconds": final_report_seconds,
            "communication_bytes": self.train_bytes,
            "evaluation_communication_bytes": self.server.batch_eval_bytes,
            "initial_model_distribution_bytes_estimate": self.initial_model_bytes,
            "communication_definition": (
                "training only: anchors download per participant plus sparse prototype upload "
                "with int64 class IDs; no recurrent model transfer; evaluation prototype "
                "distribution and one-time initial model distribution estimate reported separately; "
                "excludes scalar control, protocol overhead and simulator CPU/GPU copies"),
        })
        # Columns may first appear after round one (e.g. local gap diagnostics).
        fields = {}
        with self.history_path.open() as handle:
            for line in handle:
                fields.update(dict.fromkeys(json.loads(line)))
        with (self.output / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields))
            writer.writeheader()
            with self.history_path.open() as source:
                for line in source:
                    writer.writerow(json.loads(line))
