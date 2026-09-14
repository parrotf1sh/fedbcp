"""Final fixed-round report for existing algorithms opting into longtail data."""

import json
import os
import time
import uuid

import wandb

from experiment_logging import log_final, tracking_enabled
from longtail_metrics import evaluate, groups_from_counts


def save_baseline_report(server, config):
    data = server.data_distributed
    groups = groups_from_counts(data["global_class_counts"].tolist())
    algorithm = config["train_setups"]["algo"]["name"]
    directory = os.path.abspath(os.path.join("results", "longtail_baselines", algorithm,
        time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]))
    os.makedirs(directory, exist_ok=False)
    report = dict(algorithm=algorithm, config=config, groups=groups,
                  split_sha256=data["split_manifest"]["sha256"],
                  split_manifest_path=data["split_manifest_path"],
                  selection="fixed final round; no validation-best selection",
                  last_global_validation=evaluate(server.model, data["global"]["validation"],
                      server.num_classes, groups, server.device),
                  last_global_test=evaluate(server.model, server.testloader,
                      server.num_classes, groups, server.device))
    with open(os.path.join(directory, "final.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    if tracking_enabled(wandb.run):
        wandb.run.config.update({"longtail_runtime": dict(groups=groups,
            split_sha256=report["split_sha256"], selection=report["selection"])})
    log_final(wandb.run, {key: report[key] for key in ("last_global_validation", "last_global_test")})
    print(">>> Longtail baseline report: {}".format(directory))
    return directory
