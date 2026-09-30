"""Shared W&B lifecycle and automatic data preparation for baseline runs."""
import platform
from pathlib import Path

import torch
import wandb

from algorithms.fedavg.reporting import write_json
from fedavg_data import prepare_data, TINY_URL


def run_tracked(config, train):
    output = None
    if config["train_setups"]["algo"]["name"] != "fedproto":
        output = Path(config["batch_protocol"]["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
    config["runtime"] = {"python": platform.python_version(), "torch": str(torch.__version__),
                         "cuda": torch.version.cuda, "wandb": wandb.__version__}
    config["batch_protocol"]["tiny_imagenet_url"] = TINY_URL
    if output is not None:
        write_json(output / "config.json", config)
    run = None
    try:
        run = wandb.init(config=config, **config["wandb_setups"])
        prepare_data(config)
        summary = train(config)
        if not summary:
            raise RuntimeError("Batch training did not return a metrics summary")
        summary.update(status="success", wandb_run_id=run.id, wandb_url=run.url)
        if output is None:
            run.summary.update(summary)
        else:
            run.summary["status"] = "success"
        run.finish(exit_code=0)
        if output is not None:
            write_json(output / "summary.json", summary)
    except BaseException as exc:
        summary = {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                   "error": "{}: {}".format(type(exc).__name__, exc)}
        if output is not None:
            write_json(output / "summary.json", summary)
        if run is not None:
            try:
                run.summary.update(summary)
                run.finish(exit_code=130 if isinstance(exc, KeyboardInterrupt) else 1)
            except Exception:
                pass  # Keep the original failure and let the coordinator continue.
        raise
