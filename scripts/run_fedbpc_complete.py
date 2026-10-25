#!/usr/bin/env python3
"""Launch one explicit run with a unique output directory; never download data."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("full", "mean_only", "no_alignment", "alignment_only", "coupled_backbone", "fedavg")


def build_config(args, output):
    config = json.loads((ROOT / "src/config" / ("fedbpc_complete_" + args.partition + ".json")).read_text())
    config["data_setups"].update(root=str(args.data_root.expanduser().resolve()),
        partition_seed=19940817, partition_cache={"enabled": True})
    config["train_setups"]["scenario"].update(device=args.device, n_rounds=args.rounds)
    config["train_setups"]["seed"] = args.seed
    params = config["train_setups"]["algo"]["params"]
    if args.variant == "mean_only":
        params["variance_scale"] = 0.
    elif args.variant == "no_alignment":
        params["lambda_align"] = 0.
    elif args.variant == "alignment_only":
        params["completion_strength"] = 0.
    elif args.variant == "coupled_backbone":
        params["preserve_backbone_gradients"] = False
    elif args.variant == "fedavg":
        config["train_setups"]["algo"] = {"name": "fedavg", "params": {"metrics_only": True}}
    config["wandb_setups"].update(name=output.name, mode="offline", dir=str(output.parent))
    config["batch_protocol"]["output_dir"] = str(output)
    config["batch_protocol"]["download_if_missing"] = False
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("sharding", "lda"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "fedbpc_complete_experiments")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configuration without training or writing files")
    args = parser.parse_args()
    if args.rounds < 1 or args.seed < 0:
        parser.error("Positive rounds and nonnegative seed required")
    name = "{}_{}_seed{}_{}_{}".format(args.partition, args.variant, args.seed,
        datetime.now().strftime("%Y%m%d_%H%M%S"), uuid.uuid4().hex[:8])
    output = args.output_root.expanduser().resolve() / name
    config = build_config(args, output)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    # Keep the input outside the reserved run directory: the training entry point
    # refuses nonempty result directories before touching any existing result.
    config_dir = output.parent / ".configs"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / (name + ".json")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print("Result directory: " + str(output), flush=True)
    environment = dict(os.environ, WANDB_MODE="offline", PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([sys.executable, "-B", str(ROOT / "src/main.py"),
                    "--config_path", str(config_path)], cwd=str(ROOT / "src"),
                   env=environment, check=True)


if __name__ == "__main__":
    main()
