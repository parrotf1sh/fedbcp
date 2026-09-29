import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import algorithms
from train_tools import *
from utils import *

import numpy as np
import argparse
import warnings
import wandb
import random
import pprint
import os

warnings.filterwarnings("ignore")

# Set torch base print precision
torch.set_printoptions(10)

ALGO = {
    "fedavg": algorithms.fedavg.Server,
    "moon": algorithms.moon.Server,
    "fedbpc": algorithms.fedbpc.Server,
    "fedbtr": algorithms.fedbtr.Server,
    "fedproc": algorithms.fedproc.Server,
}

SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}


def _get_setups(args):
    """Get train configuration"""

    # Fix randomness for data distribution
    np.random.seed(19940817)
    random.seed(19940817)

    # Distribute the data to clients
    data_options = dict(args.data_setups)
    pipeline = data_options.pop("pipeline", "legacy")
    if pipeline == "longtail":
        from train_tools.preprocessing.longtail_datasetter import longtail_data_distributer
        data_distributed = longtail_data_distributer(**data_options)
    elif pipeline == "legacy":
        data_distributed = data_distributer(**data_options)
    else:
        raise ValueError("Unknown data pipeline: {}".format(pipeline))

    # Fix randomness for experiment
    _random_seeder(args.train_setups.seed)
    model = create_models(
        args.train_setups.model.name,
        args.data_setups.dataset_name,
        **args.train_setups.model.params,
    )

    # Optimization setups
    if args.train_setups.algo.name == "fedproc":
        algo_params = args.train_setups.algo.params
        if algo_params.get("use_project_head", False):
            model = algorithms.fedproc.ModelWithProjection(
                model, out_dim=algo_params.get("out_dim", 256)
            )
        else:
            model = algorithms.fedproc.ModelWithFeatures(model)
        optimizer = algorithms.fedproc.create_optimizer(
            model,
            name=args.train_setups.optimizer.get("name", "sgd"),
            **args.train_setups.optimizer.params,
        )
    else:
        optimizer = optim.SGD(model.parameters(), **args.train_setups.optimizer.params)
    scheduler = None

    if args.train_setups.scheduler.enabled:
        scheduler = SCHEDULER[args.train_setups.scheduler.name](
            optimizer, **args.train_setups.scheduler.params
        )

    # Algorith-specific global server container
    algo_params = args.train_setups.algo.params
    if args.train_setups.algo.name == "fedbtr":
        algo_params = dict(algo_params)
        algo_params.setdefault("experiment_seed", args.train_setups.seed)
    server = ALGO[args.train_setups.algo.name](
        algo_params,
        model,
        data_distributed,
        optimizer,
        scheduler,
        **args.train_setups.scenario,
    )

    if pipeline == "longtail":
        # Some baseline constructors inspect training loaders. Do not let that
        # consume the opt-in protocol's initial sampling/augmentation streams.
        _random_seeder(args.train_setups.seed)
        for loaders in [data_distributed["global"]] + list(data_distributed["local"].values()):
            for loader in loaders.values():
                if isinstance(loader, torch.utils.data.DataLoader) and loader.generator is not None:
                    loader.generator.manual_seed(loader.generator.initial_seed())

    return server


def _random_seeder(seed):
    """Fix randomness"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """Execute experiment"""

    # Load the configuration
    server = _get_setups(args)
    if args.get("batch_protocol"):
        server.experiment_config = args
    if args.train_setups.algo.name == "fedbtr":
        # Preserve the resolved architecture/data/scenario as well as algo params.
        server.set_experiment_config(args)

    # Conduct FL
    server.run()

    if (args.data_setups.get("pipeline") == "longtail"
            and args.train_setups.algo.name != "fedbtr"):
        from longtail_report import save_baseline_report
        save_baseline_report(server, args)

    # Save the final global model
    # model_path = os.path.join(wandb.run.dir, "model.pth")
    # torch.save(server.model.state_dict(), model_path)

    # Upload model to wandb
    # wandb.save(model_path)

    return getattr(server, "batch_summary", None)


# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Process Configs")
parser.add_argument("--config_path", default="./config/fedavg.json", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--n_clients", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--partition_method", type=str)
parser.add_argument("--partition_s", type=int)
parser.add_argument("--partition_alpha", type=float)
parser.add_argument("--model_name", type=str)
parser.add_argument("--n_rounds", type=int)
parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--local_epochs", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--momentum", type=float)
parser.add_argument("--wd", type=float)
parser.add_argument("--algo_name", type=str)
parser.add_argument("--device", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--group", type=str)
parser.add_argument("--exp_name", type=str)
args = parser.parse_args()

#######################################################################################

if __name__ == "__main__":
    # Load configuration from .json file
    opt = ConfLoader(args.config_path).opt

    # Overwrite config by parsed arguments
    opt = config_overwriter(opt, args)

    # Print configuration dictionary pretty
    print("")
    print("=" * 50 + " Configuration " + "=" * 50)
    pp = pprint.PrettyPrinter(compact=True)
    pp.pprint(opt)
    print("=" * 120)

    if opt.get("batch_protocol"):
        from fedavg_batch import run_tracked
        run_tracked(opt, main)
    else:
        # Preserve the existing entry point for all other experiments.
        wandb.init(config=opt, **opt.wandb_setups)
        wandb.config.log_interval = 10
        main(opt)
