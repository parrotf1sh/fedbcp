"""Overall performance of persistent client predictors on a shared test set."""

import time

import wandb

from experiment_logging import flatten_metrics, log_final, log_round, tracking_enabled
from .utils import feature_module


def run_experiment(server):
    run = wandb.run
    if not tracking_enabled(run):
        raise ValueError("FedProto requires an active W&B run to save experiment results")
    config = getattr(server, "experiment_config", None)
    _, use_output = feature_module(server.model)
    metadata = dict(
        algorithm="fedproto",
        overall_test_definition="equal-client mean accuracy on the same complete global test set",
        primary_prediction="MSE nearest global prototype with upstream local-class restriction",
        prototype_candidates="intersection of global prototype keys and client training classes",
        missing_class_distance=server.cfg["missing_class_distance"],
        missing_candidate_behavior="finite sentinel and argmin ties retained; all sentinel values select class 0",
        classifier_prediction="unrestricted classifier argmax; auxiliary metric only",
        global_model=False,
        feature_location="layer4 output before pooling" if use_output else "classifier input",
        initialization="common project model initialization; client states persist independently",
        optimizer_state="reset at every client participation",
        prototype_aggregation="equal contributor mean, current-round uploads only",
        prototype_collection="last local epoch; online pre-step features in training mode",
        evaluation_clients=server.n_clients,
        client_std_definition="population standard deviation across clients, not across experiment seeds",
        accuracy_units="fraction in [0, 1]",
        selection="fixed final round; no validation-best selection",
        groups=server.groups,
        train_drop_last={str(index): bool(loaders["train"].drop_last)
                         for index, loaders in server.data_distributed["local"].items()},
    )
    if "split_manifest" in server.data_distributed:
        metadata.update(
            split_sha256=server.data_distributed["split_manifest"]["sha256"],
            split_manifest_path=server.data_distributed["split_manifest_path"],
        )
    run.config.update(config if config is not None else dict(
        algo_params=server.cfg, n_rounds=server.n_rounds,
        sample_ratio=server.sample_ratio, local_epochs=server.local_epochs,
        device=str(server.device),
    ))
    run.config.update({"fedproto_runtime": metadata})
    run.define_metric("round")
    run.define_metric("*", step_metric="round")

    for round_idx in range(server.n_rounds):
        sampled = server._client_sampling(round_idx)
        server.server_results["client_history"].append(sampled)
        learning_rate = server.optimizer.param_groups[0]["lr"]
        started = time.perf_counter()
        train_results = server._clients_training(sampled)
        evaluation = server.evaluate_clients()
        prototype, classifier = evaluation["prototype"], evaluation["classifier"]
        if prototype is None:
            raise ValueError("Cannot record FedProto overall accuracy without global prototypes")
        metrics = dict(
            round=round_idx + 1,
            overall_test_accuracy=prototype["accuracy"],
            server_test_acc=prototype["accuracy"],
        )
        metrics.update({
            "global/top1": prototype["accuracy"],
            "global/macro_f1": prototype["macro_f1"],
            "global/worst20_recall": prototype["worst20_recall"],
            "classifier_test_accuracy": classifier["accuracy"],
            "train/learning_rate": learning_rate,
            "prototype/coverage": len(server.global_prototypes) / server.num_classes,
            "prototype/clients_without_candidates": sum(
                record["eligible_classes"] == 0 for record in evaluation["clients"]
            ),
        })
        for key, values in train_results.items():
            metrics["train/" + key] = sum(values) / len(values)
        metrics.update(flatten_metrics(prototype, "test/prototype"))
        metrics.update(flatten_metrics(classifier, "test/classifier"))
        metrics.update(flatten_metrics([int(index) for index in sampled], "sampled_clients"))
        for record in evaluation["clients"]:
            prefix = "clients/{:03d}/".format(record["client_idx"])
            metrics[prefix + "prototype_accuracy"] = record["prototype"]["accuracy"]
            metrics[prefix + "classifier_accuracy"] = record["classifier"]["accuracy"]
            metrics[prefix + "eligible_classes"] = record["eligible_classes"]
        server.step_scheduler()
        metrics["time/round_seconds"] = time.perf_counter() - started
        log_round(run, metrics)
        server.server_results["test_accuracy"].append(prototype["accuracy"])
        server.server_results["classifier_test_accuracy"].append(classifier["accuracy"])
        print("[Round {}/{}] Overall test acc={:.4f}, classifier acc={:.4f} "
              "(all {} clients), elapsed={:.1f}s".format(
                  round_idx + 1, server.n_rounds, prototype["accuracy"],
                  classifier["accuracy"], server.n_clients, metrics["time/round_seconds"]),
              flush=True)

    validation_loader = server.data_distributed["global"].get("validation")
    validation = (server.evaluate_clients(validation_loader)
                  if validation_loader is not None else None)
    local_tests = server.evaluate_clients(local=True)
    history = server.server_results["test_accuracy"]
    summary = dict(
        completed_rounds=len(history),
        overall_test_accuracy=history[-1],
        final_top1=history[-1],
        last10_top1_mean=sum(history[-10:]) / len(history[-10:]),
        final_macro_f1=prototype["macro_f1"],
        final_worst20_recall=prototype["worst20_recall"],
        final_client_accuracy_std=prototype["client_accuracy_std"],
        final_classifier_test_accuracy=classifier["accuracy"],
        evaluation_definition=metadata["overall_test_definition"],
        primary_prediction=metadata["primary_prediction"],
        selection=metadata["selection"],
    )
    if config is not None:
        summary["seed"] = config["train_setups"]["seed"]
    run.summary.update({"fedproto_final": dict(
        overall_test=evaluation, overall_validation=validation,
        local_test=local_tests,
        local_test_definition="existing project local test loaders only; unavailable clients omitted",
    )})
    final_metrics = dict(overall_test=prototype, classifier_test=classifier)
    if validation is not None:
        final_metrics["overall_validation"] = validation["prototype"]
        final_metrics["classifier_validation"] = validation["classifier"]
    log_final(run, final_metrics)
    run.summary.update(summary)
    server.batch_summary = summary
    return summary
