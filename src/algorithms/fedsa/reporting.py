"""FedSA metrics for persistent client predictors, using the shared W&B lifecycle."""

import time

import torch
import wandb

from experiment_logging import flatten_metrics, log_final, log_round, tracking_enabled
from .measures import local_test_summary
from .utils import class_margins


def run_experiment(server):
    run = wandb.run
    if not tracking_enabled(run):
        raise ValueError("FedSA requires an active W&B run to save experiment results")
    config = getattr(server, "experiment_config", None)
    batch = None
    if config and config.get("batch_protocol", {}).get("fedsa_batch") is True:
        from .batch import BatchReporter
        from ..fedavg.reporting import synchronize
        batch = BatchReporter(server)
    metadata = dict(
        algorithm="fedsa",
        overall_test_definition="equal-client mean accuracy on the same complete global test set",
        primary_prediction="MSE nearest current-round global prototype",
        prototype_candidates="all uploaded global prototype classes; no local-class restriction",
        missing_class_distance="positive infinity",
        initial_prototype_evaluation="unavailable before the first aggregation; no fabricated accuracy",
        classifier_prediction="unrestricted classifier argmax; auxiliary metric only",
        global_model=False,
        feature_location="existing project classifier input; no added projection or normalization",
        feature_dim=server.semantic_anchors.shape[1],
        initialization="common project model initialization; client weights and prototypes persist independently",
        optimizer_state="SGD rebuilt at every client participation; shared project schedule",
        prototype_aggregation="equal contributor mean; current-round uploads only",
        prototype_collection="all local epochs; online pre-step features in training mode",
        anchor_initialization="CPU Linear/SyncBN/ReLU twice, training mode, detached output",
        anchor_seed=server.cfg["anchor_seed"],
        anchor_momentum=server.cfg["anchor_momentum"],
        initial_anchor_margins=class_margins(server.semantic_anchors),
        gap_timing="once before local training; previous participation prototypes and current anchors",
        local_gap_reporting="negative infinity retained in loss; finite mean excludes those clients",
        train_loss_averaging="equal batch mean per selected client, then equal selected-client mean",
        protocol_adaptations="project loaders, model, sampling, SGD parameters and scheduler; exact n_rounds, post-round evaluation",
        slow_client_simulation=False,
        random_join_ratio=False,
        early_stopping=False,
        evaluation_clients=server.n_clients,
        client_std_definition="population standard deviation across clients, not experiment seeds",
        accuracy_units="fraction in [0, 1]",
        selection="fixed final round; no validation-best selection",
        groups=server.groups,
        train_drop_last={str(index): bool(loaders["train"].drop_last)
                         for index, loaders in server.data_distributed["local"].items()},
    )
    partition = server.data_distributed.get("partition_metadata")
    if partition is not None:
        metadata["partition"] = partition
    if "split_manifest" in server.data_distributed:
        metadata.update(split_sha256=server.data_distributed["split_manifest"]["sha256"],
                        split_manifest_path=server.data_distributed["split_manifest_path"])
    if batch is not None:
        metadata.update(evaluation_rounds=batch.evaluation_rounds,
                        evaluation_interval=config["batch_protocol"]["eval_interval"],
                        metric_files=True,
                        communication="training anchors/prototypes; evaluation and initialization separate")
    run.config.update({"fedsa_runtime": metadata})
    run.define_metric("round")
    run.define_metric("*", step_metric="round")

    for round_idx in range(server.n_rounds):
        sampled = server._client_sampling(round_idx)
        server.server_results["client_history"].append(sampled)
        learning_rate = server.optimizer.param_groups[0]["lr"]
        if batch is not None:
            synchronize(server.device)
        started = time.perf_counter()
        train_results = server._clients_training(sampled)
        if batch is not None:
            synchronize(server.device)
        training_seconds = time.perf_counter() - started
        evaluation_started = time.perf_counter()
        evaluate_now = batch is None or round_idx + 1 in batch.evaluation_rounds
        evaluation = server.evaluate_clients() if evaluate_now else None
        if batch is not None and evaluate_now:
            synchronize(server.device)
        evaluation_seconds = time.perf_counter() - evaluation_started if evaluate_now else 0.0
        metrics = dict(round=round_idx + 1)
        metrics.update({
            "train/learning_rate": learning_rate,
            "train/local_gap_finite_clients": len(train_results.get("local_gap", [])),
            "prototype/coverage": len(server.global_prototypes) / server.num_classes,
        })
        if evaluation is not None:
            prototype, classifier = evaluation["prototype"], evaluation["classifier"]
            if prototype is None:
                raise ValueError("Cannot record FedSA overall accuracy without global prototypes")
            metrics.update({
                "overall_test_accuracy": prototype["accuracy"],
                "server_test_acc": prototype["accuracy"],
                "global/top1": prototype["accuracy"],
                "global/macro_f1": prototype["macro_f1"],
                "global/worst20_recall": prototype["worst20_recall"],
                "classifier_test_accuracy": classifier["accuracy"],
            })
            metrics.update(flatten_metrics(prototype, "test/prototype"))
            metrics.update(flatten_metrics(classifier, "test/classifier"))
        for key, values in train_results.items():
            metrics["train/" + key] = sum(values) / len(values)
        metrics.update(flatten_metrics([int(index) for index in sampled], "sampled_clients"))
        metrics.update(flatten_metrics(class_margins(server.semantic_anchors), "anchors/margin"))
        # Preserve the source's zero-filled prototype diagnostic, never use it
        # as a replacement for the actual sparse global prototype dictionary.
        prototype_centers = torch.zeros_like(server.semantic_anchors)
        for label, value in server.global_prototypes.items():
            prototype_centers[label] = value
        metrics.update(flatten_metrics(class_margins(prototype_centers), "prototype/margin"))
        if evaluation is not None:
            for record in evaluation["clients"]:
                prefix = "clients/{:03d}/".format(record["client_idx"])
                metrics[prefix + "prototype_accuracy"] = record["prototype"]["accuracy"]
                metrics[prefix + "classifier_accuracy"] = record["classifier"]["accuracy"]
        server.step_scheduler()
        metrics["time/round_seconds"] = time.perf_counter() - started
        if batch is not None:
            batch.record(metrics, sampled, training_seconds, evaluation_seconds)
        log_round(run, metrics)
        if evaluation is not None:
            server.server_results["test_accuracy"].append(prototype["accuracy"])
            server.server_results["classifier_test_accuracy"].append(classifier["accuracy"])
            print("[Round {}/{}] Overall test acc={:.4f}, classifier acc={:.4f} "
                  "(all {} clients), elapsed={:.1f}s".format(
                      round_idx + 1, server.n_rounds, prototype["accuracy"],
                      classifier["accuracy"], server.n_clients, metrics["time/round_seconds"]),
                  flush=True)
        else:
            print("[Round {}/{}] Training complete; no scheduled test, elapsed={:.1f}s".format(
                round_idx + 1, server.n_rounds, metrics["time/round_seconds"]), flush=True)

    if batch is not None:
        synchronize(server.device)
    final_report_started = time.perf_counter()
    validation_loader = server.data_distributed["global"].get("validation")
    validation = (server.evaluate_clients(validation_loader)
                  if validation_loader is not None else None)
    local_tests = server.evaluate_clients(local=True)
    local_summary = local_test_summary(local_tests)
    history = server.server_results["test_accuracy"]
    summary = dict(
        completed_rounds=server.n_rounds,
        overall_test_accuracy=history[-1],
        final_top1=history[-1],
        last10_top1_mean=sum(history[-10:]) / len(history[-10:]),
        final_macro_f1=prototype["macro_f1"],
        final_worst20_recall=prototype["worst20_recall"],
        final_client_accuracy_std=prototype["client_accuracy_std"],
        final_classifier_test_accuracy=classifier["accuracy"],
        final_classifier_loss=classifier["cross_entropy"],
        evaluation_definition=metadata["overall_test_definition"],
        primary_prediction=metadata["primary_prediction"],
        selection=metadata["selection"],
    )
    if config is not None:
        summary["seed"] = config["train_setups"]["seed"]
    run.summary.update({"fedsa_final": dict(
        overall_test=evaluation, overall_validation=validation,
        local_test=local_tests, local_test_summary=local_summary,
        local_test_definition="sample-weighted accuracy on existing local test loaders; unavailable clients omitted",
    )})
    final_metrics = dict(overall_test=prototype, classifier_test=classifier)
    if validation is not None:
        final_metrics["overall_validation"] = validation["prototype"]
        final_metrics["classifier_validation"] = validation["classifier"]
    if local_summary is not None:
        final_metrics["local_test"] = local_summary
    if batch is not None:
        synchronize(server.device)
        batch.finish(summary, time.perf_counter() - final_report_started)
    log_final(run, final_metrics)
    run.summary.update(summary)
    server.batch_summary = summary
    return summary
