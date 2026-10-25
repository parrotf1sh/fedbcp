"""Scientific invariants and real CPU integration; no downloads or remote logging."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from algorithms.fedbpc_complete import Server
from algorithms.fedbpc_complete.config import resolve_config
from algorithms.fedbpc_complete.criterion import (
    PrototypeCompletionLoss, alignment_weights, completion_weights, distributional_replay)
from algorithms.fedbpc_complete.utils import (
    PrototypeBank, collect_moments, make_statistics_loader)
from algorithms.fedavg.Server import Server as FedAvgServer
from fedavg_batch import run_tracked
from train_tools.models.fedavgnet import FedAvgNetCIFAR


class FeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.classifier = nn.Linear(2, 3)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(2))

    def forward(self, images, get_features=False):
        features = self.encoder(images)
        logits = self.classifier(features)
        return (logits, features) if get_features else logits


class TransformDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.targets, self.transform = images, labels, transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.targets[i]


def data_fixture():
    images = torch.tensor([[2., 0.], [1., 0.], [0., 2.], [0., 1.],
                           [-1., -1.], [-2., -1.]])
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    local = {}
    for k, ids in enumerate(([0, 1, 2], [3, 4, 5])):
        dataset = TensorDataset(images[list(ids)], targets[list(ids)])
        local[k] = dict(datasize=3,
            train=DataLoader(dataset, batch_size=2, shuffle=True,
                generator=torch.Generator().manual_seed(500 + k)),
            calibration=DataLoader(dataset, batch_size=2, shuffle=False,
                generator=torch.Generator().manual_seed(700 + k)))
    return dict(local=local, num_classes=3,
                global_=None, data_map=np.array([[2, 1, 0], [0, 1, 2]]),
                **{"global": {"test": DataLoader(TensorDataset(images, targets), batch_size=3)}})


def server_fixture(params=None):
    torch.manual_seed(12)
    model = FeatureModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    return Server(params or {}, model, data_fixture(), optimizer, scheduler,
                  n_rounds=2, sample_ratio=1, local_epochs=1, device="cpu")


class CompletionTests(unittest.TestCase):
    def test_exact_class_mass_and_conservation(self):
        prior = torch.tensor([1/3]*3)
        counts = torch.tensor([90., 10., 0.])
        for strength in (0., .5, 1.):
            real, replay = completion_weights(counts, prior, torch.ones(3), strength)
            mass = counts / counts.sum() * real + replay
            expected = (1-strength) * counts / counts.sum() + strength * prior
            torch.testing.assert_close(mass, expected)
            self.assertAlmostEqual(mass.sum().item(), 1, places=6)
            self.assertTrue(((0 <= real) & (real <= 1)).all())

    def test_missing_prototype_does_not_remove_real_mass_without_replacement(self):
        counts, prior = torch.tensor([9., 1., 0.]), torch.ones(3)/3
        for reliability in (torch.zeros(3), torch.tensor([1., 0., .25])):
            real, replay = completion_weights(counts, prior, reliability, 1.)
            self.assertAlmostEqual(((counts/counts.sum())*real).sum().item()
                                   + replay.sum().item(), 1., places=6)
            self.assertEqual(replay[1].item(), 0.)
        real, replay = completion_weights(counts, prior, torch.zeros(3), 1.)
        torch.testing.assert_close(real, torch.ones(3))
        self.assertEqual(replay.sum().item(), 0)

    def test_iid_completion_is_identity(self):
        real, replay = completion_weights(torch.tensor([5., 5., 5.]),
            torch.ones(3)/3, torch.ones(3), 1)
        torch.testing.assert_close(real, torch.ones(3))
        self.assertEqual(replay.sum(), 0)

    def test_fixed_weights_survive_single_class_minibatch(self):
        loss_fn = PrototypeCompletionLoss()
        logits = torch.zeros(1, 3, requires_grad=True)
        features = torch.ones(1, 2, requires_grad=True)
        head = nn.Linear(2, 3)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        weights = torch.tensor([.25, 1., 1.])
        _, terms = loss_fn(logits, torch.tensor([0]), features, head,
            torch.ones(3, 2), weights, torch.zeros(3), torch.zeros(3), 0.)
        self.assertAlmostEqual(terms["real_ce"].item(), .25 * np.log(3), places=6)

    def test_replay_provides_missing_positive_head_gradient_only(self):
        model = FeatureModel()
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.zero_()
        inputs = torch.tensor([[1., 0.]])
        logits, features = model(inputs, get_features=True)
        prototypes = torch.tensor([[1., 0.], [0., 2.], [-1., -1.]], requires_grad=True)
        _, terms = PrototypeCompletionLoss()(logits, torch.tensor([0]), features,
            model.classifier, prototypes, torch.ones(3), torch.tensor([0., .3, 0.]),
            torch.zeros(3), 0.)
        terms["replay_ce"].backward()
        self.assertIsNone(model.encoder.weight.grad)
        self.assertIsNone(prototypes.grad)
        self.assertLess(model.classifier.weight.grad[1, 1].item(), 0.)
        self.assertGreater(model.classifier.weight.grad[0, 1].item(), 0.)

    def test_alignment_full_data_balances_present_classes_without_batch_normalizer(self):
        counts = torch.tensor([9., 1., 0.])
        weights = alignment_weights(counts, torch.ones(3))
        torch.testing.assert_close(counts/counts.sum() * weights, torch.tensor([.5, .5, 0.]))

    def test_gradient_routing_keeps_real_backbone_learning_and_reweights_head(self):
        torch.manual_seed(81)
        model = FeatureModel()
        images = torch.randn(5, 2)
        targets = torch.tensor([0, 0, 0, 0, 1])
        logits, features = model(images, get_features=True)
        weights = torch.tensor([.2, 1., 1.])
        natural = nn.functional.cross_entropy(logits, targets)
        weighted = (nn.functional.cross_entropy(logits, targets, reduction="none") * weights[targets]).mean()
        encoder_grad = torch.autograd.grad(natural, model.encoder.weight, retain_graph=True)[0]
        head_grad = torch.autograd.grad(weighted, model.classifier.weight, retain_graph=True)[0]
        loss, _ = PrototypeCompletionLoss()(logits, targets, features, model.classifier,
            torch.zeros(3, 2), weights, torch.zeros(3), torch.zeros(3), 0.)
        torch.testing.assert_close(loss, weighted)
        loss.backward()
        torch.testing.assert_close(model.encoder.weight.grad, encoder_grad)
        torch.testing.assert_close(model.classifier.weight.grad, head_grad)

    def test_invalid_configs_fail_fast(self):
        for params in ({"completion_strength": 1.1}, {"lambda_align": float("nan")},
                       {"warmup_rounds": True}, {"min_confidence": .05}):
            with self.assertRaises(ValueError):
                resolve_config(params)


class MomentTests(unittest.TestCase):
    def test_sample_weighted_raw_means_include_singletons(self):
        bank = PrototypeBank(2, 2, 1)
        payloads = [
            dict(feature_sums=torch.tensor([[2., 0.], [0., 0.]]),
                 feature_square_sums=torch.tensor([[4., 0.], [0., 0.]]),
                 class_counts=torch.tensor([1, 0]), encoder_version=4),
            dict(feature_sums=torch.tensor([[0., 6.], [0., 3.]]),
                 feature_square_sums=torch.tensor([[0., 12.], [0., 9.]]),
                 class_counts=torch.tensor([3, 1]), encoder_version=4)]
        update = bank.update(payloads, 4)
        torch.testing.assert_close(bank.means, torch.tensor([[.5, 1.5], [0., 3.]]))
        self.assertTrue(bank.valid.all())
        self.assertEqual(update["contributors"].tolist(), [2, 1])
        torch.testing.assert_close(bank.variances, torch.tensor([[1., 1.], [0., 0.]]))

    def test_between_client_scatter_and_zero_mean_are_preserved(self):
        bank = PrototypeBank(1, 1, 1)
        payloads = [dict(feature_sums=torch.tensor([[x]]),
                        feature_square_sums=torch.tensor([[x*x]]),
                        class_counts=torch.tensor([1]), encoder_version=0) for x in (-1., 1.)]
        bank.update(payloads, 0)
        self.assertTrue(bank.valid.item())
        self.assertEqual(bank.means.item(), 0.)
        self.assertEqual(bank.variances.item(), 2.)

    def test_staleness_refresh_and_expiry(self):
        bank = PrototypeBank(1, 2, 1)
        payload = dict(feature_sums=torch.tensor([[3., 4.]]),
                       feature_square_sums=torch.tensor([[9., 16.]]),
                       class_counts=torch.tensor([1]), encoder_version=0)
        bank.update([payload], 0)
        bank.update([], 1)
        self.assertEqual(bank.reliability.item(), .5)
        bank.update([], 2)
        self.assertFalse(bank.valid.item())
        self.assertEqual(bank.means.count_nonzero(), 0)
        payload["encoder_version"] = 3
        bank.update([payload], 3)
        self.assertEqual(bank.age.item(), 0)
        self.assertEqual(bank.reliability.item(), 1)

    def test_mixed_versions_rejected(self):
        bank = PrototypeBank(1, 2, 2)
        payload = dict(feature_sums=torch.ones(1, 2),
                       class_counts=torch.ones(1), encoder_version=0)
        with self.assertRaisesRegex(ValueError, "encoder versions"):
            bank.update([payload], 1)

    def test_all_clients_match_centralized_moment_without_encoder_drift(self):
        server = server_fixture()
        payloads = [collect_moments(server.model, loader, 3, "cpu", 0)
                    for loader in server.statistics_loaders.values()]
        bank = PrototypeBank(3, 2, 1)
        bank.update(payloads, 0)
        torch.testing.assert_close(bank.means,
            torch.tensor([[1.5, 0.], [0., 1.5], [-1.5, -1.]]))
        torch.testing.assert_close(bank.variances,
            torch.tensor([[.5, 0.], [0., .5], [.5, 0.]]))

    def test_stats_use_training_images_and_preserve_training_rng(self):
        model = FeatureModel()
        images, labels = torch.ones(3, 2), torch.tensor([0, 1, 2])
        def random_transform(x):
            return x + torch.rand_like(x) + np.random.rand() + random.random()
        train = DataLoader(TransformDataset(images, labels, random_transform),
            batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(15))
        test = DataLoader(TransformDataset(images + 100, labels, lambda x: x),
                          batch_size=2)
        loader = make_statistics_loader(train, test, 12)
        torch_state, np_state, py_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
        generator_state = train.generator.get_state()
        payload = collect_moments(model, loader, 3, "cpu", 0)
        torch.testing.assert_close(payload["feature_sums"], torch.ones(3, 2, dtype=torch.float64))
        self.assertIs(train.dataset.transform, random_transform)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertTrue(torch.equal(generator_state, train.generator.get_state()))
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertEqual(py_state, random.getstate())
        self.assertTrue(model.training)

    def test_cifar_feature_interface_runs(self):
        model = FedAvgNetCIFAR(10)
        loader = DataLoader(TensorDataset(torch.zeros(2, 3, 32, 32),
                                         torch.tensor([0, 1])), batch_size=2)
        payload = collect_moments(model, loader, 10, "cpu", 0)
        self.assertEqual(payload["feature_sums"].shape, (10, 512))
        self.assertEqual(payload["class_counts"].sum(), 2)


class DistributionTests(unittest.TestCase):
    def test_gaussian_risk_bound_and_same_mean_counterexample(self):
        head = nn.Linear(1, 2).double()
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.], [-1.]]))
            head.bias.zero_()
        means, targets = torch.tensor([[1.]], dtype=torch.float64), torch.tensor([0])
        zero_risk = distributional_replay(head, means, torch.zeros_like(means), targets, 1.)
        mean_only = distributional_replay(head, means, torch.ones_like(means), targets, 0.)
        bound = distributional_replay(head, means, torch.ones_like(means), targets, 1.)
        # Deterministic Gauss-Hermite integration of a one-dimensional Gaussian.
        nodes, weights = np.polynomial.hermite.hermgauss(80)
        samples = means + torch.from_numpy(nodes[:, None]) * np.sqrt(2)
        true_risk = (nn.functional.cross_entropy(head(samples),
            torch.zeros(80, dtype=torch.long), reduction="none")
            * torch.from_numpy(weights / np.sqrt(np.pi))).sum()
        torch.testing.assert_close(zero_risk, mean_only)
        self.assertGreater(true_risk.item(), mean_only.item() + .1)
        self.assertGreater(bound.item(), true_risk.item())

    def test_covariance_head_gradients_and_shift_invariance(self):
        torch.manual_seed(41)
        head = nn.Linear(3, 4).double()
        means = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        variance = torch.rand(2, 3, dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([1, 3])
        losses = distributional_replay(head, means, variance, targets, .7)
        direct_logits = head(means.detach())
        diffs = head.weight[None, :, :] - head.weight[targets, None, :]
        direct_logits = direct_logits + .35 * (variance.detach()[:, None, :] * diffs.square()).sum(2)
        direct = nn.functional.cross_entropy(direct_logits, targets, reduction="none")
        torch.testing.assert_close(losses, direct)
        expected = torch.autograd.grad(direct.sum(), head.weight, retain_graph=True)[0]
        losses.sum().backward()
        torch.testing.assert_close(head.weight.grad, expected)
        self.assertIsNone(means.grad)
        self.assertIsNone(variance.grad)
        with torch.no_grad():
            head.weight.add_(torch.tensor([[1., -2., 3.]], dtype=torch.float64))
            head.bias.add_(2.)
        torch.testing.assert_close(distributional_replay(head, means, variance, targets, .7), losses)


class IntegrationTests(unittest.TestCase):
    def test_existing_results_are_rejected_before_logging_or_data_access(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "metrics.csv"
            existing.write_text("previous results\n")
            config = {"batch_protocol": {"output_dir": directory},
                      "train_setups": {"algo": {"name": "fedbpc_complete"}}}
            with mock.patch("fedavg_batch.wandb.init") as init:
                with self.assertRaises(FileExistsError):
                    run_tracked(config, lambda _: None)
                init.assert_not_called()
            self.assertEqual(existing.read_text(), "previous results\n")

    def test_disabled_auxiliary_update_matches_fedavg(self):
        ours = server_fixture({"completion_strength": 0., "lambda_align": 0.})
        reference_model = copy.deepcopy(ours.model)
        optimizer = torch.optim.SGD(reference_model.parameters(), lr=.03, momentum=.9)
        reference = FedAvgServer({"metrics_only": True}, reference_model, data_fixture(),
            optimizer, None, n_rounds=2, sample_ratio=1, local_epochs=1, device="cpu")
        sampled = np.array([0, 1])
        ours.server_results["client_history"].append(sampled)
        reference.server_results["client_history"].append(sampled)
        w, sizes, _ = ours._clients_training(sampled)
        expected, ns, _ = reference._clients_training(sampled)
        for a, b in zip(w, expected):
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=1e-7, rtol=1e-6)
        self.assertEqual(sizes, ns)

    def test_two_round_report_and_checkpoint(self):
        server = server_fixture()
        with tempfile.TemporaryDirectory() as directory:
            server.experiment_config = {
                "batch_protocol": {"output_dir": directory, "save_checkpoints": True},
                "train_setups": {"seed": 12, "algo": {"name": "fedbpc_complete"}}}
            run = wandb.init(mode="disabled")
            try:
                server.run()
            finally:
                run.finish()
            self.assertEqual(server.batch_summary["completed_rounds"], 2)
            self.assertAlmostEqual(server.optimizer.param_groups[0]["lr"], .03*.9**2)
            self.assertTrue((Path(directory)/"metrics.csv").is_file())
            self.assertTrue((Path(directory)/"algorithm.json").is_file())
            self.assertEqual(len((Path(directory)/"clients.jsonl").read_text().splitlines()), 2)
            state = torch.load(Path(directory)/"final_state.pt", map_location="cpu")
            self.assertEqual(state["completed_rounds"], 2)
            self.assertTrue(torch.isfinite(state["prototype_bank"]["means"]).all())
            self.assertGreater(server.batch_summary["final_completion_mass"], 0)

    def test_reused_trainer_does_not_keep_momentum(self):
        server = server_fixture({"warmup_rounds": 0})
        server._set_client_data(0)
        server.client.download_global(server.model.state_dict(), server.optimizer.state_dict(),
            server.bank, server.global_prior, 0)
        server.client.train()
        self.assertTrue(server.client.optimizer.state)
        server._set_client_data(1)
        server.client.download_global(server.model.state_dict(), server.optimizer.state_dict(),
            server.bank, server.global_prior, 0)
        self.assertFalse(server.client.optimizer.state)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
