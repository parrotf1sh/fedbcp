"""FedSC control-flow and metric checks with NumPy stand-ins, no training."""
import contextlib
import copy
import csv
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from test_fednh_metrics import Tensor, Base, raw, fake_torch, load_file, ROOT


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@contextlib.contextmanager
def modules():
    # Extend the shared stand-in only inside this context; do not affect other tests.
    methods = {
        '__len__': lambda x: len(x.a), '__index__': lambda x: int(x.item()),
        '__radd__': lambda x, y: x + y,
        '__matmul__': lambda x, y: Tensor(x.a @ raw(y)),
        'data': property(lambda x: x),
        'new_zeros': lambda x, shape: Tensor(np.zeros(shape, np.float32)),
        'unsqueeze': lambda x, dim: Tensor(np.expand_dims(x.a, dim)),
        'squeeze': lambda x, dim: Tensor(np.squeeze(x.a, dim)),
        'transpose': lambda x, a, b: Tensor(np.swapaxes(x.a, a, b)),
        'mean': lambda x, dim=None: Tensor(x.a.mean(axis=dim)),
        'size': lambda x, dim: x.shape[dim],
        'expand': lambda x, *shape: Tensor(np.broadcast_to(x.a, tuple(
            x.shape[i] if size == -1 else size for i, size in enumerate(shape)))),
        'fill_diagonal_': lambda x, value: np.fill_diagonal(x.a, value),
    }
    with contextlib.ExitStack() as stack:
        for name, value in methods.items():
            stack.enter_context(mock.patch.object(Tensor, name, value, create=True))
        torch = fake_torch()
        def mean(value, dim):
            if not isinstance(value, Tensor):
                raise TypeError('torch.mean requires Tensor, not list')
            return value.mean(dim)
        def cross_entropy(logits, targets, reduction='mean'):
            a = logits.a - logits.a.max(axis=1, keepdims=True)
            losses = np.log(np.exp(a).sum(axis=1)) - a[np.arange(len(a)), targets.a]
            return Tensor(losses.mean() if reduction == 'mean' else losses.sum())
        torch.mean = mean
        torch.stack = lambda xs: Tensor(np.stack([x.a for x in xs]))
        torch.from_numpy = lambda x: Tensor(x)
        torch.cdist = lambda x, y, p: Tensor(np.linalg.norm(x.a[:, None, :] - y.a[None, :, :], axis=-1))
        torch.min = lambda x, dim: (Tensor(x.a.min(axis=dim)), Tensor(x.a.argmin(axis=dim)))
        torch.long = 'long'
        torch.zeros = lambda shape, dtype=None, **kwargs: Tensor(np.zeros(
            shape, dtype=np.int64 if dtype == 'long' else np.float32))
        torch.optim = types.SimpleNamespace(SGD=mock.Mock())
        torch.nn.Module = Module
        torch.nn.CrossEntropyLoss = lambda: cross_entropy
        torch.nn.functional.cross_entropy = cross_entropy
        torch.nn.functional.mse_loss = lambda x, y: Tensor(((x.a - y.a) ** 2).mean())
        wandb = types.SimpleNamespace(run=types.SimpleNamespace(summary={}), define_metric=mock.Mock(), log=mock.Mock())
        config = load_file('fedsc_test_config', ROOT / 'src/algorithms/fedsc/config.py')
        stack.enter_context(mock.patch.dict(sys.modules, {
            'torch': torch, 'torch.nn': torch.nn, 'torch.nn.functional': torch.nn.functional,
            'wandb': wandb, 'algorithms.BaseServer': types.SimpleNamespace(BaseServer=Base),
            'algorithms.BaseClientTrainer': types.SimpleNamespace(BaseClientTrainer=object),
            'algorithms.fedsc.config': config,
        }))
        criterion = load_file('fedsc_test_criterion', ROOT / 'src/algorithms/fedsc/criterion.py')
        utils = load_file('fedsc_test_utils', ROOT / 'src/algorithms/fedsc/utils.py')
        report = load_file('fedsc_test_report', ROOT / 'src/algorithms/fedavg/reporting.py')
        stack.enter_context(mock.patch.dict(sys.modules, {
            'algorithms.fedsc.criterion': criterion, 'algorithms.fedsc.utils': utils,
            'algorithms.fedavg.reporting': report,
        }))
        client = load_file('algorithms.fedsc.client_test', ROOT / 'src/algorithms/fedsc/ClientTrainer.py')
        stack.enter_context(mock.patch.dict(sys.modules, {'algorithms.fedsc.ClientTrainer': client}))
        server = load_file('algorithms.fedsc.server_test', ROOT / 'src/algorithms/fedsc/Server.py')
        stack.enter_context(mock.patch.object(report, 'synchronize'))
        yield criterion, utils, client, server, report, torch


class FedSCMetricsTests(unittest.TestCase):
    def test_first_round_never_reads_placeholders_and_later_preserves_fallback(self):
        with modules() as (module, _, __, ___, ____, torch):
            loss = module.SemanticCollaborationLoss(collect_metrics=True)
            logits, features, labels = Tensor([[1., 0.], [0., 1.]]), Tensor([[1., 0.], [0., 1.]]), Tensor([0, 1])
            # Neither __getitem__ nor iteration is legal for round-zero placeholders.
            first, parts = loss(logits, features, labels, object(), object(), 0)
            self.assertEqual(parts['rpcl_loss'].item(), 0.)
            self.assertEqual(parts['cpdr_loss'].item(), 0.)
            self.assertEqual(loss.last_metrics, {'fallback_count': 0, 'lookup_count': 0, 'negative_pool_size': 0})
            prototypes = {0: [Tensor([1., 1.])], 1: [Tensor([1., 0.]), Tensor([0., 1.])]}
            total, parts = loss(logits, features, labels, prototypes, {}, 1)
            self.assertEqual(parts['cpdr_loss'].item(), 0.)
            self.assertGreater(parts['rpcl_loss'].item(), 0.)
            self.assertAlmostEqual(total.item(), sum(parts[k].item() for k in ('ce_loss', 'rpcl_loss', 'cpdr_loss')), places=5)
            self.assertEqual(loss.last_metrics, {'fallback_count': 2, 'lookup_count': 2, 'negative_pool_size': 3})
            plain = module.SemanticCollaborationLoss()
            other, _ = plain(logits, features, labels, prototypes, {}, 1)
            self.assertEqual(other.item(), total.item())
            with self.assertRaisesRegex(ValueError, 'nonempty negative'):
                loss(logits[:1], features[:1], labels[:1], {0: prototypes[0]}, {}, 1)

    def test_source_prototypes_position_weights_mutation_and_failures(self):
        with modules() as (_, utils, __, ___, ____, _____):
            local = [{0: Tensor([1., 0.])}, {0: Tensor([3., 0.]), 1: Tensor([0., 5.])}]
            relational = utils.get_relational_prototypes(local)
            np.testing.assert_array_equal(relational[0][0].a, [1., 0.])
            np.testing.assert_array_equal(relational[0][1].a, [3., 0.])
            distances = utils.get_distribution_distances([[1, 0], [1, 2]], 2)
            raw_weights = 1 / (1 + np.exp(-(np.array([1, 3]) / 4 - np.array(distances) / sum(distances))))
            weights = raw_weights / raw_weights.sum()
            consistent = utils.get_consistent_prototypes(relational, distances, [1, 3], 'cpu')
            # Class 1 came only from client 1 but still uses weight POSITION 0.
            np.testing.assert_allclose(relational[1][0].a, [0., 5 * weights[0]], atol=1e-6)
            np.testing.assert_allclose(consistent[0].a, [weights[0] + 3 * weights[1], 0.], atol=1e-6)
            with self.assertRaises(ValueError): utils.get_distribution_distances([[0, 0]], 2)
            with self.assertRaises(ValueError): utils.get_consistent_prototypes({}, [0., 0.], [1, 1], 'cpu')

    def test_client_last_epoch_only_sample_weighted_metrics_no_extra_forward(self):
        with modules() as (_, __, module, ___, ____, torch):
            client = module.ClientTrainer.__new__(module.ClientTrainer)
            client.metrics_only, client.device, client.round_idx = True, 'cpu', 1
            client.local_epochs, client.num_classes, client.datasize = 2, 2, 3
            client.relational_prototypes, client.consistent_prototypes = {}, {}
            client.trainloader = [(Tensor([[1.], [2.]]), Tensor([0, 0])), (Tensor([[3.]]), Tensor([1]))]
            client.model = mock.Mock()
            calls = []
            def forward(data, get_features):
                calls.append(1)
                return Tensor(np.tile([1., 0.], (len(data), 1))), data + 10 * len(calls)
            client.model.side_effect = forward
            client.optimizer = mock.Mock()
            client._get_local_stats = mock.Mock(side_effect=AssertionError('extra eval'))
            client.criterion = mock.Mock()
            def criterion(logits, features, targets, *args):
                value = 1. if len(targets) == 2 else 4.
                client.criterion.last_metrics = {'fallback_count': len(targets), 'lookup_count': len(targets), 'negative_pool_size': 3}
                return Tensor(value), {k: Tensor(value if k in ('ce_loss', 'total_loss') else 0.) for k in ('ce_loss', 'rpcl_loss', 'cpdr_loss', 'total_loss')}
            client.criterion.side_effect = criterion
            totals, size = client.train()
            self.assertEqual((size, totals['seen'], totals['correct']), (3, 6, 4))
            self.assertEqual(totals['ce_loss_sum'], 12.)  # 2 * (2 * 1 + 1 * 4).
            self.assertEqual(totals['loss_sum'], 12.)
            self.assertEqual(totals['prototype_samples'], 3)
            self.assertEqual(client.prototype_payload['class_counts'], [2, 1])
            self.assertEqual(len(calls), 4)
            self.assertAlmostEqual(client.prototype_payload['prototypes'][0].item(), 31.5)
            self.assertAlmostEqual(client.prototype_payload['prototypes'][1].item(), 43.)
            client._get_local_stats.assert_not_called()
            self.assertEqual(totals['fallback_count'], 6)
            self.assertEqual(totals['negative_pool_sum'], 12)
            # Fresh optimizer, server state loaded then cleared for every participation.
            optimizer = mock.Mock(); optimizer.state = {'momentum': 1}
            torch.optim.SGD.return_value = optimizer
            client.download_global({}, {'param_groups': [{'lr': .0099}]}, {}, {}, 2)
            torch.optim.SGD.assert_called_once()
            self.assertEqual(optimizer.state, {})
            optimizer.load_state_dict.assert_called_once_with({'param_groups': [{'lr': .0099}]})
            client.trainloader = []
            with self.assertRaisesRegex(ValueError, 'no training batches'): client.train()

    def test_report_two_rounds_payload_once_and_model_update(self):
        with modules() as (_, __, ___, module, report, _____):
            server = module.Server.__new__(module.Server)
            server.cfg = {'metrics_only': True, 'cl_tau': .05}
            server.num_classes, server.device, server.n_rounds = 3, 'cpu', 2
            server.prototype_total_bytes = server.total_fallbacks = server.total_lookups = 0
            server.prototype_total_seconds = 0.
            server.relational_prototypes = {i: Tensor([float('nan')] * 3) for i in range(3)}
            server.consistent_prototypes = copy.deepcopy(server.relational_prototypes)
            server.server_results = {'client_history': [], 'test_accuracy': []}
            server.data_distributed = {'local': {0: {'datasize': 1}, 1: {'datasize': 3}}}
            state = {'w': Tensor([0.])}
            server.model = mock.Mock()
            server.model.state_dict.side_effect = lambda: state
            server.model.load_state_dict.side_effect = lambda weights: state.update(copy.deepcopy(weights))
            server.model.parameters.return_value = [Tensor([0.], True)]
            server.optimizer = mock.Mock(); server.optimizer.state_dict.return_value = {}
            server.optimizer.param_groups = [{'lr': .01}]
            server.scheduler = server.testloader = None
            server._print_start = lambda: None
            server._client_sampling = lambda step: [0, 1]
            downloads = []
            class Client:
                def download_global(self, weights, optimizer, relational, consistent, step):
                    self.weights = copy.deepcopy(weights); self.step = step
                    downloads.append(weights['w'].item())
                def train(self):
                    stats = {'seen': self.datasize, 'correct': 0, 'loss_sum': self.datasize,
                             'fallback_count': self.datasize if self.step else 0,
                             'lookup_count': self.datasize if self.step else 0,
                             'negative_pool_sum': 2 if self.step else 0, 'negative_pool_max': 2 if self.step else 0,
                             'batches': 1, 'prototype_seconds': .01, 'prototype_samples': self.datasize}
                    stats.update({name + '_sum': self.datasize if name in ('ce_loss', 'total_loss') else 0.
                                  for name in ('ce_loss', 'rpcl_loss', 'cpdr_loss', 'total_loss')})
                    return stats, self.datasize
                def upload_local(self): return {'w': self.weights['w'] + self.datasize}
                def upload_prototypes(self):
                    label = 0 if self.datasize == 1 else 1
                    return {'prototypes': {label: Tensor([1., 0.])}, 'class_counts': [1, 0, 0] if label == 0 else [0, 3, 0]}
                def reset(self): pass
            server.client = Client()
            metrics = {'global/top1': .5, 'global/loss': 1., 'global/macro_f1': .4, 'global/worst20_recall': .3}
            with tempfile.TemporaryDirectory() as directory, mock.patch.object(report, 'log_partition'), \
                    mock.patch.object(report, 'evaluate', return_value=metrics), contextlib.redirect_stdout(io.StringIO()):
                server.experiment_config = {'batch_protocol': {'output_dir': directory}, 'train_setups': {'seed': 2022, 'algo': {'name': 'fedsc'}}}
                server.run()
                with (Path(directory) / 'metrics.csv').open() as handle: rows = list(csv.DictReader(handle))
                self.assertEqual({p.suffix for p in Path(directory).iterdir()}, {'.csv', '.jsonl'})
            self.assertEqual(downloads, [0., 0., 2.5, 2.5])
            self.assertEqual(state['w'].item(), 5.)
            # Round 1: down 2 clients * 2 maps * 3 classes * (12 tensor + 8 ID) = 240.
            # Round 2: down 2 * (2 * (8 tensor + 8 ID + 8 length) + 2 * (8 tensor + 8 ID)) = 160.
            # Each round uploads 2 * (8 tensor + 8 ID + 3 * 8 class counts) = 80.
            self.assertEqual(server.prototype_total_bytes, 560)
            self.assertEqual(server.batch_summary['communication_bytes'], 592)  # Plus four clients * two 4-byte models.
            self.assertEqual(float(rows[0]['fedsc/prototype_download_bytes']), 240)
            self.assertEqual(float(rows[1]['fedsc/prototype_download_bytes']), 160)
            self.assertEqual(float(rows[1]['fedsc/prototype_missing_classes']), 1)
            self.assertEqual(server.batch_summary['fallback_rate'], 1.)


if __name__ == '__main__':
    unittest.main()
