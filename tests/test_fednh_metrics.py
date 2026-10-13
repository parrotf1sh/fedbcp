"""Exercise FedNH control flow/math with NumPy tensor stand-ins, no PyTorch training."""
import contextlib
import copy
import csv
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tensor:
    def __init__(self, values, requires_grad=False):
        self.a = np.asarray(values)
        if self.a.dtype.kind == 'f':
            self.a = self.a.astype(np.float32)
        self.requires_grad = requires_grad
        self.device = 'cpu'
    @property
    def shape(self): return self.a.shape
    def to(self, *args, **kwargs): return self
    def detach(self): return self
    def cpu(self): return self
    def clone(self): return Tensor(self.a.copy(), self.requires_grad)
    def numel(self): return self.a.size
    def element_size(self): return self.a.itemsize
    def item(self): return self.a.item()
    def tolist(self): return self.a.tolist()
    def view(self, *shape): return Tensor(self.a.reshape(shape))
    def all(self): return Tensor(self.a.all())
    def sum(self): return Tensor(self.a.sum())
    def min(self): return Tensor(self.a.min())
    def max(self): return Tensor(self.a.max())
    def mean(self): return Tensor(self.a.mean())
    def argmax(self, axis): return Tensor(self.a.argmax(axis))
    def clamp(self, min): return Tensor(np.maximum(self.a, min))
    def copy_(self, other): self.a[...] = other.a; return self
    def __getitem__(self, key): return Tensor(self.a[key.a if isinstance(key, Tensor) else key])
    def __setitem__(self, key, value): self.a[key] = value.a if isinstance(value, Tensor) else value
    def __add__(self, other): return Tensor(self.a + raw(other))
    def __sub__(self, other): return Tensor(self.a - raw(other))
    def __mul__(self, other): return Tensor(self.a * raw(other))
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return Tensor(self.a / raw(other))
    def __eq__(self, other): return Tensor(self.a == raw(other))
    def backward(self): pass  # Never autograd or actual training.


def raw(value):
    return value.a if isinstance(value, Tensor) else value


def normalize(value, dim):
    norm = np.linalg.norm(value.a, axis=dim, keepdims=True)
    return Tensor(value.a / np.maximum(norm, 1e-12))


def fake_torch():
    functional = types.SimpleNamespace(normalize=normalize,
        softmax=lambda x, dim: Tensor(np.exp(x.a) / np.exp(x.a).sum(axis=dim, keepdims=True)))
    return types.SimpleNamespace(Tensor=Tensor, no_grad=lambda: lambda fn: fn,
        nn=types.SimpleNamespace(functional=functional, utils=types.SimpleNamespace(clip_grad_norm_=mock.Mock())),
        tensor=lambda x, **kwargs: Tensor(x), zeros=lambda shape, **kwargs: Tensor(np.zeros(shape, np.float32)),
        zeros_like=lambda x: Tensor(np.zeros_like(x.a)), isfinite=lambda x: Tensor(np.isfinite(x.a)),
        norm=lambda x, p=2, dim=None, keepdim=False: Tensor(np.linalg.norm(x.a, axis=dim, keepdims=keepdim)),
        sum=lambda x, dim=None, keepdim=False: Tensor(x.a.sum(axis=dim, keepdims=keepdim)),
        exp=lambda x: Tensor(np.exp(x.a)), div=lambda x, y: x / y,
        unique=lambda x: Tensor(np.unique(x.a)), cat=lambda xs, dim: Tensor(np.concatenate([x.a for x in xs], axis=dim)),
        gather=lambda x, dim, index: Tensor(np.take_along_axis(x.a, index.a, axis=dim)))


class Base:
    def _set_client_data(self, idx):
        self.client.datasize = self.data_distributed['local'][idx]['datasize']
    def _results_updater(self, result, local):
        for key, value in local.items(): result.setdefault(key, []).append(value)
        return result


@contextlib.contextmanager
def modules():
    torch = fake_torch()
    cfg = load_file('fednh_config_numeric', ROOT / 'src/algorithms/fednh/config.py')
    wandb = types.SimpleNamespace(run=types.SimpleNamespace(summary={}), define_metric=mock.Mock(), log=mock.Mock())
    stubs = {'torch': torch, 'torch.nn': torch.nn, 'torch.nn.functional': torch.nn.functional, 'wandb': wandb,
        'algorithms.BaseClientTrainer': types.SimpleNamespace(BaseClientTrainer=object),
        'algorithms.BaseServer': types.SimpleNamespace(BaseServer=Base), 'algorithms.fednh.config': cfg,
        'algorithms.fednh.model': types.SimpleNamespace(ModelWithNormalizedHead=object)}
    with mock.patch.dict(sys.modules, stubs):
        utils = load_file('fednh_numeric_utils', ROOT / 'src/algorithms/fednh/utils.py')
        report = load_file('fednh_numeric_report', ROOT / 'src/algorithms/fedavg/reporting.py')
        with mock.patch.dict(sys.modules, {'algorithms.fednh.utils': utils, 'algorithms.fedavg.reporting': report}):
            client = load_file('algorithms.fednh.client_numeric', ROOT / 'src/algorithms/fednh/ClientTrainer.py')
            with mock.patch.dict(sys.modules, {'algorithms.fednh.ClientTrainer': client}):
                server = load_file('algorithms.fednh.server_numeric', ROOT / 'src/algorithms/fednh/Server.py')
                with mock.patch.object(report, 'synchronize'):
                    yield cfg, utils, client, server, report


class MetricsTests(unittest.TestCase):
    def test_equal_model_update_and_count_weighted_smoothed_prototypes(self):
        with modules() as (config, utils, _, __, ___):
            cfg = config.resolve_config({})
            state = {'w': Tensor([0.]), 'scaling': Tensor([1.]), 'prototype': Tensor([[1., 0.], [0., 1.]])}
            weights = [dict(state, w=Tensor([1.]), scaling=Tensor([1.])),
                       dict(state, w=Tensor([9.]), scaling=Tensor([3.]))]
            payloads = [{'scaled_prototype': Tensor([[1., 0.], [0., 0.]]), 'count_by_class_full': Tensor([1., 1e-12])},
                        {'scaled_prototype': Tensor([[0., 9.], [0., 0.]]), 'count_by_class_full': Tensor([9., 1e-12])}]
            updated = utils.aggregate_weights(state, weights, payloads, cfg, 0, set())
            self.assertAlmostEqual(updated['w'].item(), 5.)  # Equal clients, not 8.2 sample weighting.
            self.assertAlmostEqual(updated['scaling'].item(), 2.)
            mean = np.array([.1, .9]); mean /= np.linalg.norm(mean)
            expected = .9 * np.array([1., 0.]) + .1 * mean; expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(updated['prototype'].a[0], expected, atol=1e-6)
            np.testing.assert_allclose(updated['prototype'].a[1], [0., 1.])
            zero = utils.aggregate_weights(state, weights, payloads, dict(cfg, smoothing=0), 0, set())
            np.testing.assert_array_equal(zero['prototype'].a[1], [0., 0.])  # Preserve the original missing-class formula.
            decayed = utils.aggregate_weights(state, weights, payloads, dict(cfg, server_lr=.5, server_lr_decay=.5), 1, set())
            self.assertAlmostEqual(decayed['w'].item(), 1.25)
            np.testing.assert_array_equal(state['prototype'].a, [[1., 0.], [0., 1.]])

    def test_online_metrics_prototype_pass_and_advanced_estimator(self):
        with modules() as (config, _, client_module, __, ___):
            client = client_module.ClientTrainer.__new__(client_module.ClientTrainer)
            client.cfg = config.resolve_config({'metrics_only': True})
            client.datasize, client.local_epochs, client.device = 3, 2, 'cpu'
            client.round_idx, client.class_counts = 0, [2, 1, 0]
            client.trainloader = [(Tensor([[1., 0.], [0., 1.]]), Tensor([0, 0])), (Tensor([[1., 0.]]), Tensor([1]))]
            client.model = mock.Mock()
            client.model.prototype = Tensor(np.zeros((3, 2), np.float32))
            client.model.parameters.return_value = [Tensor([1.], requires_grad=True)]
            client.model.side_effect = lambda data, get_features=False: (Tensor(np.zeros((data.shape[0], 3), np.float32)), data) if get_features else Tensor(np.zeros((data.shape[0], 3), np.float32))
            client.optimizer = mock.Mock()
            client.criterion = mock.Mock(side_effect=[Tensor(v) for v in (1., 4., 1., 4.)])
            client._get_local_stats = mock.Mock(side_effect=AssertionError('extra evaluation'))
            totals, size = client.train()
            self.assertEqual((size, totals['seen'], totals['correct'], totals['prototype_samples']), (3, 6, 4, 3))
            self.assertEqual(totals['loss_sum'], 12.)
            self.assertEqual(client.model.call_count, 6)  # Four simulated train calls and two prototype calls.
            client._get_local_stats.assert_not_called()
            expected = [[2**.5, 2**.5], [1., 0.], [0., 0.]]
            np.testing.assert_allclose(client.prototype_payload['scaled_prototype'].a, expected, atol=1e-6)
            client.prototype_samples = 0
            advanced = client._estimate_prototype_adv()
            np.testing.assert_allclose(advanced['adv_agg_prototype'].a[0], [2**-.5, 2**-.5], atol=1e-6)
            self.assertEqual(client.prototype_samples, 3)
            client.criterion.side_effect = [Tensor(float('nan'))]
            with self.assertRaises(FloatingPointError): client.train()

    def test_optimizer_state_is_cleared_and_project_lr_used(self):
        with modules() as (config, utils, client_module, _, __):
            client = client_module.ClientTrainer.__new__(client_module.ClientTrainer)
            client.cfg = config.resolve_config({'metrics_only': True})
            client.model = mock.Mock()
            client.model.parameters.return_value = [Tensor([1.], requires_grad=True), Tensor([2.])]
            optimizer = mock.Mock(); optimizer.param_groups = [{'lr': 0.}]
            client.optimizer_class = mock.Mock(return_value=optimizer)
            client.initial_lr, client.n_rounds, client.device = .01, 300, 'cpu'
            source = {'state': {0: {'momentum_buffer': 'previous'}}, 'param_groups': [{'lr': .0099}]}
            optimizer.load_state_dict.side_effect = lambda state: optimizer.param_groups[0].update(lr=state['param_groups'][0]['lr'])
            client.download_global({'w': Tensor([1.])}, source, 1)
            self.assertEqual(optimizer.load_state_dict.call_args.args[0]['state'], {})
            self.assertEqual(optimizer.param_groups[0]['lr'], .0099)
            self.assertTrue(source['state'])
            self.assertEqual(utils.client_learning_rate(client.cfg, .01, 99, 300, .004), .004)

    def test_report_uses_updated_server_state_and_counts_payload_once(self):
        with modules() as (config, _, __, module, report):
            server = module.Server.__new__(module.Server)
            server.cfg = config.resolve_config({'metrics_only': True})
            server.num_classes, server.n_clients, server.device = 2, 2, 'cpu'
            server.server_weights = {'w': Tensor([0.]), 'scaling': Tensor([1.]), 'prototype': Tensor([[1., 0.], [0., 1.]])}
            server.exclude_layer_keys = set(); server.initial_private_weights = {}; server.client_private_weights = {}
            server.prototype_total_bytes = server.prototype_total_samples = 0; server.prototype_total_seconds = 0.
            server.server_results = {'client_history': [], 'test_accuracy': []}
            server.data_distributed = {'local': {0: {'datasize': 1}, 1: {'datasize': 9}}, 'data_map': [[1, 0], [0, 9]]}
            state = copy.deepcopy(server.server_weights)
            server.model = mock.Mock()
            server.model.state_dict.side_effect = lambda: state
            server.model.load_state_dict.side_effect = lambda weights: state.update(copy.deepcopy(weights))
            server.model.parameters.return_value = [Tensor([0.], True), Tensor([1.], True), Tensor([[1., 0.], [0., 1.]])]
            server.optimizer = mock.Mock(); server.optimizer.state_dict.return_value = {}; server.optimizer.param_groups = [{'lr': .01}]
            server.n_rounds, server.scheduler, server.testloader = 2, None, None
            server._print_start = lambda: None
            server._client_sampling = lambda step: [1, 0] if step == 0 else [0]
            downloads = []
            class Client:
                def download_global(self, weights, optimizer, step):
                    downloads.append(weights['w'].item()); self.weights = copy.deepcopy(weights)
                def train(self):
                    return {'loss_sum': self.datasize * 2., 'seen': self.datasize * 2, 'correct': self.datasize,
                            'prototype_samples': self.datasize, 'prototype_seconds': .01}, self.datasize
                def upload_local(self):
                    self.weights['w'] = self.weights['w'] + (1. if self.datasize == 1 else 9.)
                    return self.weights
                def upload_prototypes(self):
                    return {'scaled_prototype': Tensor([[float(self.class_counts[0]), 0.], [0., float(self.class_counts[1])]]),
                            'count_by_class_full': Tensor([float(c) if c else 1e-12 for c in self.class_counts])}
                def reset(self): pass
            server.client = Client()
            def evaluate(*args):
                return {'global/top1': .5, 'global/loss': 1., 'global/macro_f1': .4, 'global/worst20_recall': .3}
            with tempfile.TemporaryDirectory() as directory, mock.patch.object(report, 'log_partition'), \
                    mock.patch.object(report, 'evaluate', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
                server.experiment_config = {'batch_protocol': {'output_dir': directory}, 'train_setups': {'seed': 2022, 'algo': {'name': 'fednh'}}}
                server.run()
                with (Path(directory) / 'metrics.csv').open() as handle: rows = list(csv.DictReader(handle))
                self.assertEqual({p.suffix for p in Path(directory).iterdir()}, {'.csv', '.jsonl'})
            self.assertEqual(downloads, [0., 0., 5.])
            self.assertEqual(server.server_weights['w'].item(), 6.)
            self.assertEqual(state['w'].item(), 6.)
            # Full model: 24 bytes; payload: prototype 16 + counts 8 = 24 bytes per client.
            self.assertEqual(server.prototype_total_bytes, 72)
            self.assertEqual(server.batch_summary['communication_bytes'], 216)
            self.assertEqual(server.prototype_total_samples, 11)
            self.assertEqual(int(rows[1]['train/processed_samples']), 22)
            self.assertEqual(int(rows[0]['fednh/prototype_covered_classes']), 2)
            self.assertEqual(int(rows[1]['fednh/prototype_missing_classes']), 1)
            self.assertEqual(server.prototype_payloads, [])
            self.assertEqual(server.batch_summary['trainable_parameter_count'], 2)


if __name__ == '__main__':
    unittest.main()
