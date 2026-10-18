"""FedHPro queue protocol checks using NumPy stand-ins; no PyTorch training."""
import contextlib
import copy
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from test_fednh_metrics import Tensor, fake_torch, load_file, ROOT


class Optimizer:
    def __init__(self, params, lr=0):
        self.state = {}; self.param_groups = [{'lr': lr}]
    def load_state_dict(self, state):
        self.state = copy.deepcopy(state.get('state', {}))
        self.param_groups = copy.deepcopy(state['param_groups'])
    def zero_grad(self, **kwargs): pass
    def step(self): pass


@contextlib.contextmanager
def modules():
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(Tensor, '__radd__', lambda x, y: x + y, create=True))
        stack.enter_context(mock.patch.object(Tensor, 'reshape', lambda x, *shape: Tensor(x.a.reshape(*shape)), create=True))
        stack.enter_context(mock.patch.object(Tensor, 'mean', lambda x, dim=None: Tensor(x.a.mean(axis=dim))))
        torch = fake_torch()
        torch.optim = types.SimpleNamespace(SGD=Optimizer)
        config = load_file('fedhpro_config_numeric', ROOT / 'src/algorithms/fedhpro/config.py')
        wandb = types.SimpleNamespace(run=types.SimpleNamespace(summary={}), define_metric=mock.Mock(), log=mock.Mock())
        stubs = {'torch': torch, 'torch.nn': torch.nn, 'torch.nn.functional': torch.nn.functional,
                 'torch.utils.data': types.SimpleNamespace(DataLoader=object, TensorDataset=object),
                 'wandb': wandb, 'algorithms.measures': types.SimpleNamespace(__all__=[]),
                 'algorithms.fedhpro.config': config,
                 'algorithms.fedhpro.augmentation': types.SimpleNamespace(AugmentParams=object, diff_augment=mock.Mock()),
                 'algorithms.fedhpro.criterion': types.SimpleNamespace(HyperPrototypeLoss=object, match_loss=mock.Mock(), match_meta=mock.Mock())}
        stack.enter_context(mock.patch.dict(sys.modules, stubs))
        base_client = load_file('algorithms.base_client_fedhpro_test', ROOT / 'src/algorithms/BaseClientTrainer.py')
        base_server = load_file('algorithms.base_server_fedhpro_test', ROOT / 'src/algorithms/BaseServer.py')
        batch = load_file('algorithms.fedhpro.batch_test', ROOT / 'src/algorithms/fedhpro/batch.py')
        utils = load_file('algorithms.fedhpro.utils_test', ROOT / 'src/algorithms/fedhpro/utils.py')
        report = load_file('fedhpro_shared_report_test', ROOT / 'src/algorithms/fedavg/reporting.py')
        stack.enter_context(mock.patch.dict(sys.modules, {
            'algorithms.BaseClientTrainer': base_client, 'algorithms.BaseServer': base_server,
            'algorithms.fedhpro.batch': batch, 'algorithms.fedhpro.utils': utils,
            'algorithms.fedavg.reporting': report}))
        client = load_file('algorithms.fedhpro.client_test', ROOT / 'src/algorithms/fedhpro/ClientTrainer.py')
        stack.enter_context(mock.patch.dict(sys.modules, {'algorithms.fedhpro.ClientTrainer': client}))
        server = load_file('algorithms.fedhpro.server_test', ROOT / 'src/algorithms/fedhpro/Server.py')
        stack.enter_context(mock.patch.object(report, 'synchronize'))
        yield config, client, server, report, batch, torch


class FedHProMetricsTests(unittest.TestCase):
    def test_dsa_mapping_uses_all_context_fields_and_no_wall_clock(self):
        with modules() as (config, client, _, __, batch, ___):
            context = [2022, 0, 2, 0, 1]
            expected = batch.derive_dsa_seed(*context)
            for index in range(5):
                other = context[:]; other[index] += 1
                self.assertNotEqual(expected, batch.derive_dsa_seed(*other))
            with self.assertRaises(ValueError): batch.derive_dsa_seed(True, 0, 0, 0, 0)
            trainer = client.ClientTrainer.__new__(client.ClientTrainer)
            trainer.cfg = config.resolve_config({'dsa_seed_mode': 'deterministic'})
            trainer.dsa_context = tuple(context[:3])
            with mock.patch.object(client.time, 'time', side_effect=AssertionError('wall clock used')):
                self.assertEqual(trainer._dsa_seed(0, 1), expected)
            trainer.dsa_context = None
            with self.assertRaises(RuntimeError): trainer._dsa_seed(0, 1)
            trainer.cfg['dsa_seed_mode'] = 'wall_clock'
            with mock.patch.object(client.time, 'time', return_value=123.456):
                self.assertEqual(trainer._dsa_seed(0, 1), 23456)
        script = "import runpy; f=runpy.run_path('src/algorithms/fedhpro/batch.py')['derive_dsa_seed']; print(f(2022,0,2,0,1))"
        for hash_seed in ('1', '777'):
            result = subprocess.check_output([sys.executable, '-B', '-c', script], cwd=ROOT,
                                             env=dict(os.environ, PYTHONHASHSEED=hash_seed), text=True)
            self.assertEqual(int(result), expected)

    def test_local_online_losses_skip_extra_evaluation_and_snapshot_is_independent(self):
        with modules() as (config, module, _, __, ___, torch):
            client = module.ClientTrainer.__new__(module.ClientTrainer)
            client.cfg = config.resolve_config({'metrics_only': True})
            client.device, client.num_classes, client.feature_dim = 'cpu', 2, 2
            client.datasize, client.local_epochs, client.k_margin = 3, 2, 0.
            client.hyper_prototypes = Tensor(np.ones((10, 2), np.float32))
            client.trainloader = [(Tensor([[1., 0.], [0., 1.]]), Tensor([0, 0])), (Tensor([[1., 0.]]), Tensor([1]))]
            client.model = mock.Mock()
            state = {'w': Tensor([1.])}
            client.model.state_dict.side_effect = lambda: state
            client.model.side_effect = lambda data, get_features: (Tensor(np.tile([1., 0.], (len(data.a), 1))), data)
            client.optimizer = Optimizer([], .01)
            torch.nn.functional.cross_entropy = lambda logits, targets: Tensor(1. if targets.numel() == 2 else 4.)
            def loss(ce, *args):
                return ce + 3, {'ce_loss': ce, 'hpcl_loss': Tensor(1.), 'hpal_loss': Tensor(2.), 'total_loss': ce + 3}
            client.criterion = loss
            client._get_local_stats = mock.Mock(side_effect=AssertionError('extra local evaluation'))
            totals, size = client.train()
            self.assertEqual((size, totals['seen'], totals['correct']), (3, 6, 4))
            self.assertEqual(totals['ce_loss_sum'], 12.)
            self.assertEqual(totals['loss_sum'], 30.)
            self.assertEqual(client.model.call_count, 4)
            first = client.upload_local()
            state['w'].a[0] = 99.
            self.assertEqual(first['w'].item(), 1.)  # Actual inherited BaseClientTrainer method.
            self.assertEqual(client.upload_local()['w'].item(), 99.)
            source = {'state': {0: {'momentum_buffer': 1}}, 'param_groups': [{'lr': .0099}]}
            client.download_global({}, source, client.hyper_prototypes)
            self.assertEqual(client.optimizer.state, {})
            self.assertEqual(client.optimizer.param_groups[0]['lr'], .0099)
            self.assertTrue(source['state'])
            client.trainloader = []
            with self.assertRaisesRegex(ValueError, 'no batches'): client.train()

    def test_real_gradient_then_training_then_meta_revision_then_hp_and_report(self):
        with modules() as (config, _, module, report, batch, __):
            server = module.Server.__new__(module.Server)
            server.cfg = config.resolve_config({'metrics_only': True, 'dsa_seed_mode': 'deterministic',
                                               'meta_rounds': 20, 'meta_epochs': 10, 'rounds_of_hp': 10})
            server.training_seed, server.device = 2022, 'cpu'
            server.num_classes, server.n_clients, server.local_epochs, server.n_rounds = 2, 2, 5, 2
            state = {'w': Tensor([0.]), 'classifier.weight': Tensor([1., 2.]), 'classifier.bias': Tensor([3.])}
            server.model = mock.Mock()
            server.model.state_dict.side_effect = lambda: state
            server.model.load_state_dict.side_effect = lambda update: state.update(copy.deepcopy(update))
            server.model.parameters.return_value = list(state.values())
            server.meta_weights = {'weight': Tensor([10., 20.]), 'bias': Tensor([30.])}
            server.hyper_prototypes = Tensor([[1., 2.], [3., 4.]])
            server.meta_prototypes = Tensor([[1., 2.], [3., 4.]])
            server.optimizer = mock.Mock(); server.optimizer.state_dict.return_value = {}
            server.optimizer.param_groups = [{'lr': .01}]
            server.scheduler = None; server.testloader = None
            server.server_results = {'client_history': [], 'test_accuracy': []}
            server.data_distributed = {'local': {0: {'datasize': 1, 'train': None}, 1: {'datasize': 3, 'train': None}},
                                       'global': {'test': None}, 'data_map': [[1, 0], [0, 3]]}
            server.stage_seconds = {}; server.total_stage_seconds = {}; server.total_extra_communication = 0
            server.prototype_metrics = {}
            server._print_start = lambda: None
            server._client_sampling = lambda step: [0, 1]
            events, contexts, snapshots = [], [], []
            class Client:
                def compute_gradient(self, weights, counts):
                    events.append('real'); contexts.append(self.dsa_context)
                    self.asserted_meta = weights['classifier.weight'][0].item()
                    return {0: Tensor([1., 2.])}, {0: [Tensor([1., 2.]), Tensor([3.])]}
                def download_global(self, weights, optimizer, hp):
                    events.append('download'); self.weights = copy.deepcopy(weights)
                def train(self):
                    events.append('local')
                    stats = {'seen': self.datasize, 'correct': self.datasize, 'loss_sum': 3. * self.datasize}
                    stats.update({key + '_sum': self.datasize if key != 'total_loss' else 3. * self.datasize
                                  for key in ('ce_loss', 'hpcl_loss', 'hpal_loss', 'total_loss')})
                    return stats, self.datasize
                def upload_local(self):
                    result = dict(self.weights, w=self.weights['w'] + self.datasize)
                    snapshots.append(copy.deepcopy(result)); return result
                def reset(self): pass
            server.client = Client()
            def update_meta():
                events.append('meta')
                self.assertEqual(len(server.client_meta_gradients), 2)
                self.assertEqual(server.global_meta_weights['classifier.weight'][0].item(), 10. if len(contexts) == 2 else 90.)
                server.prototype_metrics.update({'fedhpro/meta_matching_loss': 4., 'fedhpro/gradient_classes': 1})
            def revise(aggregated):
                events.append('revise')
                self.assertAlmostEqual(aggregated['w'].item(), 2.5 if len(contexts) == 2 else 5.)
                weights = {'weight': Tensor([90., 91.]), 'bias': Tensor([92.])}
                server.prototype_metrics['fedhpro/meta_classifier_loss'] = 5.
                return weights, module.replace_classifier(aggregated, weights)
            def update_hp():
                events.append('hp')
                self.assertEqual(state['classifier.weight'][0].item(), 90.)
                server.prototype_metrics['fedhpro/hyper_matching_loss'] = 6.
            server._update_meta_prototypes = update_meta
            server._revise_classifier = revise
            server._update_hyper_prototypes = update_hp
            def evaluate(*args):
                events.append('eval')
                self.assertEqual(state['classifier.bias'].item(), 92.)
                return {'global/top1': .5, 'global/loss': 1., 'global/macro_f1': .4, 'global/worst20_recall': .3}
            with tempfile.TemporaryDirectory() as directory, mock.patch.object(report, 'log_partition'), \
                    mock.patch.object(report, 'evaluate', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
                server.experiment_config = {'batch_protocol': {'output_dir': directory},
                                           'train_setups': {'seed': 2022, 'algo': {'name': 'fedhpro'}}}
                server.run()
                with (Path(directory) / 'metrics.csv').open() as handle: rows = list(csv.DictReader(handle))
                self.assertEqual({p.suffix for p in Path(directory).iterdir()}, {'.csv', '.jsonl'})
            self.assertEqual(events, ['real', 'download', 'local', 'real', 'download', 'local', 'meta', 'revise', 'hp', 'eval'] * 2)
            self.assertEqual(contexts, [(2022, 0, 0), (2022, 0, 1), (2022, 1, 0), (2022, 1, 1)])
            self.assertEqual(snapshots[0]['w'].item(), 1.)
            self.assertEqual(server.client_gradients, [])
            self.assertEqual(server.client_meta_gradients, [])
            self.assertIsNone(server.global_meta_weights)
            # Per client: model up/down 32; HP down 16; meta head down 12; feature gradient up 16; head gradient up 20.
            self.assertEqual(server.batch_summary['communication_bytes'], 4 * (32 + 16 + 12 + 16 + 20))
            self.assertEqual(server.total_extra_communication, 4 * 64)
            self.assertEqual(rows[0]['fedhpro/meta_matching_loss_last_step'], '4.0')
            self.assertEqual(rows[0]['fedhpro/classifier_updates_per_round'], '20')
            self.assertEqual(set(server.total_stage_seconds), {'real_gradient_seconds', 'local_train_seconds',
                'meta_matching_seconds', 'classifier_revision_seconds', 'hyper_matching_seconds'})


if __name__ == '__main__':
    unittest.main()
