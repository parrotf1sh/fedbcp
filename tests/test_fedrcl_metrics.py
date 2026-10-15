"""FedRCL reporting/optimizer control checks; stdlib stand-ins, no training."""
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

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Scalar:
    def __init__(self, value): self.value = value
    def __add__(self, other): return Scalar(self.value + (other.value if isinstance(other, Scalar) else other))
    __radd__ = __add__
    def __truediv__(self, value): return Scalar(self.value / value)
    def numel(self): return 1
    def element_size(self): return 4


class Optimizer:
    def __init__(self, params, lr):
        self.param_groups = [{'lr': lr}]
        self.state = {}
    def load_state_dict(self, state):
        self.param_groups = copy.deepcopy(state['param_groups'])
        self.state = copy.deepcopy(state.get('state', {}))


class LocalScheduler:
    def __init__(self, optimizer, lr_lambda):
        self.optimizer, self.lr_lambda, self.epoch = optimizer, lr_lambda, 0
        self.base = [group['initial_lr'] for group in optimizer.param_groups]
        self.update()
    def update(self):
        for base, group in zip(self.base, self.optimizer.param_groups):
            group['lr'] = base * self.lr_lambda(self.epoch)
    def step(self):
        self.epoch += 1
        self.update()


@contextlib.contextmanager
def modules():
    functional = types.SimpleNamespace()
    torch = types.SimpleNamespace(
        Tensor=Scalar, nn=types.SimpleNamespace(functional=functional),
        no_grad=lambda: lambda fn: fn,
        device=lambda device: types.SimpleNamespace(type=str(device).split(':')[0]),
        optim=types.SimpleNamespace(SGD=Optimizer, lr_scheduler=types.SimpleNamespace(LambdaLR=LocalScheduler)))
    wandb = types.SimpleNamespace(run=types.SimpleNamespace(summary={}), log=mock.Mock(), define_metric=mock.Mock())
    config = load_file('fedrcl_metrics_config', ROOT / 'src/algorithms/fedrcl/config.py')
    stubs = {'torch': torch, 'torch.nn.functional': functional, 'torch.nn': torch.nn,
             'torch.cuda.amp': types.SimpleNamespace(autocast=mock.Mock(), GradScaler=mock.Mock()),
             'wandb': wandb, 'algorithms.BaseServer': types.SimpleNamespace(BaseServer=object),
             'algorithms.BaseClientTrainer': types.SimpleNamespace(BaseClientTrainer=object),
             'algorithms.fedrcl.config': config,
             'algorithms.fedrcl.model': types.SimpleNamespace(ModelWithLayerFeatures=object),
             'algorithms.fedrcl.criterion': types.SimpleNamespace(CLLoss=object)}
    with mock.patch.dict(sys.modules, stubs):
        client = load_file('algorithms.fedrcl.client_metrics_test', ROOT / 'src/algorithms/fedrcl/ClientTrainer.py')
        report = load_file('fedrcl_shared_report_test', ROOT / 'src/algorithms/fedavg/reporting.py')
        with mock.patch.dict(sys.modules, {'algorithms.fedrcl.ClientTrainer': client,
                                          'algorithms.fedavg.reporting': report}):
            server = load_file('algorithms.fedrcl.server_metrics_test', ROOT / 'src/algorithms/fedrcl/Server.py')
            yield config, client, server, report


class FedRCLMetricsTests(unittest.TestCase):
    def test_equal_aggregation_independent_of_client_sizes(self):
        with modules() as (_, __, module, ___):
            server = module.Server.__new__(module.Server)
            weights = [{'w': Scalar(2.)}, {'w': Scalar(8.)}]
            self.assertEqual(server._aggregation(weights, [1, 9])['w'].value, 5.)
            self.assertEqual(server._aggregation(weights, [9, 1])['w'].value, 5.)
            self.assertEqual(weights[0]['w'].value, 2.)
            with self.assertRaises(ValueError): server._aggregation([], [])

    def test_local_scheduler_restarts_at_current_round_lr_and_clears_momentum(self):
        with modules() as (config, module, _, __):
            client = module.ClientTrainer.__new__(module.ClientTrainer)
            client.cfg = config.resolve_config({}, 4)
            client.model = mock.Mock()
            state = {'state': {0: {'momentum_buffer': 123}},
                     'param_groups': [{'lr': .004, 'initial_lr': .01}]}
            client.download_global({'w': 1}, state)
            first_optimizer = client.optimizer
            self.assertEqual(first_optimizer.state, {})
            self.assertEqual(first_optimizer.param_groups[0]['lr'], .004)
            for epoch in range(5):
                self.assertAlmostEqual(client.optimizer.param_groups[0]['lr'], .004 * .998 ** epoch)
                client.local_scheduler.step()
            client.download_global({'w': 2}, state)
            self.assertIsNot(client.optimizer, first_optimizer)
            self.assertEqual(client.optimizer.param_groups[0]['lr'], .004)
            self.assertEqual(state['param_groups'][0]['initial_lr'], .01)
            self.assertTrue(state['state'])
            self.assertEqual(client.get_weights(), {'cls': 1., 'cossim': 0., 'prox': 0., 'scl': 1., 'penalty': 1.})

    def test_round_report_and_summary_do_not_count_local_reference_as_communication(self):
        with modules() as (config, _, module, report):
            server = module.Server.__new__(module.Server)
            server.cfg = config.resolve_config({'metrics_only': True}, 4)
            server.device, server.n_rounds, server.local_epochs, server.num_classes = 'cpu', 2, 5, 2
            server.server_results = {'client_history': [], 'test_accuracy': []}
            state = {'w': Scalar(0.)}
            server.model = mock.Mock()
            server.model.state_dict.side_effect = lambda: state
            server.model.load_state_dict.side_effect = lambda update: state.update(copy.deepcopy(update))
            server.model.parameters.return_value = [Scalar(0.)]
            server.optimizer = mock.Mock()
            server.optimizer.state_dict.return_value = {}
            server.optimizer.param_groups = [{'lr': .01}]
            def decay(): server.optimizer.param_groups[0]['lr'] *= .99
            server.scheduler = mock.Mock(); server.scheduler.step.side_effect = decay
            server._print_start = lambda: None
            server._client_sampling = lambda step: [0, 1]
            server.testloader = None
            downloads = []
            def clients(sampled):
                downloads.append(state['w'].value)
                local = {'seen': [1, 3], 'correct': [1, 1], 'loss_sum': [6., 24.],
                         'cls_loss_sum': [1., 9.], 'scl_loss_sum': [2., 6.],
                         'penalty_loss_sum': [3., 9.], 'prox_loss_sum': [4., 12.],
                         'cossim_loss_sum': [5., 15.]}
                return [{'w': state['w'] + 2}, {'w': state['w'] + 8}], [1, 3], local
            server._clients_training = clients
            def evaluate(*args):
                return {'global/top1': .5, 'global/loss': 1., 'global/macro_f1': .4, 'global/worst20_recall': .3}
            with tempfile.TemporaryDirectory() as directory, mock.patch.object(report, 'log_partition'), \
                    mock.patch.object(report, 'evaluate', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
                server.experiment_config = {'batch_protocol': {'output_dir': directory},
                                           'train_setups': {'seed': 2022, 'algo': {'name': 'fedrcl'}}}
                server.run()
                with (Path(directory) / 'metrics.csv').open() as handle: rows = list(csv.DictReader(handle))
                self.assertEqual({p.suffix for p in Path(directory).iterdir()}, {'.csv', '.jsonl'})
            self.assertEqual(downloads, [0., 5.])
            self.assertEqual(state['w'].value, 10.)
            summary = server.batch_summary
            self.assertEqual(summary['communication_bytes'], 32)  # 2 rounds * 2 clients * 2 model transfers * 4 bytes.
            self.assertEqual(summary['final_train_ce_loss'], 2.5)
            self.assertEqual(summary['final_train_prox_loss'], 4.)
            self.assertEqual(summary['final_train_cossim_loss'], 5.)
            self.assertEqual(summary['final_train_total_loss'], 7.5)  # Zero-weight branches excluded.
            self.assertAlmostEqual(summary['final_local_first_epoch_lr'], .0099)
            self.assertAlmostEqual(summary['final_local_last_epoch_lr'], .0099 * .998 ** 4)
            self.assertEqual(rows[1]['train/ce_loss'], rows[1]['train/cls_loss'])
            self.assertFalse(summary['autocast_enabled'])
            self.assertFalse(summary['grad_scaler_enabled'])
            server.scheduler.step.assert_called_with()

    def test_amp_summary_and_metrics_mode_requirements(self):
        with modules() as (config, _, module, __):
            server = module.Server.__new__(module.Server)
            server.cfg = config.resolve_config({'metrics_only': True, 'use_amp': False}, 4)
            with self.assertRaisesRegex(ValueError, 'batch_protocol'): server.run()
            server.device = 'cuda:0'
            server.model = mock.Mock(); server.model.parameters.return_value = []
            metrics = {'train/total_loss': 1., 'fedrcl/local_first_epoch_lr': .01,
                       'fedrcl/local_last_epoch_lr': .009}
            summary = server.batch_summary_metrics(metrics)
            self.assertFalse(summary['autocast_enabled'])
            self.assertTrue(summary['grad_scaler_enabled'])
            server.cfg['use_amp'] = True
            self.assertTrue(server.batch_summary_metrics(metrics)['autocast_enabled'])


if __name__ == '__main__':
    unittest.main()
