"""Real SDK offline integration in a subprocess, with all HTTP requests forbidden."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('swanlab'), 'Install requirements-tracking.txt for real SDK tests')
class SDKTests(unittest.TestCase):
    def test_offline_lifecycle_and_migration_without_network(self):
        code = r'''
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock
import requests
import swanlab
import yaml
sys.path.insert(0, str(Path('src').resolve()))
import experiment_tracker as tracking
from fedavg_batch import run_tracked
spec = importlib.util.spec_from_file_location('migration', 'scripts/migrate_results_to_swanlab.py')
migration = importlib.util.module_from_spec(spec); spec.loader.exec_module(migration)
with tempfile.TemporaryDirectory() as temp, mock.patch.object(requests.Session, 'request', side_effect=AssertionError('Network forbidden')):
    root = Path(temp)
    output = root / 'results' / 'task' / 'attempt-123'
    config = {'train_setups': {'algo': {'name': 'fedavg'}, 'scenario': {'n_rounds': 2}},
              'batch_protocol': {'output_dir': str(output), 'task_id': 'task'},
              'swanlab_setups': {'project': 'offline-test', 'experiment_name': 'test',
                                'mode': 'offline', 'logdir': str(root / 'training-cache')}}
    def train(config):
        tracking.define_metric('round')
        tracking.define_metric('*', step_metric='round')
        tracking.log({'round': 1, 'global/top1': .4}, step=1)
        tracking.log({'round': 2, 'global/top1': .6}, step=2)
        (output / 'metrics.csv').write_text('round,global/top1\n1,0.4\n2,0.6\n')
        return {'completed_rounds': 2, 'final_top1': .6}
    with mock.patch('fedavg_batch.prepare_data'):
        run_tracked(config, train)
    result = json.loads((output / 'summary.json').read_text())
    assert result['status'] == 'success' and result['swanlab_url'] is None
    assert result['swanlab_run_id']
    config_files = list((root / 'training-cache').rglob('config.yaml'))
    assert config_files
    assert yaml.safe_load(config_files[0].read_text())['result_summary']['value']['final_top1'] == .6
    assert tracking.run is None
    original = {p: p.read_bytes() for p in output.iterdir()}
    item = migration.load_result(output)
    assert len(item['rows']) == 2
    assert migration.migrate([item], root / 'migration', 'offline-test', mode='offline') == 0
    assert migration.migrate([item], root / 'migration', 'offline-test', mode='offline') == 0
    assert original == {p: p.read_bytes() for p in output.iterdir()}
    assert list((root / 'migration').rglob('*.swanlab'))
    state = json.loads(next((root / 'migration').glob('*.json')).read_text())
    assert state['status'] == 'offline_saved'
    assert 'wandb' not in sys.modules
print('OFFLINE INTEGRATION PASSED')
'''
        completed = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                   capture_output=True, text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn('OFFLINE INTEGRATION PASSED', completed.stdout)


if __name__ == '__main__':
    unittest.main()
