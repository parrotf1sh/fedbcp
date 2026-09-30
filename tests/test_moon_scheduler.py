"""MOON queue and CPU-cache tests; no model training or dataset downloads."""
import contextlib
import importlib.util
import io
import json
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


moon = load_file("moon_queue_test", ROOT / "scripts/run_moon_experiments.py")
history = load_file("moon_history_test", ROOT / "src/algorithms/moon/history.py")


class MoonQueueTests(unittest.TestCase):
    def test_exact_pairing_with_fedavg(self):
        moon_tasks = moon.build_tasks("/data/paired")
        fedavg_tasks = moon.queue.build_tasks("/data/paired")
        self.assertEqual(len(fedavg_tasks), 66)
        fedavg_tasks = [t for t in fedavg_tasks if t["config"]["train_setups"]["seed"] in moon.TRAIN_SEEDS]
        self.assertEqual(len(moon_tasks), 44)
        self.assertEqual(len(fedavg_tasks), 44)
        self.assertEqual(sum("main" in t["categories"] for t in moon_tasks), 36)
        self.assertEqual(len({t["id"] for t in moon_tasks}), 44)
        groups = {}
        for ours, reference in zip(moon_tasks, fedavg_tasks):
            a, b = ours["config"], reference["config"]
            self.assertEqual(a["data_setups"], b["data_setups"])
            for key in ("scenario", "model", "optimizer", "scheduler", "seed"):
                self.assertEqual(a["train_setups"][key], b["train_setups"][key])
            for key in b["batch_protocol"]:
                self.assertEqual(a["batch_protocol"][key], b["batch_protocol"][key])
            self.assertEqual(ours["categories"], reference["categories"])
            self.assertNotEqual(ours["id"], reference["id"])
            self.assertEqual(a["train_setups"]["algo"],
                             {"name": "moon", "params": {"metrics_only": True, "mu": 0.1, "tau": 0.5}})
            self.assertEqual(a["batch_protocol"]["history_storage"], "cpu_memory_only")
            self.assertEqual(a["wandb_setups"]["project"], "Point1")
            self.assertIn("moon", a["wandb_setups"]["tags"])
            groups.setdefault(a["wandb_setups"]["group"], []).append(a["train_setups"]["seed"])
        self.assertEqual(len(groups), 22)
        self.assertTrue(all(seeds == [2022, 2023] for seeds in groups.values()))

    def test_retained_seeds_keep_existing_task_identity(self):
        current = moon.build_tasks("/data/paired")
        with mock.patch.object(moon, "TRAIN_SEEDS", [2022, 2023, 2024]):
            previous = moon.build_tasks("/data/paired")
        retained = [t for t in previous if t["config"]["train_setups"]["seed"] in moon.TRAIN_SEEDS]
        self.assertEqual(current, retained)

    def test_mu_tau_change_identity_without_mutating_fedavg(self):
        originals = {t["id"] for t in moon.build_tasks()}
        fedavg = moon.queue.build_tasks()
        for parameter, value in (("MU", 0.2), ("TAU", 0.7)):
            with mock.patch.object(moon, parameter, value):
                changed = moon.build_tasks()
            self.assertFalse(originals & {t["id"] for t in changed})
            self.assertTrue(all(t["config"]["train_setups"]["algo"]["params"][parameter.lower()] == value
                                for t in changed))
        self.assertEqual(fedavg, moon.queue.build_tasks())

    def test_invalid_contrastive_parameters(self):
        for parameter, value in (("MU", -1), ("MU", float("nan")), ("TAU", 0), ("TAU", float("inf"))):
            with mock.patch.object(moon, parameter, value), self.assertRaises(ValueError):
                moon.build_tasks()

    def test_head_settings_control_all_tasks(self):
        with mock.patch.object(moon, "N_ROUNDS", 400), mock.patch.object(moon, "LOCAL_EPOCHS", 2), \
                mock.patch.object(moon, "INCLUDE_SUPPLEMENTARY", False):
            tasks = moon.build_tasks()
        self.assertEqual(len(tasks), 36)
        self.assertTrue(all(t["config"]["train_setups"]["scenario"]["n_rounds"] == 400
                            and t["config"]["train_setups"]["scenario"]["local_epochs"] == 2 for t in tasks))

    def test_moon_failure_continues_and_success_skips(self):
        tasks = moon.build_tasks()[:2]
        calls = []

        def child(command, log, env):
            cfg = json.loads(Path(command[-1]).read_text())
            self.assertEqual(cfg["train_setups"]["algo"]["name"], "moon")
            calls.append(cfg)
            if len(calls) == 1:
                return 1
            moon.queue.write_json(Path(log).parent / "summary.json", {"status": "success", "final_top1": 0.6})
            return 0

        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(moon.queue, "run_child", side_effect=child), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            output = Path(temporary)
            self.assertEqual(moon.queue.execute(tasks, output), 1)
            self.assertEqual(len(calls), 2)
            self.assertEqual(moon.queue.execute(tasks, output), 0)
            self.assertEqual(len(calls), 3)


class FakeTensor:
    """Only the snapshot interface; this does not simulate model training."""
    def __init__(self, values, device="cuda", requires_grad=True):
        self.values = values
        self.device = device
        self.requires_grad = requires_grad

    def detach(self):
        return FakeTensor(self.values, self.device, False)

    def cpu(self):
        return FakeTensor(self.values, "cpu", self.requires_grad)

    def clone(self):
        return FakeTensor(list(self.values), self.device, self.requires_grad)

    def numel(self):
        return len(self.values)

    def element_size(self):
        return 4


class HistoryTests(unittest.TestCase):
    def test_initial_snapshot_is_detached_cpu_and_shared_readonly(self):
        source = {"weight": FakeTensor([1, 2])}
        cache = history.ClientHistory(source, 100)
        source["weight"].values[0] = 99
        self.assertEqual(cache.get(0)["weight"].values, [1, 2])
        self.assertEqual(cache.get(0)["weight"].device, "cpu")
        self.assertFalse(cache.get(0)["weight"].requires_grad)
        self.assertIs(cache.get(0), cache.get(99))
        self.assertEqual(cache.nbytes(), 8)

    def test_client_updates_do_not_alias_sources_or_other_clients(self):
        cache = history.ClientHistory({"weight": FakeTensor([0])}, 3)
        update = {"weight": FakeTensor([10])}
        cache.update(0, update, 2)
        update["weight"].values[0] = 20
        cache.update(1, update, 5)
        update["weight"].values[0] = 99
        self.assertEqual(cache.get(0)["weight"].values, [10])
        self.assertEqual(cache.get(1)["weight"].values, [20])
        self.assertEqual(cache.get(2)["weight"].values, [0])
        self.assertIsNone(cache.age(2, 7))
        self.assertEqual(cache.age(0, 7), 5)
        self.assertEqual(cache.age(1, 7), 2)
        self.assertEqual(cache.nbytes(), 12)
        cache.update(0, {"weight": FakeTensor([30])}, 7)
        self.assertEqual(cache.age(0, 8), 1)
        self.assertEqual(cache.nbytes(), 12)  # Replacement does not grow the cache.

    def test_client_index_validation(self):
        cache = history.ClientHistory({"w": FakeTensor([0])}, 2)
        for client in (-1, 2):
            with self.assertRaises(IndexError):
                cache.get(client)

    def test_server_updates_only_participating_clients(self):
        # Exercise the real server coordination using a fake client, not a network/model.
        modules = {
            "algorithms.moon.ClientTrainer": types.SimpleNamespace(ClientTrainer=object),
            "algorithms.moon.criterion": types.SimpleNamespace(ModelContrastiveLoss=object),
            "algorithms.moon.history": history,
            "algorithms.BaseServer": types.SimpleNamespace(BaseServer=object),
            "algorithms.measures": types.ModuleType("algorithms.measures"),
        }
        with mock.patch.dict(sys.modules, modules):
            module = load_file("moon_server_coordination", ROOT / "src/algorithms/moon/Server.py")
        server = module.Server.__new__(module.Server)
        server.history = history.ClientHistory({"w": FakeTensor([0])}, 3)
        server.model = types.SimpleNamespace(state_dict=lambda: {"w": FakeTensor([100])})
        server.optimizer = types.SimpleNamespace(state_dict=lambda: {})
        server.server_results = {"client_history": [[2, 0]]}
        loaded = []

        class Client:
            def download_global(self, weights, optimizer, previous):
                loaded.append((server.selected, previous["w"].values[0]))

            def train(self):
                return {"seen": 1}, 1

            def upload_local(self):
                return {"w": FakeTensor([server.selected + 10])}

            def reset(self):
                pass

        server.client = Client()
        server._set_client_data = lambda client: setattr(server, "selected", client)
        server._results_updater = lambda results, local: {"seen": results.get("seen", []) + [local["seen"]]}
        server._clients_training([2, 0])
        self.assertEqual(loaded, [(2, 0), (0, 0)])
        self.assertEqual(server.history.get(1)["w"].values, [0])
        self.assertEqual(server.history_metrics["moon/first_participations"], 2)
        server.server_results["client_history"].append([2])
        server._clients_training([2])
        self.assertEqual(loaded[-1], (2, 12))
        self.assertEqual(server.history_metrics["moon/history_age_mean"], 1)
        self.assertEqual(server.history.last_updated, {2: 1, 0: 0})


if __name__ == "__main__":
    unittest.main()
