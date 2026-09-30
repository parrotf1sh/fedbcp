"""Label/index-only checks. No training, image loading or W&B required."""
from concurrent.futures import ProcessPoolExecutor
import copy
import importlib.util
import json
import multiprocessing
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/train_tools/preprocessing"))
import partition_cache as splits


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parallel_request(directory, seed=19940817):
    labels = np.repeat(np.arange(10), 500)
    train, _, metadata = splits.load_or_create_partition(
        labels, 20, {"method": "lda", "alpha": 0.3, "insufficient_policy": "repair"},
        "cifar10", directory, seed=seed, progress=False)
    return metadata["cache_hit"], metadata["indices_sha256"], [v.tolist() for v in train.values()]


def legacy_lda(labels, clients, alpha, seed):
    rng = np.random.RandomState(seed)
    for _ in range(1000):
        rows = [[] for _ in range(clients)]
        for cls in range(len(np.unique(labels))):
            indices = np.where(labels == cls)[0]
            rng.shuffle(indices)
            p = rng.dirichlet(np.repeat(alpha, clients))
            p = np.array([x * (len(row) < len(labels) / clients) for x, row in zip(p, rows)])
            cuts = (np.cumsum(p / p.sum()) * len(indices)).astype(int)[:-1]
            rows = [row + part.tolist() for row, part in zip(rows, np.split(indices, cuts))]
        if min(map(len, rows)) >= 10:
            for row in rows:
                rng.shuffle(row)
            return rows
    raise AssertionError("Test fixture should be feasible without repair")


class PartitionTests(unittest.TestCase):
    def test_strict_matches_legacy_successful_draw_exactly(self):
        labels = np.repeat(np.arange(10), 500)
        for seed in (1, 19940817):
            rows, metadata = splits.lda_indices(labels, 20,
                {"method": "lda", "alpha": 0.5, "insufficient_policy": "strict"}, seed, False)
            self.assertEqual([v.tolist() for v in rows.values()], legacy_lda(labels, 20, 0.5, seed))
            self.assertEqual(metadata["repaired_samples"], 0)

    def test_extreme_partition_is_bounded_and_repair_is_explicit(self):
        labels = np.repeat(np.arange(10), 5000)
        strict = {"method": "lda", "alpha": 0.03, "max_attempts": 8, "insufficient_policy": "strict"}
        with self.assertRaisesRegex(RuntimeError, "rejected all 8 attempts"):
            splits.lda_indices(labels, 100, strict, progress=False)
        rows, info = splits.lda_indices(labels, 100, dict(strict, insufficient_policy="repair"), progress=False)
        flattened = np.concatenate(list(rows.values()))
        np.testing.assert_array_equal(np.sort(flattened), np.arange(len(labels)))
        self.assertGreater(info["repaired_samples"], 0)
        self.assertGreaterEqual(min(map(len, rows.values())), 10)
        self.assertEqual(info["attempts"], 8)
        np.testing.assert_array_equal(np.bincount(labels[flattened]), np.bincount(labels))

    def test_local_rng_does_not_change_global_training_rng(self):
        before_numpy = copy.deepcopy(np.random.get_state())
        before_python = random.getstate()
        labels = np.repeat(np.arange(10), 500)
        splits.lda_indices(labels, 20, {"method": "lda", "alpha": 0.5}, progress=False)
        after = np.random.get_state()
        np.testing.assert_array_equal(before_numpy[1], after[1])
        self.assertEqual(before_numpy[2:], after[2:])
        self.assertEqual(before_python, random.getstate())

    def test_cache_reuse_never_calls_splitter(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = np.repeat(np.arange(10), 500)
            args = (labels, 20, {"method": "lda", "alpha": 0.5}, "cifar10", directory)
            first, _, meta = splits.load_or_create_partition(*args, progress=False)
            self.assertFalse(meta["cache_hit"])
            with mock.patch.object(splits, "lda_indices", side_effect=AssertionError("should use cache")):
                second, _, hit = splits.load_or_create_partition(*args, progress=False)
            self.assertTrue(hit["cache_hit"])
            self.assertEqual(meta["indices_sha256"], hit["indices_sha256"])
            for client in first:
                np.testing.assert_array_equal(first[client], second[client])

    def test_cache_key_includes_all_partition_inputs(self):
        labels = np.repeat(np.arange(10), 500)
        partition = {"method": "lda", "alpha": 0.1, "insufficient_policy": "strict"}
        key = splits.cache_identity("cifar10", labels, 20, partition)[0]
        variants = [
            splits.cache_identity("cifar100", labels, 20, partition)[0],
            splits.cache_identity("cifar10", labels[::-1], 20, partition)[0],
            splits.cache_identity("cifar10", labels, 30, partition)[0],
            splits.cache_identity("cifar10", labels, 20, partition, seed=1)[0],
        ]
        for change in ({"alpha": 0.3}, {"min_samples": 1}, {"max_attempts": 64}, {"insufficient_policy": "repair"}):
            variants.append(splits.cache_identity("cifar10", labels, 20, dict(partition, **change))[0])
        self.assertNotIn(key, variants)
        # Irrelevant inactive parameters do not create another split.
        self.assertEqual(key, splits.cache_identity("cifar10", labels, 20, dict(partition, shard_per_user=2))[0])

    def test_infeasible_partition_fails_immediately(self):
        with self.assertRaisesRegex(ValueError, "Not enough samples"):
            splits.lda_indices(np.repeat(np.arange(2), 20), 100, {"method": "lda", "alpha": 0.1}, progress=False)

    def test_paired_sharding_and_test_cache(self):
        labels, test_labels = np.repeat(np.arange(10), 100), np.repeat(np.arange(10), 20)
        with tempfile.TemporaryDirectory() as directory:
            train, test, metadata = splits.load_or_create_partition(
                labels, 20, {"method": "sharding", "shard_per_user": 2}, "cifar10", directory,
                test_targets=test_labels, progress=False)
            np.testing.assert_array_equal(np.sort(np.concatenate(list(train.values()))), np.arange(len(labels)))
            np.testing.assert_array_equal(np.sort(np.concatenate(list(test.values()))), np.arange(len(test_labels)))
            for client in train:
                self.assertEqual(set(labels[train[client]]), set(test_labels[test[client]]))
                self.assertLessEqual(len(set(labels[train[client]])), 2)
            _, again, hit = splits.load_or_create_partition(
                labels, 20, {"method": "sharding", "shard_per_user": 2}, "cifar10", directory,
                test_targets=test_labels, progress=False)
            self.assertTrue(hit["cache_hit"])
            self.assertEqual(metadata["indices_sha256"], hit["indices_sha256"])
            for client in test:
                np.testing.assert_array_equal(test[client], again[client])

    def test_corrupt_cache_is_quarantined_and_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = np.repeat(np.arange(10), 500)
            args = (labels, 20, {"method": "lda", "alpha": 0.5}, "cifar10", directory)
            _, _, first = splits.load_or_create_partition(*args, progress=False)
            Path(first["cache_path"]).write_bytes(b"broken archive")
            _, _, second = splits.load_or_create_partition(*args, progress=False)
            self.assertFalse(second["cache_hit"])
            self.assertEqual(first["indices_sha256"], second["indices_sha256"])
            self.assertEqual(len(list(Path(directory).glob("*.invalid-*.npz"))), 1)

    def test_detects_duplicate_indices_even_with_updated_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = np.repeat(np.arange(10), 500)
            args = (labels, 20, {"method": "lda", "alpha": 0.5}, "cifar10", directory)
            _, _, first = splits.load_or_create_partition(*args, progress=False)
            path = Path(first["cache_path"])
            with np.load(path, allow_pickle=False) as data:
                content = {key: data[key] for key in data.files}
            content["train_indices"][0] = content["train_indices"][1]
            metadata = json.loads(str(content["metadata"].item()))
            metadata["indices_sha256"] = splits._payload_digest([content[key] for key in
                ("train_indices", "train_offsets", "test_indices", "test_offsets")])
            content["metadata"] = np.asarray(json.dumps(metadata))
            np.savez_compressed(path, **content)
            _, _, second = splits.load_or_create_partition(*args, progress=False)
            self.assertFalse(second["cache_hit"])
            self.assertEqual(first["indices_sha256"], second["indices_sha256"])

    def test_concurrent_processes_share_one_cache_and_same_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = [pool.submit(parallel_request, directory) for _ in range(2)]
                results = [future.result(timeout=30) for future in futures]
            self.assertEqual(sorted(result[0] for result in results), [False, True])
            self.assertEqual(results[0][1:], results[1][1:])
            self.assertEqual(len(list(Path(directory).glob("*.npz"))), 1)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_warmup_deduplicates_66_runs_to_18_splits(self):
        warmup = load_script("prepare_partitions_test", ROOT / "scripts/prepare_partitions.py")
        import run_fedavg_experiments as fedavg
        import run_moon_experiments as moon
        first = warmup.unique_configs(fedavg.build_tasks("/data/paired"))
        second = warmup.unique_configs(moon.build_tasks("/data/paired"))
        self.assertEqual(len(first), 18)
        self.assertEqual([cfg["data_setups"] for cfg in first], [cfg["data_setups"] for cfg in second])


if __name__ == "__main__":
    unittest.main()
