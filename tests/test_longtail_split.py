import importlib.util
from pathlib import Path
import random
import tempfile
import unittest


PATH = Path(__file__).resolve().parents[1] / "src/train_tools/preprocessing/longtail_split.py"
SPEC = importlib.util.spec_from_file_location("longtail_split", PATH)
split = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(split)


class ManifestTests(unittest.TestCase):
    def options(self, **overrides):
        settings = dict(n_clients=10, partition={"method": "iid"}, imbalance_factor=20,
                        validation_fraction=0.1, split_seed=7, class_order_seed=8)
        settings.update(overrides)
        return settings

    def test_disjoint_complete_and_reproducible(self):
        targets = [c for c in range(10) for _ in range(100)]
        first = split.build_manifest(targets, **self.options())
        self.assertEqual(first, split.build_manifest(targets, **self.options()))
        self.assertFalse(set(first["train_indices"]) & set(first["validation_indices"]))
        allocated = [i for row in first["client_indices"] for i in row]
        self.assertEqual(sorted(allocated), first["train_indices"])
        self.assertEqual(len(allocated), len(set(allocated)))
        self.assertEqual(len(first["validation_indices"]), 100)

    def test_integer_scarcity_is_not_hidden_by_duplication(self):
        targets = [c for c in range(100) for _ in range(500)]
        manifest = split.build_manifest(targets, **self.options(
            n_clients=100, imbalance_factor=100, validation_fraction=0.04))
        rare = min(range(100), key=lambda c: manifest["global_counts"][c])
        self.assertEqual(manifest["global_counts"][rare], 4)
        self.assertEqual(manifest["class_coverage"][rare], 4)
        for c in range(100):
            counts = [row[c] for row in manifest["data_map"]]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_dirichlet_conserves_and_uses_local_rng(self):
        targets = [c for c in range(10) for _ in range(100)]
        before = random.getstate()
        manifest = split.build_manifest(targets, **self.options(partition={"method": "lda", "alpha": 0.5}))
        self.assertEqual(before, random.getstate())
        split.validate_manifest(manifest, targets)
        self.assertTrue(all(len(row) >= 1 for row in manifest["client_indices"]))

    def test_validation_precedes_longtail_subsampling(self):
        targets = [c for c in range(10) for _ in range(100)]
        mild = split.build_manifest(targets, **self.options(imbalance_factor=10))
        severe = split.build_manifest(targets, **self.options(imbalance_factor=50))
        self.assertEqual(mild["validation_indices"], severe["validation_indices"])
        self.assertTrue(set(severe["train_indices"]).issubset(mild["train_indices"]))

    def test_same_training_pool_for_both_settings(self):
        targets = [c for c in range(10) for _ in range(100)]
        iid = split.build_manifest(targets, **self.options())
        lda = split.build_manifest(targets, **self.options(partition={"method": "lda", "alpha": 1}))
        self.assertEqual(iid["train_indices"], lda["train_indices"])
        self.assertEqual(iid["global_counts"], lda["global_counts"])

    def test_round_trip_and_incompatible_reuse(self):
        targets = [c for c in range(10) for _ in range(100)]
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "manifest.json")
            first = split.load_or_create_manifest(path, targets, **self.options())
            self.assertEqual(first, split.load_or_create_manifest(path, targets, **self.options()))
            with self.assertRaises(ValueError):
                split.load_or_create_manifest(path, targets, **self.options(imbalance_factor=5))
            with self.assertRaises(ValueError):
                split.load_or_create_manifest(path, list(reversed(targets)), **self.options())

    def test_detects_leakage_even_with_recomputed_checksum(self):
        targets = [c for c in range(10) for _ in range(100)]
        manifest = split.build_manifest(targets, **self.options())
        manifest["validation_indices"].append(manifest["train_indices"][0])
        manifest["sha256"] = split._digest({k: v for k, v in manifest.items() if k != "sha256"})
        with self.assertRaises(ValueError):
            split.validate_manifest(manifest, targets)

    def test_rejects_impossible_settings(self):
        targets = [0] * 10 + [1] * 10
        for options in (self.options(n_clients=100), self.options(imbalance_factor=0),
                        self.options(partition={"method": "lda", "alpha": 0}),
                        self.options(partition={"method": "unknown"})):
            with self.subTest(options=options), self.assertRaises(ValueError):
                split.build_manifest(targets, **options)

    def test_zero_validation_supported_by_splitter(self):
        manifest = split.build_manifest([0] * 20 + [1] * 20,
            **self.options(n_clients=2, imbalance_factor=1, validation_fraction=0))
        self.assertEqual(manifest["validation_indices"], [])
        self.assertEqual(len(manifest["train_indices"]), 40)


if __name__ == "__main__":
    unittest.main()
