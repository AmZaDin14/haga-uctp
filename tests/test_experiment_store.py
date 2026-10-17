"""Behavioral storage regressions using small, deterministic run records."""

from copy import deepcopy
import errno
from functools import cached_property
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from src.experiment_store import ExperimentStore


class ExperimentStoreTests(unittest.TestCase):
    @cached_property
    def root(self) -> Path:
        directory = self.enterContext(tempfile.TemporaryDirectory())
        return Path(directory) / "experiment"

    @cached_property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment_id": "storage-regression",
            "stage": "smoke",
            "config": {
                "instances": [{
                    "id": "tiny",
                    "path": str(self.root.parent / "tiny.ctt"),
                    "sha256": "0" * 64,
                }],
                "algorithms": {"HAGA": {"pop_size": 2, "max_evaluations": 4}},
                "seeds": [7],
                "max_evals": 4,
                "pop_size": 2,
            },
            "provenance": {"fixture": "handcrafted"},
        }

    @property
    def run_path(self) -> Path:
        return self.root / "tiny" / "runs" / "HAGA" / "7.json"

    @property
    def failure_path(self) -> Path:
        return self.run_path.with_suffix(".failed.json")

    @staticmethod
    def record() -> dict[str, Any]:
        identity = {"instance": "tiny", "algorithm": "HAGA", "seed": 7}
        return {
            "schema_version": 1,
            **identity,
            "summary": {
                **identity,
                "best_hcv": 0,
                "best_scv": 3,
                "best_fitness": 3,
                "feasible": 1,
                "evaluations": 4,
                "generations": 1,
                "runtime_seconds": 0.125,
            },
            "history": [
                {
                    **identity,
                    "generation": 0,
                    "evaluations": 2,
                    "best_hcv": 1,
                    "best_scv": 5,
                    "best_fitness": 1005,
                    "mean_fitness": 1200.0,
                    "h_norm": 0.5,
                    "pc": 0.8,
                    "pm": 0.1,
                },
                {
                    **identity,
                    "generation": 1,
                    "evaluations": 4,
                    "best_hcv": 0,
                    "best_scv": 3,
                    "best_fitness": 3,
                    "mean_fitness": 3.5,
                    "h_norm": 0.25,
                    "pc": 0.8,
                    "pm": 0.1,
                },
            ],
        }

    def test_saved_run_survives_reopen(self):
        record = self.record()
        with ExperimentStore(self.root, self.manifest) as store:
            self.assertIsNone(store.load("tiny", "HAGA", 7))
            store.save(record)
            self.assertEqual(store.load("tiny", "HAGA", 7), record)
        with ExperimentStore(self.root, deepcopy(self.manifest)) as store:
            self.assertEqual(store.load("tiny", "HAGA", 7), record)
        self.assertEqual(json.loads((self.root / "manifest.json").read_text()), self.manifest)

    def test_manifest_mismatch_preserves_manifest_and_saved_data(self):
        with ExperimentStore(self.root, self.manifest) as store:
            store.save(self.record())
        manifest_before = (self.root / "manifest.json").read_bytes()
        run_before = self.run_path.read_bytes()
        variants = []
        for field, replacement in (("stage", "pilot"), ("experiment_id", "different")):
            changed = deepcopy(self.manifest)
            changed[field] = replacement
            variants.append(changed)
        changed = deepcopy(self.manifest)
        changed["config"]["max_evals"] = 5
        variants.append(changed)
        changed = deepcopy(self.manifest)
        changed["provenance"]["fixture"] = "different-code"
        variants.append(changed)
        for changed in variants:
            with self.subTest(manifest=changed):
                with self.assertRaisesRegex(ValueError, "manifest"):
                    with ExperimentStore(self.root, changed):
                        self.fail("A different manifest must not be accepted")
                self.assertEqual((self.root / "manifest.json").read_bytes(), manifest_before)
                self.assertEqual(self.run_path.read_bytes(), run_before)
        with ExperimentStore(self.root, self.manifest) as store:
            self.assertEqual(store.load("tiny", "HAGA", 7), self.record())

    def test_manifest_is_snapshotted_before_caller_mutation(self):
        supplied = deepcopy(self.manifest)
        store = ExperimentStore(self.root, supplied)
        supplied["config"]["max_evals"] = 100
        with store:
            store.save(self.record())
        self.assertEqual(json.loads((self.root / "manifest.json").read_text()), self.manifest)

    def test_existing_data_without_manifest_is_not_adopted(self):
        self.root.mkdir()
        old = self.root / "old-result.json"
        old.write_text('{"preserve": true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no manifest"):
            with ExperimentStore(self.root, self.manifest):
                self.fail("Existing unowned data must not be adopted")
        self.assertFalse((self.root / "manifest.json").exists())
        self.assertEqual(old.read_text(), '{"preserve": true}')

    def test_existing_lock_file_alone_allows_fresh_manifest(self):
        self.root.mkdir()
        (self.root / ".lock").touch()
        with ExperimentStore(self.root, self.manifest) as store:
            store.save(self.record())
        self.assertTrue((self.root / "manifest.json").is_file())

    def test_malformed_manifest_is_not_replaced(self):
        self.root.mkdir()
        path = self.root / "manifest.json"
        path.write_bytes(b'{"incomplete":')
        with self.assertRaisesRegex(ValueError, "manifest"):
            with ExperimentStore(self.root, self.manifest):
                self.fail("Malformed manifests must not be replaced")
        self.assertEqual(path.read_bytes(), b'{"incomplete":')

    def test_corrupt_or_incomplete_cached_runs_are_rerunnable(self):
        missing_history = self.record()
        missing_history.pop("history")
        short_budget = self.record()
        short_budget["summary"]["evaluations"] = 3
        wrong_identity = self.record()
        wrong_identity["summary"]["seed"] = 8
        corruptions = [
            b'{"incomplete":',
            b"\xff\xfe",
            b"null",
            b"[]",
            json.dumps(missing_history).encode(),
            json.dumps(short_budget).encode(),
            json.dumps(wrong_identity).encode(),
        ]
        with ExperimentStore(self.root, self.manifest) as store:
            self.run_path.parent.mkdir(parents=True, exist_ok=True)
            for content in corruptions:
                with self.subTest(content=content):
                    self.run_path.write_bytes(content)
                    self.assertIsNone(store.load("tiny", "HAGA", 7))
                    store.save(self.record())
                    self.assertEqual(store.load("tiny", "HAGA", 7), self.record())

    def test_abandoned_temporary_files_are_not_completed_runs(self):
        with ExperimentStore(self.root, self.manifest) as store:
            self.run_path.parent.mkdir(parents=True, exist_ok=True)
            abandoned = self.run_path.parent / ".7.json.abandoned.tmp"
            abandoned.write_text('{"partial":', encoding="utf-8")
            self.assertIsNone(store.load("tiny", "HAGA", 7))
            store.save(self.record())
        with ExperimentStore(self.root, self.manifest) as store:
            self.assertEqual(store.load("tiny", "HAGA", 7), self.record())
        self.assertEqual(abandoned.read_text(), '{"partial":')

    def test_failed_attempt_is_not_completion_and_success_clears_it(self):
        with ExperimentStore(self.root, self.manifest) as store:
            store.fail("tiny", "HAGA", 7, "worker failed")
            self.assertIsNone(store.load("tiny", "HAGA", 7))
            failure = json.loads(self.failure_path.read_text())
            self.assertEqual(failure["error"], "worker failed")
            self.assertEqual(failure["seed"], 7)
            self.assertFalse(self.run_path.exists())
        with ExperimentStore(self.root, self.manifest) as store:
            self.assertIsNone(store.load("tiny", "HAGA", 7))
            store.save(self.record())
            self.assertFalse(self.failure_path.exists())
            self.assertEqual(store.load("tiny", "HAGA", 7), self.record())

    def test_valid_completed_runs_cannot_be_overwritten(self):
        with ExperimentStore(self.root, self.manifest) as store:
            store.save(self.record())
            original = self.run_path.read_bytes()
            replacement = self.record()
            replacement["summary"]["runtime_seconds"] = 9.0
            with self.assertRaises(FileExistsError):
                store.save(replacement)
            self.assertEqual(self.run_path.read_bytes(), original)
            self.assertEqual(store.load("tiny", "HAGA", 7), self.record())

    def test_simultaneous_writer_is_rejected_and_lock_is_released(self):
        first = ExperimentStore(self.root, self.manifest)
        second = ExperimentStore(self.root, self.manifest)
        with first:
            with self.assertRaisesRegex(RuntimeError, "writer"):
                with second:
                    self.fail("Concurrent writers must not be accepted")
            first.save(self.record())
        with second:
            self.assertEqual(second.load("tiny", "HAGA", 7), self.record())

    def test_exception_in_context_releases_lock(self):
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with ExperimentStore(self.root, self.manifest):
                raise RuntimeError("deliberate")
        with ExperimentStore(self.root, self.manifest) as store:
            store.save(self.record())

    def test_operations_require_active_context(self):
        store = ExperimentStore(self.root, self.manifest)
        for operation in (
            lambda: store.load("tiny", "HAGA", 7),
            lambda: store.save(self.record()),
            lambda: store.fail("tiny", "HAGA", 7, "failure"),
        ):
            with self.assertRaises(RuntimeError):
                operation()
        with store:
            with self.assertRaises(RuntimeError):
                with store:
                    self.fail("A store cannot acquire its own context twice")
        with self.assertRaises(RuntimeError):
            store.load("tiny", "HAGA", 7)

    def test_invalid_caller_identities_are_not_cache_misses(self):
        identities = [
            ("../tiny", "HAGA", 7),
            ("/tiny", "HAGA", 7),
            ("tiny\\escape", "HAGA", 7),
            ("tiny", "../HAGA", 7),
            ("tiny", ".", 7),
            ("unknown", "HAGA", 7),
            ("tiny", "unknown", 7),
            ("tiny", "HAGA", 8),
            ("tiny", "HAGA", True),
            ("tiny", "HAGA", "7"),
        ]
        with ExperimentStore(self.root, self.manifest) as store:
            for identity in identities:
                with self.subTest(identity=identity):
                    with self.assertRaises(ValueError):
                        store.load(*identity)
                    with self.assertRaises(ValueError):
                        store.fail(*identity, "failure")
                    record = self.record()
                    record.update(zip(("instance", "algorithm", "seed"), identity))
                    with self.assertRaises(ValueError):
                        store.save(record)
        self.assertFalse(self.run_path.exists())

    def test_invalid_records_are_rejected_before_persistence(self):
        cases = []
        for field, values in {
            "evaluations": (3, 5, 4.0, True),
            "best_hcv": (-1, 0.0, False),
            "best_scv": (-1, 3.0),
            "best_fitness": (4, float("nan"), float("inf")),
            "generations": (-1, 1.0),
            "feasible": (0, 2, True),
            "runtime_seconds": (-1, float("nan"), float("inf"), "0.1"),
        }.items():
            for value in values:
                record = self.record()
                record["summary"][field] = value
                cases.append((f"summary {field}={value!r}", record))
        for field, value in (
            ("generation", -1),
            ("evaluations", 0),
            ("evaluations", 5),
            ("best_hcv", -1),
            ("best_scv", 5.0),
            ("best_fitness", 1006),
            ("mean_fitness", float("inf")),
            ("mean_fitness", True),
            ("h_norm", float("nan")),
            ("pc", "0.8"),
            ("pm", float("inf")),
        ):
            record = self.record()
            record["history"][0][field] = value
            cases.append((f"history {field}={value!r}", record))
        for field, value in (("evaluations", 3), ("generation", 2), ("best_scv", 4)):
            record = self.record()
            record["history"][-1][field] = value
            if field == "best_scv":
                record["history"][-1]["best_fitness"] = value
            cases.append((f"final {field} differs from summary", record))
        record = self.record()
        record["history"] = []
        cases.append(("empty history", record))
        record = self.record()
        record["history"][0]["seed"] = 8
        cases.append(("history identity mismatch", record))
        record = self.record()
        record["summary"].pop("runtime_seconds")
        cases.append(("missing summary field", record))
        record = self.record()
        record["schema_version"] = True
        cases.append(("invalid schema version", record))
        record = self.record()
        record["extra"] = (1, 2)
        cases.append(("non-JSON tuple", record))
        record = self.record()
        record["extra"] = record
        cases.append(("circular JSON", record))
        with ExperimentStore(self.root, self.manifest) as store:
            for name, record in cases:
                with self.subTest(case=name):
                    with self.assertRaises(ValueError):
                        store.save(record)
                    self.assertFalse(self.run_path.exists())

    def test_history_evaluation_counts_must_not_decrease(self):
        record = self.record()
        middle = deepcopy(record["history"][0])
        record["history"][0]["evaluations"] = 3
        middle["evaluations"] = 2
        record["history"].insert(1, middle)
        with ExperimentStore(self.root, self.manifest) as store:
            with self.assertRaisesRegex(ValueError, "evaluations"):
                store.save(record)

    def test_history_best_fitness_must_not_increase(self):
        record = self.record()
        middle = deepcopy(record["history"][0])
        middle["evaluations"] = 3
        middle["best_scv"] = 6
        middle["best_fitness"] = 1006
        record["history"].insert(1, middle)
        with ExperimentStore(self.root, self.manifest) as store:
            with self.assertRaisesRegex(ValueError, "nonincreasing"):
                store.save(record)

    def test_history_may_repeat_evaluation_counts(self):
        record = self.record()
        record["history"].insert(1, deepcopy(record["history"][0]))
        with ExperimentStore(self.root, self.manifest) as store:
            store.save(record)
            self.assertEqual(store.load("tiny", "HAGA", 7), record)

    def test_initial_population_can_be_the_entire_history(self):
        record = self.record()
        record["summary"]["generations"] = 0
        record["history"] = [record["history"][-1]]
        record["history"][0]["generation"] = 0
        with ExperimentStore(self.root, self.manifest) as store:
            store.save(record)
            self.assertEqual(store.load("tiny", "HAGA", 7), record)

    def test_corrupt_metric_history_cannot_count_as_cached_completion(self):
        corruptions = []
        for field, value in (
            ("evaluations", 0),
            ("best_fitness", 1006),
            ("mean_fitness", float("nan")),
            ("pc", float("inf")),
        ):
            record = self.record()
            record["history"][0][field] = value
            corruptions.append(record)
        with ExperimentStore(self.root, self.manifest) as store:
            self.run_path.parent.mkdir(parents=True, exist_ok=True)
            for record in corruptions:
                with self.subTest(record=record):
                    self.run_path.write_text(json.dumps(record), encoding="utf-8")
                    self.assertIsNone(store.load("tiny", "HAGA", 7))

    def test_permission_and_io_errors_are_not_cache_misses(self):
        with ExperimentStore(self.root, self.manifest) as store:
            for error in (
                PermissionError(errno.EACCES, "permission denied"),
                OSError(errno.EIO, "I/O error"),
            ):
                with self.subTest(error=error):
                    with patch("pathlib.Path.open", side_effect=error):
                        with self.assertRaises(type(error)):
                            store.load("tiny", "HAGA", 7)

    def test_failed_atomic_replace_cleans_temp_and_preserves_rerun(self):
        with ExperimentStore(self.root, self.manifest) as store:
            with patch("src.experiment_store.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    store.save(self.record())
            self.assertFalse(self.run_path.exists())
            self.assertEqual(list(self.run_path.parent.glob("*.tmp")), [])
            self.assertIsNone(store.load("tiny", "HAGA", 7))
            store.save(self.record())
            self.assertEqual(store.load("tiny", "HAGA", 7), self.record())

    def test_failed_fsync_cleans_temporary_file(self):
        with ExperimentStore(self.root, self.manifest) as store:
            with patch("src.experiment_store.os.fsync", side_effect=OSError("fsync failed")):
                with self.assertRaisesRegex(OSError, "fsync failed"):
                    store.save(self.record())
            self.assertFalse(self.run_path.exists())
            self.assertEqual(list(self.run_path.parent.glob("*.tmp")), [])

    def test_invalid_manifest_budget_or_identity_is_rejected(self):
        manifests = []
        for field, value in (("max_evals", 0), ("max_evals", True), ("pop_size", -1)):
            manifest = deepcopy(self.manifest)
            manifest["config"][field] = value
            manifests.append(manifest)
        for identity in ("..", "../tiny", "/absolute", ".lock", "manifest.json"):
            manifest = deepcopy(self.manifest)
            manifest["config"]["instances"][0]["id"] = identity
            manifests.append(manifest)
        for manifest in manifests:
            with self.subTest(manifest=manifest):
                with self.assertRaises(ValueError):
                    ExperimentStore(self.root, manifest)
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
