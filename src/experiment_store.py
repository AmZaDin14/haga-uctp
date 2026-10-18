"""Locked, resumable storage for immutable experiment configurations and runs."""

from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, BinaryIO

from src.problem import HARD_CONSTRAINT_WEIGHT


class ExperimentStore:
    """Persist one experiment while holding an exclusive, nonblocking writer lock.

    Use as a context manager. Invalid caller data raises ``ValueError``; an
    incompatible manifest also raises ``ValueError``. Concurrent or unlocked
    use raises ``RuntimeError``. Valid completed runs cannot be overwritten
    (``FileExistsError``), but corrupt or incomplete cached runs can be replaced.
    Filesystem errors propagate, rather than turning an unreadable run into a
    cache miss. Failed-attempt files and abandoned temporary files are never
    completion records.
    """

    def __init__(self, root: Path, manifest: dict[str, Any]):
        self._root = Path(root)
        self._manifest_bytes = self._json_bytes(manifest)
        self._manifest = json.loads(self._manifest_bytes)
        self._instances, self._algorithms, self._seeds, self._max_evals = self._validate_manifest()
        self._lock_file: BinaryIO | None = None

    def __enter__(self) -> ExperimentStore:
        if self._lock_file is not None:
            raise RuntimeError("ExperimentStore is already inside a context")
        self._root.mkdir(parents=True, exist_ok=True)
        lock_file = (self._root / ".lock").open("a+b")
        try:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Experiment already has an active writer") from error
            self._open_manifest()
        except BaseException:
            lock_file.close()
            raise
        self._lock_file = lock_file
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        lock_file = self._require_context()
        self._lock_file = None
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    def load(self, instance: str, algorithm: str, seed: int) -> dict[str, Any] | None:
        """Return a validated completed run, or None for an absent/invalid cache."""
        self._require_context()
        path = self._run_path(instance, algorithm, seed)
        try:
            with path.open("r", encoding="utf-8") as source:
                record = json.load(source)
        except FileNotFoundError:
            return None
        except (ValueError, RecursionError):
            return None
        try:
            self._validate_record(record, (instance, algorithm, seed))
        except (ValueError, RecursionError):
            return None
        return record

    def save(self, record: dict[str, Any]) -> None:
        """Commit a complete run without replacing a valid existing result."""
        self._require_context()
        identity = self._validate_record(record)
        path = self._run_path(*identity)
        if self.load(*identity) is not None:
            raise FileExistsError(f"Completed run already exists: {path}")
        self._atomic_write(path, self._json_bytes(record))
        failure_path = path.with_suffix(".failed.json")
        try:
            failure_path.unlink()
        except FileNotFoundError:
            pass
        else:
            self._fsync_directory(path.parent)

    def fail(self, instance: str, algorithm: str, seed: int, error: str) -> None:
        """Record an unsuccessful attempt separately from any completed run."""
        self._require_context()
        path = self._run_path(instance, algorithm, seed).with_suffix(".failed.json")
        if not isinstance(error, str):
            raise ValueError("Failure error must be a string")
        record = {
            "schema_version": 1,
            "instance": instance,
            "algorithm": algorithm,
            "seed": seed,
            "error": error,
        }
        self._atomic_write(path, self._json_bytes(record))

    def _require_context(self) -> BinaryIO:
        if self._lock_file is None:
            raise RuntimeError("ExperimentStore methods require an active context")
        return self._lock_file

    def _open_manifest(self) -> None:
        path = self._root / "manifest.json"
        try:
            with path.open("r", encoding="utf-8") as source:
                existing = json.load(source)
        except FileNotFoundError:
            if any(entry.name != ".lock" for entry in self._root.iterdir()):
                raise ValueError("Experiment directory contains data but no manifest")
            self._atomic_write(path, self._manifest_bytes)
            return
        except (ValueError, RecursionError) as error:
            raise ValueError("Existing experiment manifest is malformed") from error
        try:
            matches = self._json_bytes(existing) == self._manifest_bytes
        except (ValueError, RecursionError) as error:
            raise ValueError("Existing experiment manifest is malformed") from error
        if not matches:
            raise ValueError("Existing experiment manifest does not match requested configuration")

    def _validate_manifest(self) -> tuple[set[str], set[str], set[int], int]:
        manifest = self._manifest
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be a dictionary")
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            raise ValueError("Manifest schema_version must be 1")
        if not isinstance(manifest.get("experiment_id"), str) or not manifest["experiment_id"]:
            raise ValueError("Manifest experiment_id must be a nonempty string")
        if manifest.get("stage") not in ("smoke", "pilot", "main"):
            raise ValueError("Manifest stage must be smoke, pilot, or main")
        if not isinstance(manifest.get("provenance"), dict):
            raise ValueError("Manifest provenance must be a dictionary")
        config = manifest.get("config")
        if not isinstance(config, dict):
            raise ValueError("Manifest config must be a dictionary")
        instances = config.get("instances")
        if not isinstance(instances, list) or not instances:
            raise ValueError("Manifest must contain instances")
        instance_ids: set[str] = set()
        for instance in instances:
            if not isinstance(instance, dict):
                raise ValueError("Manifest instance must be a dictionary")
            identity = self._safe_segment(instance.get("id"))
            if identity in (".lock", "manifest.json") or identity in instance_ids:
                raise ValueError("Manifest instance IDs must be unique and not reserved")
            path = instance.get("path")
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise ValueError("Manifest instance path must be absolute")
            if not isinstance(instance.get("sha256"), str) or not instance["sha256"]:
                raise ValueError("Manifest instance must contain a sha256 string")
            instance_ids.add(identity)
        algorithms = config.get("algorithms")
        if not isinstance(algorithms, dict) or not algorithms:
            raise ValueError("Manifest must contain algorithms")
        for name, parameters in algorithms.items():
            self._safe_segment(name)
            if not isinstance(parameters, dict):
                raise ValueError("Manifest algorithm parameters must be dictionaries")
        seeds = config.get("seeds")
        if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
            raise ValueError("Manifest seeds must be a nonempty list of integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("Manifest seeds must be unique")
        self._positive_int(config.get("max_evals"), "Manifest max_evals")
        self._positive_int(config.get("pop_size"), "Manifest pop_size")
        return instance_ids, set(algorithms), set(seeds), config["max_evals"]

    @staticmethod
    def _safe_segment(value: Any) -> str:
        if (
            not isinstance(value, str)
            or not value
            or value in (".", "..")
            or "/" in value
            or "\\" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError("Instance and algorithm IDs must be safe path segments")
        return value

    def _validate_identity(self, instance: Any, algorithm: Any, seed: Any) -> tuple[str, str, int]:
        instance = self._safe_segment(instance)
        algorithm = self._safe_segment(algorithm)
        if instance not in self._instances:
            raise ValueError(f"Instance is not in the manifest: {instance}")
        if algorithm not in self._algorithms:
            raise ValueError(f"Algorithm is not in the manifest: {algorithm}")
        if type(seed) is not int or seed not in self._seeds:
            raise ValueError("Seed is not an integer listed in the manifest")
        return instance, algorithm, seed

    def _run_path(self, instance: str, algorithm: str, seed: int) -> Path:
        self._validate_identity(instance, algorithm, seed)
        return self._root / instance / "runs" / algorithm / f"{seed}.json"

    def _validate_record(
        self, record: Any, expected: tuple[str, str, int] | None = None
    ) -> tuple[str, str, int]:
        if not isinstance(record, dict):
            raise ValueError("Run record must be a dictionary")
        self._validate_json(record)
        if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
            raise ValueError("Run schema_version must be 1")
        identity = self._validate_identity(record.get("instance"), record.get("algorithm"), record.get("seed"))
        if expected is not None and identity != expected:
            raise ValueError("Run record identity does not match its path")
        summary = record.get("summary")
        if not isinstance(summary, dict):
            raise ValueError("Run summary must be a dictionary")
        self._matching_identity(summary, identity)
        self._validate_metrics(summary)
        self._positive_int(summary.get("evaluations"), "Summary evaluations")
        if summary["evaluations"] != self._max_evals:
            raise ValueError("Run evaluations must equal the manifest max_evals budget")
        self._nonnegative_int(summary.get("generations"), "Summary generations")
        feasible = summary.get("feasible")
        if type(feasible) is not int or feasible != int(summary["best_hcv"] == 0):
            raise ValueError("Summary feasible must be 0/1 and consistent with best_hcv")
        self._finite_number(summary.get("runtime_seconds"), "Summary runtime_seconds")
        if summary["runtime_seconds"] < 0:
            raise ValueError("Summary runtime_seconds must be nonnegative")
        best_chromosome = record.get("best_chromosome")
        if not isinstance(best_chromosome, list) or not best_chromosome:
            raise ValueError("Run best_chromosome must be a nonempty list")
        for gene in best_chromosome:
            if type(gene) is not int or gene < 0:
                raise ValueError("Every gene in best_chromosome must be a nonnegative integer")
        history = record.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError("Run history must be a nonempty list")
        previous_evaluations = 0
        previous_fitness = None
        for row in history:
            if not isinstance(row, dict):
                raise ValueError("History rows must be dictionaries")
            self._matching_identity(row, identity)
            self._validate_metrics(row)
            self._nonnegative_int(row.get("generation"), "History generation")
            self._positive_int(row.get("evaluations"), "History evaluations")
            if not previous_evaluations <= row["evaluations"] <= self._max_evals:
                raise ValueError("History evaluations must be monotone and within budget")
            if previous_fitness is not None and row["best_fitness"] > previous_fitness:
                raise ValueError("History best_fitness must be monotone nonincreasing")
            self._finite_number(row.get("mean_fitness"), "History mean_fitness")
            for optional in ("h_norm", "pc", "pm"):
                if optional in row:
                    self._finite_number(row[optional], f"History {optional}")
            previous_evaluations = row["evaluations"]
            previous_fitness = row["best_fitness"]
        final = history[-1]
        for field in ("evaluations", "best_hcv", "best_scv", "best_fitness"):
            if final[field] != summary[field]:
                raise ValueError(f"Final history {field} does not match summary")
        if final["generation"] != summary["generations"]:
            raise ValueError("Final history generation does not match summary")
        return identity

    @staticmethod
    def _matching_identity(row: dict[str, Any], expected: tuple[str, str, int]) -> None:
        identity = (row.get("instance"), row.get("algorithm"), row.get("seed"))
        if identity != expected or type(row.get("seed")) is not int:
            raise ValueError("Summary/history identity does not match run identity")

    @classmethod
    def _validate_metrics(cls, row: dict[str, Any]) -> None:
        cls._nonnegative_int(row.get("best_hcv"), "best_hcv")
        cls._nonnegative_int(row.get("best_scv"), "best_scv")
        cls._finite_number(row.get("best_fitness"), "best_fitness")
        if row["best_fitness"] != HARD_CONSTRAINT_WEIGHT * row["best_hcv"] + row["best_scv"]:
            raise ValueError(f"best_fitness must equal {HARD_CONSTRAINT_WEIGHT} * best_hcv + best_scv")

    @staticmethod
    def _nonnegative_int(value: Any, label: str) -> None:
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")

    @staticmethod
    def _positive_int(value: Any, label: str) -> None:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{label} must be a positive integer")

    @staticmethod
    def _finite_number(value: Any, label: str) -> None:
        if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
            raise ValueError(f"{label} must be a finite number")

    @classmethod
    def _validate_json(cls, value: Any, active: set[int] | None = None) -> None:
        if value is None or type(value) in (str, bool, int):
            return
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Stored JSON must not contain nonfinite numbers")
            return
        if type(value) not in (dict, list):
            raise ValueError("Stored data must contain only JSON primitives, lists, and dictionaries")
        if active is None:
            active = set()
        identity = id(value)
        if identity in active:
            raise ValueError("Stored JSON must not contain circular references")
        active.add(identity)
        try:
            if isinstance(value, dict):
                for key, item in value.items():
                    if type(key) is not str:
                        raise ValueError("Stored JSON dictionary keys must be strings")
                    cls._validate_json(item, active)
            else:
                for item in value:
                    cls._validate_json(item, active)
        finally:
            active.remove(identity)

    @classmethod
    def _json_bytes(cls, value: Any) -> bytes:
        cls._validate_json(value)
        return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _atomic_write(cls, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as destination:
                temporary = Path(destination.name)
                destination.write(content)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, path)
            temporary = None
            cls._fsync_directory(path.parent)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
