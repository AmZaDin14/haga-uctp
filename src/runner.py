"""Budget-matched experiments with atomic per-run persistence and safe resume."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
from importlib.metadata import version
import inspect
import logging
import multiprocessing
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
from typing import Any

import pandas as pd

from src.benchmark_analysis import aggregate_results, export_instance
from src.canonical_ga import CanonicalGA
from src.experiment_store import ExperimentStore
from src.haga import HAGA
from src.loader import parse_ctt
from src.problem import TimetablingProblem
from src.pso import DiscretePSO
from src.saga import SAGA_Reconstructed
from src.sa import SimulatedAnnealing

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = {
    "Canonical_GA": CanonicalGA,
    "Discrete_PSO": DiscretePSO,
    "Simulated_Annealing": SimulatedAnnealing,
    "HAGA": HAGA,
    "GA_APC": HAGA,
    "GA_ELS": HAGA,
    "SAGA_Reconstructed": SAGA_Reconstructed,
}
MODES = {
    "preliminary": ["Canonical_GA", "HAGA"],
    "comparative": ["Canonical_GA", "Discrete_PSO", "Simulated_Annealing", "HAGA"],
    "ablation": ["Canonical_GA", "GA_APC", "GA_ELS", "HAGA"],
    "saga_comparison": ["Canonical_GA", "HAGA", "SAGA_Reconstructed"],
    "full_comparison": ["Canonical_GA", "Discrete_PSO", "Simulated_Annealing", "HAGA", "SAGA_Reconstructed"],
}
SUMMARY_COLUMNS = [
    "instance", "algorithm", "seed", "best_hcv", "best_scv", "best_fitness",
    "feasible", "evaluations", "generations", "runtime_seconds",
]


def _parameters(name: str, pop_size: int, budget: int) -> dict[str, Any]:
    """Resolve actual constructor defaults; these same values reach the worker."""
    parameters = {
        key: value.default
        for key, value in inspect.signature(ALGORITHMS[name]).parameters.items()
        if key not in {"problem", "random_seed"}
    }
    parameters["max_evaluations"] = budget
    if "pop_size" in parameters:
        parameters["pop_size"] = pop_size
    if "swarm_size" in parameters:
        parameters["swarm_size"] = pop_size
    if name in {"HAGA", "GA_APC", "GA_ELS"}:
        parameters["enable_apc"] = name != "GA_ELS"
        parameters["enable_els"] = name != "GA_APC"
    return parameters


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(args: argparse.Namespace, paths: list[Path]) -> dict[str, Any]:
    sources = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    sources.append(ROOT / "requirements.txt")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    cpu_info = Path("/proc/cpuinfo")
    cpu_model = platform.processor()
    if cpu_info.exists():
        cpu_model = next(
            (line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
             if line.startswith("model name")), cpu_model,
        )
    return {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "stage": args.stage,
        "config": {
            "instances": [
                {"id": path.stem, "path": str(path), "sha256": _sha256(path)}
                for path in paths
            ],
            "algorithms": {
                name: _parameters(name, args.pop, args.evals) for name in MODES[args.mode]
            },
            "seeds": list(range(101, 101 + args.seeds)),
            "max_evals": args.evals,
            "pop_size": args.pop,
        },
        "provenance": {
            "git_revision": revision,
            "source_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in sources},
            "dependencies": {name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib")},
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "cpu_model": cpu_model,
            "cpu_count": multiprocessing.cpu_count(),
        },
    }


def _worker_init() -> None:
    # The parent handles Ctrl-C and drains in-flight runs before releasing its lock.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_one(instance: str, path: str, algorithm: str, seed: int,
             parameters: dict[str, Any]) -> dict[str, Any]:
    problem = TimetablingProblem(parse_ctt(path))
    result = ALGORITHMS[algorithm](problem, **parameters, random_seed=seed).run()
    if result["algorithm"] != algorithm:
        raise ValueError(f"Unexpected algorithm identity: {result['algorithm']}")
    identity = {"instance": instance, "algorithm": algorithm, "seed": seed}
    summary = {
        **identity,
        **{key: int(result[key]) for key in ("best_hcv", "best_scv", "best_fitness", "generations")},
        "feasible": int(result["best_hcv"] == 0),
        "evaluations": int(result["total_evaluations"]),
        "runtime_seconds": float(result["runtime_seconds"]),
    }
    best_chromosome = [int(x) for x in result["best_chromosome"]]
    history = [{**row, **identity} for row in result["history"]]
    return {
        "schema_version": 1,
        **identity,
        "summary": summary,
        "best_chromosome": best_chromosome,
        "history": history,
    }


def run_experiment(manifest: dict[str, Any], output_dir: Path, workers: int,
                   *, analyze_only: bool = False) -> int:
    """Run missing keys only. SIGINT/SIGTERM drain active runs and exit 130."""
    config = manifest["config"]
    instance_ids = [item["id"] for item in config["instances"]]
    algorithms = list(config["algorithms"])
    total = len(instance_ids) * len(algorithms) * len(config["seeds"])
    completed = failed = reused = 0
    stopping = False
    summaries: list[dict[str, Any]] = []
    logger = logging.getLogger("haga.benchmark")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    with ExperimentStore(output_dir, manifest) as store:
        file_handler = logging.FileHandler(output_dir / "execution.log")
        console_handler = logging.StreamHandler(sys.stdout)
        for handler in (file_handler, console_handler):
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            logger.addHandler(handler)
        previous_signals = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            logger.info("START stage=%s workers=%d expected=%d analyze_only=%s",
                        manifest["stage"], workers, total, analyze_only)
            # Spawn avoids inheriting the parent's flock and signal handlers.
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_worker_init) as executor:
                for item in config["instances"]:
                    if stopping:
                        break
                    instance = item["id"]
                    records = []
                    missing: deque[tuple[str, int]] = deque()
                    for algorithm in algorithms:
                        for seed in config["seeds"]:
                            cached = store.load(instance, algorithm, seed)
                            if cached is None:
                                missing.append((algorithm, seed))
                            else:
                                records.append(cached)
                                completed += 1
                                reused += 1
                    logger.info("INSTANCE %s cached=%d pending=%d", instance, len(records), len(missing))
                    pending = {}
                    stop_logged = False
                    while pending or (missing and not stopping and not analyze_only):
                        while missing and len(pending) < workers and not stopping:
                            algorithm, seed = missing.popleft()
                            future = executor.submit(
                                _run_one, instance, item["path"], algorithm, seed,
                                config["algorithms"][algorithm],
                            )
                            pending[future] = (algorithm, seed)
                        if stopping and not stop_logged:
                            logger.info("STOP requested; draining %d active runs, no new submissions", len(pending))
                            stop_logged = True
                        if not pending:
                            break
                        done, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                        for future in done:
                            algorithm, seed = pending.pop(future)
                            try:
                                record = future.result()
                                store.save(record)
                            except Exception as error:
                                store.fail(instance, algorithm, seed, f"{type(error).__name__}: {error}")
                                failed += 1
                                logger.error("FAILED %s/%s/%d: %s", instance, algorithm, seed, error)
                            else:
                                records.append(record)
                                completed += 1
                                logger.info("DONE %s/%s/%d completed=%d failed=%d remaining=%d",
                                            instance, algorithm, seed, completed, failed, total - completed)
                    if records:
                        frame = pd.DataFrame([record["summary"] for record in records], columns=SUMMARY_COLUMNS)
                        history = pd.DataFrame([row for record in records for row in record["history"]])
                        export_instance(frame, history, output_dir / instance)
                        summaries.extend([record["summary"] for record in records])
                # Include cached results of not-yet-visited instances after interruption.
                visited = {summary["instance"] for summary in summaries}
                for instance in instance_ids:
                    if instance not in visited:
                        for algorithm in algorithms:
                            for seed in config["seeds"]:
                                record = store.load(instance, algorithm, seed)
                                if record is not None:
                                    summaries.append(record["summary"])
                frame = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
                inferential = manifest["stage"] == "main" and not stopping and failed == 0 and len(frame) == total
                aggregate_results(frame, instance_ids, algorithms, config["seeds"], output_dir,
                                  inferential=inferential)
            logger.info("END completed_this_invocation=%d reused=%d failed=%d stored=%d/%d interrupted=%s",
                        completed - reused, reused, failed, len(summaries), total, stopping)
            return 130 if stopping else (1 if failed or len(summaries) != total else 0)
        finally:
            for sig, handler in previous_signals.items():
                signal.signal(sig, handler)
            for handler in (file_handler, console_handler):
                logger.removeHandler(handler)
                handler.close()


def main(argv: list[str] | None = None, *, suite: bool = False) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True, help="Immutable experiment identity; reuse to resume")
    parser.add_argument("--stage", choices=["smoke", "pilot", "main"], default="smoke")
    parser.add_argument("--mode", choices=list(MODES), default="comparative")
    instances = parser.add_mutually_exclusive_group()
    instances.add_argument("--instance", help="Single CTT path")
    instances.add_argument("--instances", nargs="+", help="CTT paths, in execution order")
    parser.add_argument("--evals", type=int, default=50000)
    parser.add_argument("--pop", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds starting at 101")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--analyze-only", action="store_true",
                        help="Rebuild reports from stored runs without executing missing runs")
    args = parser.parse_args(argv)
    if not args.experiment_id or args.experiment_id in {".", ".."} or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in args.experiment_id
    ):
        parser.error("--experiment-id must contain only ASCII letters, digits, underscores or hyphens")
    if args.pop < 3 or args.evals < args.pop or args.seeds < 1 or args.workers < 1:
        parser.error("Require pop >= 3, evals >= pop, seeds >= 1 and workers >= 1")
    default_ids = ([f"comp{i:02d}" for i in range(1, 22)] if args.stage == "main"
                   else ["comp01", "comp07", "comp12"] if suite else ["comp01"])
    requested = args.instances or ([args.instance] if args.instance else
                                   [str(ROOT / "data" / "itc2007" / f"{name}.ctt") for name in default_ids])
    paths = [Path(path).resolve() for path in requested]
    if len({path.stem for path in paths}) != len(paths):
        parser.error("Instance ids (file stems) must be unique")
    if any(not path.is_file() for path in paths):
        parser.error("Every instance must be an existing CTT file")
    if args.stage == "main":
        if {path.stem for path in paths} != {f"comp{i:02d}" for i in range(1, 22)}:
            parser.error("Main benchmark requires all comp01-comp21; use smoke/pilot for subsets")
        if args.seeds != 30:
            parser.error("Main benchmark requires --seeds 30")
    try:
        manifest = _manifest(args, paths)
        return run_experiment(manifest, args.output_root.resolve() / args.experiment_id, args.workers,
                              analyze_only=args.analyze_only)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Experiment rejected: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
