"""Run the pilot subset or complete ITC suite through the shared experiment runner."""
from pathlib import Path
import sys

# Support both `python scripts/run_benchmark_suite.py` and `python -m scripts.run_benchmark_suite`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runner import main


if __name__ == "__main__":
    raise SystemExit(main(suite=True))
