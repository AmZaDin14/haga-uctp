"""
Simulated Annealing (SA) for University Course Timetabling (UCTP).
Implements a single-solution trajectory metaheuristic benchmark with strict evaluation budget accounting.
"""

from __future__ import annotations

import math
import time
from typing import Any
import numpy as np

from src.problem import TimetablingProblem


class SimulatedAnnealing:
    """Simulated Annealing solver for UCTP."""

    problem: TimetablingProblem
    t_init: float
    t_min: float
    cooling_rate: float
    steps_per_temp: int
    max_evaluations: int
    rng: np.random.Generator
    eval_count: int

    def __init__(
        self,
        problem: TimetablingProblem,
        t_init: float = 100.0,
        t_min: float = 0.01,
        cooling_rate: float = 0.990,
        steps_per_temp: int = 50,
        max_evaluations: int = 25000,
        random_seed: int | None = None,
    ):
        self.problem = problem
        self.t_init = t_init
        self.t_min = t_min
        self.cooling_rate = cooling_rate
        self.steps_per_temp = steps_per_temp
        self.max_evaluations = max_evaluations
        self.rng = np.random.default_rng(random_seed)
        self.eval_count = 0

    def _eval(self, chromosome: np.ndarray) -> tuple[int, int, int]:
        self.eval_count += 1
        return self.problem.evaluate(chromosome)

    def generate_neighbor(
        self,
        current: np.ndarray,
        conflicted: list[int] | None = None,
    ) -> np.ndarray:
        """Perturb the solution, preferring lectures involved in hard conflicts.

        Targeting the violated lectures turns the random walk into a directed
        repair process: with 160-434 lectures, a uniformly random move lands on
        a conflicted lecture only a few percent of the time.
        """
        neighbor = current.copy()
        if conflicted and self.rng.random() < 0.80:
            target = int(self.rng.choice(conflicted))
            if self.rng.random() < 0.5:
                other = int(self.rng.integers(0, self.problem.total_lectures))
                neighbor[target], neighbor[other] = neighbor[other], neighbor[target]
            else:
                neighbor[target] = int(self.rng.integers(0, self.problem.total_slots))
            return neighbor

        if self.rng.random() < 0.70:
            # Reassign single lecture to a random slot
            gene_idx = int(self.rng.integers(0, self.problem.total_lectures))
            neighbor[gene_idx] = int(self.rng.integers(0, self.problem.total_slots))
        else:
            # Swap two lecture assignments
            pair = [int(x) for x in self.rng.choice(self.problem.total_lectures, size=2, replace=False).tolist()]
            idx1, idx2 = pair[0], pair[1]
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        return neighbor

    def run(self) -> dict[str, Any]:
        """Execute Simulated Annealing until evaluation quota B_max is exhausted."""
        start_time = time.time()
        self.eval_count = 0

        # Initialize random solution
        current_state = self.rng.integers(
            0, self.problem.total_slots, size=self.problem.total_lectures
        )
        c_hcv, c_scv, c_fit = self._eval(current_state)

        best_state: np.ndarray = current_state.copy()
        best_hcv: int = int(c_hcv)
        best_scv: int = int(c_scv)
        best_fitness: int = int(c_fit)

        history: list[dict[str, Any]] = [{
            "generation": 0,
            "evaluations": self.eval_count,
            "best_hcv": best_hcv,
            "best_scv": best_scv,
            "best_fitness": best_fitness,
            "mean_fitness": float(c_fit),
            "temperature": float(self.t_init),
        }]

        t = self.t_init
        iteration = 0
        conflicted: list[int] = []

        while self.eval_count < self.max_evaluations:
            iteration += 1

            for _ in range(self.steps_per_temp):
                if self.eval_count >= self.max_evaluations:
                    break

                if not conflicted or self.eval_count % 50 == 0:
                    conflicted = self.problem.get_conflicted_lectures(current_state)

                candidate = self.generate_neighbor(current_state, conflicted)
                h_new, s_new, f_new = self._eval(candidate)
                delta = f_new - c_fit

                accept = False
                if delta <= 0:
                    accept = True
                else:
                    p_accept = math.exp(-min(delta / max(t, 1e-8), 50.0))
                    if self.rng.random() < p_accept:
                        accept = True

                if accept:
                    current_state = candidate
                    c_hcv = h_new
                    c_scv = s_new
                    c_fit = f_new

                    if c_fit < best_fitness:
                        best_state = current_state.copy()
                        best_hcv = int(c_hcv)
                        best_scv = int(c_scv)
                        best_fitness = int(c_fit)

            # Cool down
            t = max(self.t_min, t * self.cooling_rate)

            history.append({
                "generation": iteration,
                "evaluations": self.eval_count,
                "best_hcv": best_hcv,
                "best_scv": best_scv,
                "best_fitness": best_fitness,
                "mean_fitness": float(c_fit),
                "temperature": float(t),
            })

        elapsed = time.time() - start_time

        return {
            "algorithm": "Simulated_Annealing",
            "best_chromosome": best_state,
            "best_hcv": int(best_hcv),
            "best_scv": int(best_scv),
            "best_fitness": int(best_fitness),
            "total_evaluations": self.eval_count,
            "generations": iteration,
            "history": history,
            "runtime_seconds": elapsed,
        }
