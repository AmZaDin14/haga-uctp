"""
Discrete Particle Swarm Optimization (PSO) for University Course Timetabling (UCTP).
Implements a combinatorial swarm metaheuristic benchmark with strict evaluation budget accounting.
"""

from __future__ import annotations

import time
from typing import Any
import numpy as np

from src.problem import TimetablingProblem


class DiscretePSO:
    """Discrete Particle Swarm Optimization solver for UCTP."""

    problem: TimetablingProblem
    swarm_size: int
    w: float
    c1: float
    c2: float
    max_evaluations: int
    rng: np.random.Generator
    eval_count: int

    def __init__(
        self,
        problem: TimetablingProblem,
        swarm_size: int = 50,
        w: float = 0.20,
        c1: float = 0.40,
        c2: float = 0.40,
        max_evaluations: int = 25000,
        random_seed: int | None = None,
    ):
        self.problem = problem
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_evaluations = max_evaluations
        self.rng = np.random.default_rng(random_seed)
        self.eval_count = 0

    def _eval(self, position: np.ndarray) -> tuple[int, int, int]:
        self.eval_count += 1
        return self.problem.evaluate(position)

    def initialize_swarm(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize swarm particles with random slot assignments."""
        particles = self.rng.integers(
            0, self.problem.total_slots, size=(self.swarm_size, self.problem.total_lectures)
        )
        hcvs = np.zeros(self.swarm_size, dtype=np.int32)
        scvs = np.zeros(self.swarm_size, dtype=np.int32)
        fitnesses = np.zeros(self.swarm_size, dtype=np.int32)

        for i in range(self.swarm_size):
            h, s, f = self._eval(particles[i])
            hcvs[i] = h
            scvs[i] = s
            fitnesses[i] = f

        return particles, hcvs, scvs, fitnesses

    def run(self) -> dict[str, Any]:
        """Execute Discrete PSO optimization until budget B_max is exhausted."""
        start_time = time.time()
        self.eval_count = 0
        particles, hcvs, scvs, fitnesses = self.initialize_swarm()

        # Personal bests (P_i)
        pbest_positions = particles.copy()
        pbest_hcvs = hcvs.copy()
        pbest_scvs = scvs.copy()
        pbest_fitnesses = fitnesses.copy()

        # Global best (G)
        gbest_idx = int(np.argmin(fitnesses))
        gbest_pos: np.ndarray = particles[gbest_idx].copy()
        gbest_hcv: int = int(hcvs[gbest_idx])
        gbest_scv: int = int(scvs[gbest_idx])
        gbest_fitness: int = int(fitnesses[gbest_idx])

        history: list[dict[str, Any]] = [{
            "generation": 0,
            "evaluations": self.eval_count,
            "best_hcv": gbest_hcv,
            "best_scv": gbest_scv,
            "best_fitness": gbest_fitness,
            "mean_fitness": float(np.mean(fitnesses)),
        }]

        iteration = 0
        total_weight = self.w + self.c1 + self.c2
        p_c1 = self.c1 / total_weight
        p_c2 = (self.c1 + self.c2) / total_weight

        while self.eval_count < self.max_evaluations:
            iteration += 1

            for i in range(self.swarm_size):
                if self.eval_count >= self.max_evaluations:
                    break

                new_pos = particles[i].copy()
                for gene_idx in range(self.problem.total_lectures):
                    r = self.rng.random()
                    if r < p_c1:
                        # Inherit from personal best
                        new_pos[gene_idx] = pbest_positions[i, gene_idx]
                    elif r < p_c2:
                        # Inherit from global best
                        new_pos[gene_idx] = gbest_pos[gene_idx]
                    else:
                        # Exploration / Mutation: random valid slot assignment
                        new_pos[gene_idx] = int(self.rng.integers(0, self.problem.total_slots))

                h, s, f = self._eval(new_pos)
                particles[i] = new_pos
                hcvs[i] = h
                scvs[i] = s
                fitnesses[i] = f

                # Update Personal Best
                if f < pbest_fitnesses[i]:
                    pbest_positions[i] = new_pos.copy()
                    pbest_hcvs[i] = h
                    pbest_scvs[i] = s
                    pbest_fitnesses[i] = f

                # Update Global Best
                if f < gbest_fitness:
                    gbest_pos = new_pos.copy()
                    gbest_hcv = int(h)
                    gbest_scv = int(s)
                    gbest_fitness = int(f)

            history.append({
                "generation": iteration,
                "evaluations": self.eval_count,
                "best_hcv": gbest_hcv,
                "best_scv": gbest_scv,
                "best_fitness": gbest_fitness,
                "mean_fitness": float(np.mean(fitnesses)),
            })

        elapsed = time.time() - start_time

        return {
            "algorithm": "Discrete_PSO",
            "best_chromosome": gbest_pos,
            "best_hcv": int(gbest_hcv),
            "best_scv": int(gbest_scv),
            "best_fitness": int(gbest_fitness),
            "total_evaluations": self.eval_count,
            "generations": iteration,
            "history": history,
            "runtime_seconds": elapsed,
        }
