"""
Canonical Genetic Algorithm for UCTP with Strict Evaluation Budget Tracking.
Uses static parameters: Pc = 0.85, Pm = 0.05, Tournament Selection (k=3), Elitism (e=2).
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.problem import TimetablingProblem


class CanonicalGA:
    def __init__(
        self,
        problem: TimetablingProblem,
        pop_size: int = 50,
        pc: float = 0.85,
        pm: float = 0.05,
        tournament_k: int = 3,
        elitism_k: int = 2,
        max_evaluations: int = 25000,
        random_seed: Optional[int] = None,
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.tournament_k = tournament_k
        self.elitism_k = elitism_k
        self.max_evaluations = max_evaluations
        self.rng = np.random.default_rng(random_seed)

        self.eval_count = 0

    def _eval(self, chromosome: np.ndarray) -> Tuple[int, int, int]:
        self.eval_count += 1
        return self.problem.evaluate(chromosome)

    def initialize_population(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize random population and evaluate each individual."""
        pop = self.rng.integers(0, self.problem.total_slots, size=(self.pop_size, self.problem.total_lectures))
        hcvs = np.zeros(self.pop_size, dtype=np.int32)
        scvs = np.zeros(self.pop_size, dtype=np.int32)
        fitnesses = np.zeros(self.pop_size, dtype=np.int32)

        for i in range(self.pop_size):
            h, s, f = self._eval(pop[i])
            hcvs[i], scvs[i], fitnesses[i] = h, s, f

        return pop, hcvs, scvs, fitnesses

    def tournament_selection(self, fitnesses: np.ndarray) -> int:
        """Tournament selection: lowest fitness wins."""
        candidates = self.rng.choice(self.pop_size, size=self.tournament_k, replace=False)
        best_cand = candidates[np.argmin(fitnesses[candidates])]
        return best_cand

    def two_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Two-point crossover on 1D chromosome."""
        if self.rng.random() > self.pc:
            return parent1.copy(), parent2.copy()

        length = len(parent1)
        pt1, pt2 = sorted(self.rng.choice(length, size=2, replace=False))
        child1 = parent1.copy()
        child2 = parent2.copy()
        child1[pt1:pt2] = parent2[pt1:pt2]
        child2[pt1:pt2] = parent1[pt1:pt2]
        return child1, child2

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """Random slot reassignment mutation."""
        mutated = chromosome.copy()
        for gene_idx in range(len(mutated)):
            if self.rng.random() < self.pm:
                mutated[gene_idx] = self.rng.integers(0, self.problem.total_slots)
        return mutated

    def run(self) -> Dict:
        start_time = time.time()
        self.eval_count = 0

        pop, hcvs, scvs, fitnesses = self.initialize_population()

        best_idx = np.argmin(fitnesses)
        best_hcv = hcvs[best_idx]
        best_scv = scvs[best_idx]
        best_fitness = fitnesses[best_idx]
        best_chrom = pop[best_idx].copy()

        history = [{
            "generation": 0,
            "evaluations": self.eval_count,
            "best_hcv": int(best_hcv),
            "best_scv": int(best_scv),
            "best_fitness": int(best_fitness),
            "mean_fitness": float(np.mean(fitnesses)),
        }]

        generation = 0
        while self.eval_count < self.max_evaluations:
            generation += 1

            # Elitism: preserve top e individuals
            sorted_indices = np.argsort(fitnesses)
            new_pop = [pop[sorted_indices[i]].copy() for i in range(self.elitism_k)]
            new_hcvs = [hcvs[sorted_indices[i]] for i in range(self.elitism_k)]
            new_scvs = [scvs[sorted_indices[i]] for i in range(self.elitism_k)]
            new_fitnesses = [fitnesses[sorted_indices[i]] for i in range(self.elitism_k)]

            # Generate rest of the population
            while len(new_pop) < self.pop_size:
                if self.eval_count >= self.max_evaluations:
                    break

                p1_idx = self.tournament_selection(fitnesses)
                p2_idx = self.tournament_selection(fitnesses)

                c1, c2 = self.two_point_crossover(pop[p1_idx], pop[p2_idx])
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)

                # Evaluate c1
                if self.eval_count < self.max_evaluations:
                    h1, s1, f1 = self._eval(c1)
                    new_pop.append(c1)
                    new_hcvs.append(h1)
                    new_scvs.append(s1)
                    new_fitnesses.append(f1)

                # Evaluate c2
                if len(new_pop) < self.pop_size and self.eval_count < self.max_evaluations:
                    h2, s2, f2 = self._eval(c2)
                    new_pop.append(c2)
                    new_hcvs.append(h2)
                    new_scvs.append(s2)
                    new_fitnesses.append(f2)

            pop = np.array(new_pop)
            hcvs = np.array(new_hcvs)
            scvs = np.array(new_scvs)
            fitnesses = np.array(new_fitnesses)

            # Update best
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_hcv = hcvs[gen_best_idx]
                best_scv = scvs[gen_best_idx]
                best_chrom = pop[gen_best_idx].copy()

            history.append({
                "generation": generation,
                "evaluations": self.eval_count,
                "best_hcv": int(best_hcv),
                "best_scv": int(best_scv),
                "best_fitness": int(best_fitness),
                "mean_fitness": float(np.mean(fitnesses)),
            })

        elapsed = time.time() - start_time

        return {
            "algorithm": "Canonical_GA",
            "best_chromosome": best_chrom,
            "best_hcv": int(best_hcv),
            "best_scv": int(best_scv),
            "best_fitness": int(best_fitness),
            "total_evaluations": self.eval_count,
            "generations": generation,
            "history": history,
            "runtime_seconds": elapsed,
        }
