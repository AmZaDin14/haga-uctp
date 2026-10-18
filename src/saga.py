# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUnannotatedClassAttribute=false
"""Self-Adaptive Genetic Algorithm (SAGA Reconstructed) for UCTP.

Reconstruction of Pamungkas & Neta (2025) synthesized with Potter & De Jong (1994)
cooperative coevolution and Herrera & Lozano (2003) Mamdani fuzzy parameter control.

Key Mechanisms:
1. Cooperative Coevolution Engine: Decomposes decision space into two interacting
   sub-populations: Timeslots (T) and Rooms (R). Evaluated via representative collaboration.
2. Mamdani Fuzzy Inference System (FIS): Dynamically adapts crossover probability (Pc)
   and mutation probability (Pm) based on Diversity Index, Stagnation Counter, and Improvement Rate.
3. Local Search Module: First-improvement hill-climbing search applied to elite solution.
4. Strict Evaluation Budget Parity: All evaluations (coevolution + local search) consume
   from the shared quota B_max.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.fuzzy_controller import FuzzyParameterController
from src.problem import TimetablingProblem


class SAGA_Reconstructed:
    """Self-Adaptive Genetic Algorithm with Cooperative Coevolution and Fuzzy Control."""

    def __init__(
        self,
        problem: TimetablingProblem,
        pop_size: int = 50,
        max_evaluations: int = 25000,
        tournament_k: int = 3,
        els_max_steps: int = 50,
        max_stagnation: int = 15,
        random_seed: int | None = None,
    ) -> None:
        self.problem = problem
        self.pop_size = pop_size
        self.max_evaluations = max_evaluations
        self.tournament_k = tournament_k
        self.els_max_steps = els_max_steps
        self.max_stagnation = max_stagnation

        self.rng = np.random.default_rng(random_seed)
        self.eval_count = 0

        self.num_timeslots = problem.total_timeslots
        self.num_rooms = problem.num_rooms
        self.total_lectures = problem.total_lectures

        # Initialize Mamdani FIS for parameter control
        self.fis = FuzzyParameterController(
            pc_min=0.40,
            pc_max=0.95,
            pm_min=0.10,
            pm_max=0.90,
        )

    def _eval(self, chromosome: np.ndarray) -> tuple[int, int, int]:
        """Evaluate chromosome while strictly accounting for evaluation budget."""
        self.eval_count += 1
        hcv, scv, fitness = self.problem.evaluate(chromosome)
        return int(hcv), int(scv), int(fitness)

    def _combine(self, t_vec: np.ndarray, r_vec: np.ndarray) -> np.ndarray:
        """Combine timeslot vector and room vector into standard 1D slot chromosome."""
        return (t_vec.astype(np.int64) * self.num_rooms + r_vec.astype(np.int64)).astype(np.int64)

    def calculate_diversity(self, pop: np.ndarray) -> float:
        """Calculate normalized pairwise Hamming diversity of population."""
        n = len(pop)
        if n < 2:
            return 0.0
        sample_size = min(n, 20)
        indices = self.rng.choice(n, size=sample_size, replace=False)
        sub_pop = pop[indices]

        diff_count = 0
        total_pairs = 0
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                diff_count += int(np.sum(sub_pop[i] != sub_pop[j]))
                total_pairs += 1

        if total_pairs == 0 or self.total_lectures == 0:
            return 0.0
        return float(diff_count / (total_pairs * self.total_lectures))

    def tournament_select(self, fitnesses: np.ndarray) -> int:
        """Select individual index using tournament selection (lower fitness is better)."""
        candidates = self.rng.choice(len(fitnesses), size=self.tournament_k, replace=False)
        best_candidate = candidates[0]
        best_fit = fitnesses[best_candidate]
        for c in candidates[1:]:
            if fitnesses[c] < best_fit:
                best_candidate = c
                best_fit = fitnesses[c]
        return int(best_candidate)

    def two_point_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray, pc: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply two-point crossover with probability pc."""
        if self.rng.random() > pc:
            return parent1.copy(), parent2.copy()

        length = len(parent1)
        if length < 3:
            return parent1.copy(), parent2.copy()

        pts = sorted(self.rng.choice(range(1, length), size=2, replace=False))
        p1, p2 = pts[0], pts[1]

        c1 = np.empty_like(parent1)
        c2 = np.empty_like(parent2)

        c1[:p1] = parent1[:p1]
        c1[p1:p2] = parent2[p1:p2]
        c1[p2:] = parent1[p2:]

        c2[:p1] = parent2[:p1]
        c2[p1:p2] = parent1[p1:p2]
        c2[p2:] = parent2[p2:]

        return c1, c2

    def mutate_timeslots(self, chromosome: np.ndarray, pm: float) -> np.ndarray:
        """Mutate timeslot alleles with gene probability pm / total_lectures."""
        mutant = chromosome.copy()
        gene_pm = min(1.0, pm / max(1, self.total_lectures) * 5.0)
        mask = self.rng.random(size=self.total_lectures) < gene_pm
        if np.any(mask):
            mutant[mask] = self.rng.integers(0, self.num_timeslots, size=np.sum(mask))
        return mutant

    def mutate_rooms(self, chromosome: np.ndarray, pm: float) -> np.ndarray:
        """Mutate room alleles with gene probability pm / total_lectures."""
        mutant = chromosome.copy()
        gene_pm = min(1.0, pm / max(1, self.total_lectures) * 5.0)
        mask = self.rng.random(size=self.total_lectures) < gene_pm
        if np.any(mask):
            mutant[mask] = self.rng.integers(0, self.num_rooms, size=np.sum(mask))
        return mutant

    def local_search(
        self,
        best_chrom: np.ndarray,
        curr_hcv: int,
        curr_scv: int,
        curr_fit: int,
    ) -> tuple[np.ndarray, int, int, int]:
        """First-improvement hill climbing on conflicted lectures under budget."""
        refined_chrom = best_chrom.copy()
        b_hcv, b_scv, b_fit = curr_hcv, curr_scv, curr_fit

        for _ in range(self.els_max_steps):
            if self.eval_count >= self.max_evaluations:
                break

            conflicted = self.problem.get_conflicted_lectures(refined_chrom)
            if conflicted:
                target_lec = int(self.rng.choice(conflicted))
            else:
                target_lec = int(self.rng.integers(0, self.total_lectures))

            candidate = refined_chrom.copy()
            candidate[target_lec] = self.rng.integers(0, self.problem.total_slots)

            cand_hcv, cand_scv, cand_fit = self._eval(candidate)
            if cand_fit < b_fit:
                refined_chrom = candidate
                b_hcv, b_scv, b_fit = cand_hcv, cand_scv, cand_fit

        return refined_chrom, b_hcv, b_scv, b_fit

    def run(self) -> dict[str, Any]:
        """Execute SAGA Reconstructed under strict evaluation quota B_max."""
        start_time = time.time()
        self.eval_count = 0

        # 1. Initialize Sub-Populations (Timeslots and Rooms)
        pop_t = self.rng.integers(0, self.num_timeslots, size=(self.pop_size, self.total_lectures))
        pop_r = self.rng.integers(0, self.num_rooms, size=(self.pop_size, self.total_lectures))

        # Initial collaborative evaluation
        fits_t = np.zeros(self.pop_size, dtype=np.int64)
        fits_r = np.zeros(self.pop_size, dtype=np.int64)

        best_t = pop_t[0].copy()
        best_r = pop_r[0].copy()
        overall_best_chrom = self._combine(best_t, best_r)
        best_hcv, best_scv, best_fitness = self._eval(overall_best_chrom)

        # Initial evaluate of sub_pop T paired with default best_r
        for i in range(self.pop_size):
            if self.eval_count >= self.max_evaluations:
                break
            combined = self._combine(pop_t[i], best_r)
            h, s, f = self._eval(combined)
            fits_t[i] = f
            if f < best_fitness:
                best_fitness = f
                best_hcv = h
                best_scv = s
                best_t = pop_t[i].copy()
                overall_best_chrom = combined.copy()

        # Initial evaluate of sub_pop R paired with best_t
        for j in range(self.pop_size):
            if self.eval_count >= self.max_evaluations:
                break
            combined = self._combine(best_t, pop_r[j])
            h, s, f = self._eval(combined)
            fits_r[j] = f
            if f < best_fitness:
                best_fitness = f
                best_hcv = h
                best_scv = s
                best_r = pop_r[j].copy()
                overall_best_chrom = combined.copy()

        stagnation_counter = 0
        generation = 0
        prev_best_fitness = best_fitness
        history: list[dict[str, Any]] = []

        # Record generation 0
        initial_div = self.calculate_diversity(pop_t)
        pc, pm = self.fis.evaluate(initial_div, stagnation_counter, self.max_stagnation, 0.0)
        history.append({
            "generation": 0,
            "evaluations": self.eval_count,
            "best_hcv": int(best_hcv),
            "best_scv": int(best_scv),
            "best_fitness": int(best_fitness),
            "mean_fitness": float(np.mean(fits_t)),
            "pc": float(pc),
            "pm": float(pm),
        })

        # Main Coevolutionary Loop
        while self.eval_count < self.max_evaluations:
            generation += 1

            # 1. Fuzzy Parameter Adaptation
            div_t = self.calculate_diversity(pop_t)
            div_r = self.calculate_diversity(pop_r)
            avg_diversity = 0.5 * (div_t + div_r)

            improvement_rate = (
                max(0.0, float(prev_best_fitness - best_fitness) / float(max(1, prev_best_fitness)))
            )
            pc, pm = self.fis.evaluate(
                avg_diversity,
                stagnation_counter,
                self.max_stagnation,
                improvement_rate,
            )

            # 2. Evolve Timeslot Sub-Population
            new_pop_t: list[np.ndarray] = [best_t.copy()]  # 1-individual elitism
            new_fits_t: list[int] = [int(fits_t[np.argmin(fits_t)])]

            while len(new_pop_t) < self.pop_size and self.eval_count < self.max_evaluations:
                p1_idx = self.tournament_select(fits_t)
                p2_idx = self.tournament_select(fits_t)
                c1, c2 = self.two_point_crossover(pop_t[p1_idx], pop_t[p2_idx], pc)
                c1 = self.mutate_timeslots(c1, pm)
                c2 = self.mutate_timeslots(c2, pm)

                if self.eval_count < self.max_evaluations:
                    comb1 = self._combine(c1, best_r)
                    h1, s1, f1 = self._eval(comb1)
                    new_pop_t.append(c1)
                    new_fits_t.append(f1)
                    if f1 < best_fitness:
                        best_fitness = f1
                        best_hcv = h1
                        best_scv = s1
                        best_t = c1.copy()
                        overall_best_chrom = comb1.copy()

                if len(new_pop_t) < self.pop_size and self.eval_count < self.max_evaluations:
                    comb2 = self._combine(c2, best_r)
                    h2, s2, f2 = self._eval(comb2)
                    new_pop_t.append(c2)
                    new_fits_t.append(f2)
                    if f2 < best_fitness:
                        best_fitness = f2
                        best_hcv = h2
                        best_scv = s2
                        best_t = c2.copy()
                        overall_best_chrom = comb2.copy()

            if len(new_pop_t) == self.pop_size:
                pop_t = np.array(new_pop_t)
                fits_t = np.array(new_fits_t)

            # 3. Evolve Room Sub-Population
            new_pop_r: list[np.ndarray] = [best_r.copy()]  # 1-individual elitism
            new_fits_r: list[int] = [int(fits_r[np.argmin(fits_r)])]

            while len(new_pop_r) < self.pop_size and self.eval_count < self.max_evaluations:
                p1_idx = self.tournament_select(fits_r)
                p2_idx = self.tournament_select(fits_r)
                c1, c2 = self.two_point_crossover(pop_r[p1_idx], pop_r[p2_idx], pc)
                c1 = self.mutate_rooms(c1, pm)
                c2 = self.mutate_rooms(c2, pm)

                if self.eval_count < self.max_evaluations:
                    comb1 = self._combine(best_t, c1)
                    h1, s1, f1 = self._eval(comb1)
                    new_pop_r.append(c1)
                    new_fits_r.append(f1)
                    if f1 < best_fitness:
                        best_fitness = f1
                        best_hcv = h1
                        best_scv = s1
                        best_r = c1.copy()
                        overall_best_chrom = comb1.copy()

                if len(new_pop_r) < self.pop_size and self.eval_count < self.max_evaluations:
                    comb2 = self._combine(best_t, c2)
                    h2, s2, f2 = self._eval(comb2)
                    new_pop_r.append(c2)
                    new_fits_r.append(f2)
                    if f2 < best_fitness:
                        best_fitness = f2
                        best_hcv = h2
                        best_scv = s2
                        best_r = c2.copy()
                        overall_best_chrom = comb2.copy()

            if len(new_pop_r) == self.pop_size:
                pop_r = np.array(new_pop_r)
                fits_r = np.array(new_fits_r)

            # 4. Local Search on Overall Best
            if self.eval_count < self.max_evaluations:
                ref_chrom, ref_h, ref_s, ref_f = self.local_search(
                    overall_best_chrom, best_hcv, best_scv, best_fitness
                )
                if ref_f < best_fitness:
                    best_fitness = ref_f
                    best_hcv = ref_h
                    best_scv = ref_s
                    overall_best_chrom = ref_chrom.copy()
                    # Lamarckian update to sub-populations
                    best_t = (ref_chrom // self.num_rooms).astype(np.int64)
                    best_r = (ref_chrom % self.num_rooms).astype(np.int64)

            # Track Stagnation
            if best_fitness < prev_best_fitness:
                stagnation_counter = 0
                prev_best_fitness = best_fitness
            else:
                stagnation_counter += 1

            history.append({
                "generation": generation,
                "evaluations": self.eval_count,
                "best_hcv": int(best_hcv),
                "best_scv": int(best_scv),
                "best_fitness": int(best_fitness),
                "mean_fitness": float(0.5 * (np.mean(fits_t) + np.mean(fits_r))),
                "pc": float(pc),
                "pm": float(pm),
            })

        elapsed = time.time() - start_time

        return {
            "algorithm": "SAGA_Reconstructed",
            "best_chromosome": overall_best_chrom.copy(),
            "best_hcv": int(best_hcv),
            "best_scv": int(best_scv),
            "best_fitness": int(best_fitness),
            "total_evaluations": self.eval_count,
            "generations": generation,
            "history": history,
            "runtime_seconds": elapsed,
        }
