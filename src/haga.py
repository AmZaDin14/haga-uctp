"""
Hybrid Adaptive Genetic Algorithm (HAGA) for UCTP.
Features:
1. Normalized Pairwise Hamming Distance diversity metric.
2. Diversity-Guided Adaptive Parameter Controller (APC) with linear clamping.
3. Generational Stochastic First-Improvement Elite Local Search (ELS).
4. Strict evaluation budget tracking parity.
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.problem import TimetablingProblem


class HAGA:
    def __init__(
        self,
        problem: TimetablingProblem,
        pop_size: int = 50,
        pc_base: float = 0.80,
        pm_base: float = 0.05,
        pc_min: float = 0.40,
        pc_max: float = 0.95,
        pm_min: float = 0.01,
        pm_max: float = 0.10,
        tournament_k: int = 3,
        elitism_k: int = 2,
        els_elite_ratio: float = 0.10,
        els_max_steps: int = 50,
        els_final_share: float = 0.30,
        els_target: str = "violations",
        label: str = "",
        max_evaluations: int = 25000,
        enable_apc: bool = True,
        enable_els: bool = True,
        random_seed: Optional[int] = None,
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.pc_base = pc_base
        self.pm_base = pm_base
        self.pc_min = pc_min
        self.pc_max = pc_max
        self.pm_min = pm_min
        self.pm_max = pm_max
        self.tournament_k = tournament_k
        self.elitism_k = elitism_k
        self.els_elite_count = max(2, int(els_elite_ratio * pop_size))
        self.els_max_steps = els_max_steps
        self.els_final_share = float(np.clip(els_final_share, 0.0, 0.9))
        if els_target not in {"violations", "random"}:
            raise ValueError("els_target must be 'violations' or 'random'")
        self.els_target = els_target
        self.label = label
        self.max_evaluations = max_evaluations
        self.enable_apc = enable_apc
        self.enable_els = enable_els
        self.rng = np.random.default_rng(random_seed)
        self.eval_count = 0

    def _eval(self, chromosome: np.ndarray) -> Tuple[int, int, int]:
        self.eval_count += 1
        return self.problem.evaluate(chromosome)

    def calculate_diversity(self, pop: np.ndarray) -> float:
        """Calculate normalized pairwise Hamming distance across population."""
        n, l = pop.shape
        total_dist = 0
        num_pairs = n * (n - 1) // 2

        for i in range(n):
            for j in range(i + 1, n):
                diffs = np.count_nonzero(pop[i] != pop[j])
                total_dist += diffs

        h_norm = total_dist / (num_pairs * l)
        return float(np.clip(h_norm, 0.0, 1.0))

    def adapt_parameters(self, h_norm: float) -> Tuple[float, float]:
        """Linear span control law over the full configured ranges.

        High diversity (H -> 1): crossover high, mutation low.
        Low diversity  (H -> 0): crossover low, mutation high.

        Both parameters traverse their declared bounds, so the controller is
        not restricted to a near-constant band.
        """
        h = float(np.clip(h_norm, 0.0, 1.0))
        pc = self.pc_min + (self.pc_max - self.pc_min) * h
        pm = self.pm_min + (self.pm_max - self.pm_min) * (1.0 - h)
        return float(pc), float(pm)

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

    def two_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray, pc: float) -> Tuple[np.ndarray, np.ndarray]:
        """Two-point crossover."""
        if self.rng.random() > pc:
            return parent1.copy(), parent2.copy()

        length = len(parent1)
        pt1, pt2 = sorted(self.rng.choice(length, size=2, replace=False))
        child1 = parent1.copy()
        child2 = parent2.copy()
        child1[pt1:pt2] = parent2[pt1:pt2]
        child2[pt1:pt2] = parent1[pt1:pt2]
        return child1, child2

    def mutate(self, chromosome: np.ndarray, pm: float) -> np.ndarray:
        """Random slot reassignment mutation."""
        mutated = chromosome.copy()
        for gene_idx in range(len(mutated)):
            if self.rng.random() < pm:
                mutated[gene_idx] = self.rng.integers(0, self.problem.total_slots)
        return mutated

    def elite_local_search(
        self,
        chromosome: np.ndarray,
        hcv: int,
        scv: int,
        fitness: int,
        max_steps: int | None = None,
    ) -> Tuple[np.ndarray, int, int, int]:
        """
        Stochastic hill climbing with reassignment and swap moves.

        Targets lectures involved in hard conflicts while infeasible, and
        lectures involved in soft violations once feasible. The candidate pool
        is refreshed every 10 steps to keep the scan affordable.
        """
        best_chrom = chromosome.copy()
        curr_hcv, curr_scv, curr_fit = hcv, scv, fitness
        pool: List[int] = []
        steps = self.els_max_steps if max_steps is None else int(max_steps)

        for step in range(steps):
            if self.eval_count >= self.max_evaluations:
                break

            if self.els_target == "random":
                # Control arm: identical move set and budget, but no
                # conflict/soft targeting.
                pool = []
            elif step % 10 == 0:
                conflicted = self.problem.get_conflicted_lectures(best_chrom)
                if conflicted:
                    pool = conflicted
                else:
                    pool = self.problem.get_soft_violation_lectures(best_chrom)

            if pool:
                target_lec = int(self.rng.choice(pool))
            else:
                target_lec = int(self.rng.integers(0, self.problem.total_lectures))

            candidate = best_chrom.copy()
            if self.rng.random() < 0.5:
                # Swap two lectures: preserves the slot multiset and can move a
                # course to a better room/period without a duplicate-slot
                # intermediate.
                other = int(self.rng.integers(0, self.problem.total_lectures))
                candidate[target_lec], candidate[other] = (
                    candidate[other],
                    candidate[target_lec],
                )
            else:
                candidate[target_lec] = self.rng.integers(0, self.problem.total_slots)

            cand_hcv, cand_scv, cand_fit = self._eval(candidate)

            # First-improvement criterion: accept if fitness improves
            if cand_fit < curr_fit:
                best_chrom = candidate
                curr_hcv, curr_scv, curr_fit = cand_hcv, cand_scv, cand_fit

        return best_chrom, curr_hcv, curr_scv, curr_fit

    def run(self) -> Dict:
        start_time = time.time()
        self.eval_count = 0

        pop, hcvs, scvs, fitnesses = self.initialize_population()

        best_idx = np.argmin(fitnesses)
        best_hcv = hcvs[best_idx]
        best_scv = scvs[best_idx]
        best_fitness = fitnesses[best_idx]
        best_chrom = pop[best_idx].copy()

        initial_hnorm = self.calculate_diversity(pop)
        if self.enable_apc:
            init_pc, init_pm = self.adapt_parameters(initial_hnorm)
        else:
            init_pc, init_pm = self.pc_base, self.pm_base

        history = [{
            "generation": 0,
            "evaluations": self.eval_count,
            "best_hcv": int(best_hcv),
            "best_scv": int(best_scv),
            "best_fitness": int(best_fitness),
            "mean_fitness": float(np.mean(fitnesses)),
            "h_norm": initial_hnorm,
            "pc": init_pc,
            "pm": init_pm,
        }]

        generation = 0
        # Stage 1: population search on (1 - final_share) of the budget.
        # Stage 2: single-best intensification consumes the remainder exactly.
        population_budget = self.max_evaluations
        if self.enable_els and self.els_final_share > 0:
            population_budget = max(
                self.pop_size,
                int(self.max_evaluations * (1.0 - self.els_final_share)),
            )

        while self.eval_count < population_budget:
            generation += 1

            # 1. Diversity measurement & parameter adaptation
            if self.enable_apc:
                h_norm = self.calculate_diversity(pop)
                pc, pm = self.adapt_parameters(h_norm)
            else:
                # APC-disabled arms (GA_ELS, GA_ELS_Random, GA_Static) do not
                # use the diversity signal, so they must not pay its O(N^2 E)
                # per-generation cost.
                h_norm = 0.0
                pc, pm = self.pc_base, self.pm_base
            # 2. Elitism: preserve top individuals
            sorted_indices = np.argsort(fitnesses)
            new_pop = [pop[sorted_indices[i]].copy() for i in range(self.elitism_k)]
            new_hcvs = [hcvs[sorted_indices[i]] for i in range(self.elitism_k)]
            new_scvs = [scvs[sorted_indices[i]] for i in range(self.elitism_k)]
            new_fitnesses = [fitnesses[sorted_indices[i]] for i in range(self.elitism_k)]

            # 3. Selection, Crossover, Mutation for offspring
            while len(new_pop) < self.pop_size:
                if self.eval_count >= population_budget:
                    break

                p1_idx = self.tournament_selection(fitnesses)
                p2_idx = self.tournament_selection(fitnesses)

                c1, c2 = self.two_point_crossover(pop[p1_idx], pop[p2_idx], pc)
                c1 = self.mutate(c1, pm)
                c2 = self.mutate(c2, pm)

                if self.eval_count < population_budget:
                    h1, s1, f1 = self._eval(c1)
                    new_pop.append(c1)
                    new_hcvs.append(h1)
                    new_scvs.append(s1)
                    new_fitnesses.append(f1)

                if len(new_pop) < self.pop_size and self.eval_count < population_budget:
                    h2, s2, f2 = self._eval(c2)
                    new_pop.append(c2)
                    new_hcvs.append(h2)
                    new_scvs.append(s2)
                    new_fitnesses.append(f2)

            pop = np.array(new_pop)
            hcvs = np.array(new_hcvs)
            scvs = np.array(new_scvs)
            fitnesses = np.array(new_fitnesses)

            # 4. Generational Elite Local Search (ELS) on top elite individuals
            if self.enable_els and self.eval_count < population_budget:
                current_sorted = np.argsort(fitnesses)
                elite_targets = current_sorted[:min(self.els_elite_count, len(pop))]

                for e_idx in elite_targets:
                    if self.eval_count >= population_budget:
                        break
                    refined_chrom, ref_h, ref_s, ref_f = self.elite_local_search(
                        pop[e_idx], hcvs[e_idx], scvs[e_idx], fitnesses[e_idx]
                    )
                    pop[e_idx] = refined_chrom
                    hcvs[e_idx] = ref_h
                    scvs[e_idx] = ref_s
                    fitnesses[e_idx] = ref_f

            # Update overall best
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
                "h_norm": h_norm,
                "pc": pc,
                "pm": pm,
            })

        # Stage 2: intensify on the best solution found, using the remaining
        # share of the budget. Same move set, same targeting, but concentrated
        # on a single trajectory instead of five elites per generation.
        if self.enable_els and self.eval_count < self.max_evaluations:
            refined_chrom, ref_h, ref_s, ref_f = self.elite_local_search(
                best_chrom,
                best_hcv,
                best_scv,
                best_fitness,
                max_steps=self.max_evaluations,
            )
            if ref_f < best_fitness:
                best_chrom = refined_chrom
                best_hcv = ref_h
                best_scv = ref_s
                best_fitness = ref_f
            history[-1].update({
                "evaluations": self.eval_count,
                "best_hcv": int(best_hcv),
                "best_scv": int(best_scv),
                "best_fitness": int(best_fitness),
            })

        elapsed = time.time() - start_time

        if self.label:
            algo_name = self.label
        else:
            algo_name = "HAGA"
            if not self.enable_apc and not self.enable_els:
                algo_name = "GA_Static"
            elif self.enable_apc and not self.enable_els:
                algo_name = "GA_APC"
            elif not self.enable_apc and self.enable_els:
                algo_name = "GA_ELS"

        return {
            "algorithm": algo_name,
            "best_chromosome": best_chrom.copy(),
            "best_hcv": int(best_hcv),
            "best_scv": int(best_scv),
            "best_fitness": int(best_fitness),
            "total_evaluations": self.eval_count,
            "generations": generation,
            "history": history,
            "runtime_seconds": elapsed,
        }
