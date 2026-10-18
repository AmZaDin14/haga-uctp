# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUnannotatedClassAttribute=false
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="skfuzzy.*")

"""Fuzzy Parameter Controller for SAGA Reconstructed (Self-Adaptive GA).

Implements a Mamdani Fuzzy Inference System (FIS) adapting crossover probability (Pc)
and mutation probability (Pm) dynamically based on:
1. Diversity Index (Normalized genotypic Hamming distance in [0, 1])
2. Stagnation Counter (Generations without best-fitness improvement, normalized in [0, 1])
3. Improvement Rate (Relative fitness improvement in [0, 1])

Output parameter ranges follow Pamungkas & Neta (2025):
- Crossover rate Pc in [0.40, 0.95]
- Mutation rate Pm in [0.10, 0.90]
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyParameterController:
    """Mamdani Fuzzy Inference System for adaptive GA parameter control."""

    def __init__(
        self,
        pc_min: float = 0.40,
        pc_max: float = 0.95,
        pm_min: float = 0.10,
        pm_max: float = 0.90,
        resolution: int = 101,
    ) -> None:
        self.pc_min = pc_min
        self.pc_max = pc_max
        self.pm_min = pm_min
        self.pm_max = pm_max

        # 1. Universes of Discourse
        u_norm = np.linspace(0.0, 1.0, resolution)
        u_pc = np.linspace(pc_min, pc_max, resolution)
        u_pm = np.linspace(pm_min, pm_max, resolution)

        # 2. Antecedents (Inputs)
        self.diversity = ctrl.Antecedent(u_norm, "diversity")
        self.stagnation = ctrl.Antecedent(u_norm, "stagnation")
        self.improvement = ctrl.Antecedent(u_norm, "improvement")

        # 3. Consequents (Outputs)
        self.crossover = ctrl.Consequent(u_pc, "crossover")
        self.mutation = ctrl.Consequent(u_pm, "mutation")

        # 4. Membership Functions
        # Diversity: Low, Medium, High
        self.diversity["low"] = fuzz.trimf(self.diversity.universe, [0.0, 0.0, 0.5])
        self.diversity["medium"] = fuzz.trimf(self.diversity.universe, [0.2, 0.5, 0.8])
        self.diversity["high"] = fuzz.trimf(self.diversity.universe, [0.5, 1.0, 1.0])

        # Stagnation: Low, Medium, High
        self.stagnation["low"] = fuzz.trimf(self.stagnation.universe, [0.0, 0.0, 0.4])
        self.stagnation["medium"] = fuzz.trimf(self.stagnation.universe, [0.2, 0.5, 0.8])
        self.stagnation["high"] = fuzz.trimf(self.stagnation.universe, [0.6, 1.0, 1.0])

        # Improvement: Low, Medium, High
        self.improvement["low"] = fuzz.trimf(self.improvement.universe, [0.0, 0.0, 0.3])
        self.improvement["medium"] = fuzz.trimf(self.improvement.universe, [0.15, 0.5, 0.85])
        self.improvement["high"] = fuzz.trimf(self.improvement.universe, [0.5, 1.0, 1.0])

        # Crossover (Pc): Low, Medium, High
        self.crossover["low"] = fuzz.trimf(
            self.crossover.universe, [pc_min, pc_min, pc_min + 0.5 * (pc_max - pc_min)]
        )
        self.crossover["medium"] = fuzz.trimf(
            self.crossover.universe,
            [
                pc_min + 0.18 * (pc_max - pc_min),
                pc_min + 0.50 * (pc_max - pc_min),
                pc_min + 0.82 * (pc_max - pc_min),
            ],
        )
        self.crossover["high"] = fuzz.trimf(
            self.crossover.universe, [pc_min + 0.5 * (pc_max - pc_min), pc_max, pc_max]
        )

        # Mutation (Pm): Low, Medium, High
        self.mutation["low"] = fuzz.trimf(
            self.mutation.universe, [pm_min, pm_min, pm_min + 0.5 * (pm_max - pm_min)]
        )
        self.mutation["medium"] = fuzz.trimf(
            self.mutation.universe,
            [
                pm_min + 0.25 * (pm_max - pm_min),
                pm_min + 0.50 * (pm_max - pm_min),
                pm_min + 0.75 * (pm_max - pm_min),
            ],
        )
        self.mutation["high"] = fuzz.trimf(
            self.mutation.universe, [pm_min + 0.5 * (pm_max - pm_min), pm_max, pm_max]
        )

        # 5. Fuzzy Rule Base (Synthesis of Herrera & Lozano 2003 and SAGA architecture)
        rules = [
            # Stagnation High: need exploration
            ctrl.Rule(self.stagnation["high"] & self.diversity["low"], (self.crossover["low"], self.mutation["high"])),
            ctrl.Rule(self.stagnation["high"] & self.diversity["medium"], (self.crossover["medium"], self.mutation["high"])),
            ctrl.Rule(self.stagnation["high"] & self.diversity["high"], (self.crossover["high"], self.mutation["medium"])),

            # Stagnation Medium: balanced control
            ctrl.Rule(self.stagnation["medium"] & self.diversity["low"], (self.crossover["low"], self.mutation["high"])),
            ctrl.Rule(self.stagnation["medium"] & self.diversity["medium"], (self.crossover["medium"], self.mutation["medium"])),
            ctrl.Rule(self.stagnation["medium"] & self.diversity["high"], (self.crossover["high"], self.mutation["low"])),

            # Stagnation Low: active evolution
            ctrl.Rule(self.stagnation["low"] & self.diversity["low"], (self.crossover["medium"], self.mutation["medium"])),
            ctrl.Rule(self.stagnation["low"] & self.diversity["medium"], (self.crossover["high"], self.mutation["low"])),
            ctrl.Rule(self.stagnation["low"] & self.diversity["high"], (self.crossover["high"], self.mutation["low"])),

            # Improvement Modulation
            ctrl.Rule(self.improvement["high"], (self.crossover["high"], self.mutation["low"])),
            ctrl.Rule(self.improvement["low"] & self.stagnation["high"], (self.crossover["low"], self.mutation["high"])),
        ]

        self.control_system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    def evaluate(
        self,
        diversity_index: float,
        stagnation_counter: int,
        max_stagnation: int = 15,
        improvement_rate: float = 0.0,
    ) -> tuple[float, float]:
        """Compute adaptive (Pc, Pm) via Mamdani fuzzy inference.

        Parameters
        ----------
        diversity_index : float
            Normalized pairwise Hamming distance, in [0.0, 1.0].
        stagnation_counter : int
            Generations since best fitness was improved.
        max_stagnation : int
            Stagnation horizon for normalization.
        improvement_rate : float
            Relative fitness improvement over recent generation.

        Returns
        -------
        tuple[float, float]
            (Pc, Pm) clamped within configured bounds.
        """
        # Clamp inputs safely into [0.0, 1.0]
        d_val = float(np.clip(diversity_index, 0.0, 1.0))
        s_val = float(np.clip(stagnation_counter / max(1, max_stagnation), 0.0, 1.0))
        i_val = float(np.clip(improvement_rate, 0.0, 1.0))

        try:
            self.simulation.input["diversity"] = d_val
            self.simulation.input["stagnation"] = s_val
            self.simulation.input["improvement"] = i_val
            self.simulation.compute()

            pc = float(self.simulation.output["crossover"])
            pm = float(self.simulation.output["mutation"])
        except Exception:
            # Fallback to midpoint values in case of any numerical issue
            pc = 0.5 * (self.pc_min + self.pc_max)
            pm = 0.5 * (self.pm_min + self.pm_max)

        # Explicit clamping to strict bounds
        pc = float(np.clip(pc, self.pc_min, self.pc_max))
        pm = float(np.clip(pm, self.pm_min, self.pm_max))

        return pc, pm
