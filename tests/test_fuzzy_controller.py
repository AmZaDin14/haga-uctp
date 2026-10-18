"""Unit tests for FuzzyParameterController in SAGA Reconstructed."""
import unittest
from typing import override

from src.fuzzy_controller import FuzzyParameterController


class FuzzyParameterControllerTests(unittest.TestCase):
    fis: FuzzyParameterController

    @override
    def setUp(self) -> None:
        self.fis = FuzzyParameterController(
            pc_min=0.40,
            pc_max=0.95,
            pm_min=0.10,
            pm_max=0.90,
        )

    def test_output_bounds_are_strictly_respected(self):
        """Verify Pc and Pm stay strictly within clamped bounds for grid of inputs."""
        diversities = [0.0, 0.25, 0.5, 0.75, 1.0]
        stagnations = [0, 5, 10, 15, 20]
        improvements = [0.0, 0.1, 0.5, 1.0]

        for d in diversities:
            for s in stagnations:
                for imp in improvements:
                    pc, pm = self.fis.evaluate(d, s, max_stagnation=15, improvement_rate=imp)
                    self.assertGreaterEqual(pc, 0.40)
                    self.assertLessEqual(pc, 0.95)
                    self.assertGreaterEqual(pm, 0.10)
                    self.assertLessEqual(pm, 0.90)

    def test_stagnant_low_diversity_promotes_mutation(self):
        """When population is stagnant and diversity is low, mutation should be elevated."""
        pc_stagnant, pm_stagnant = self.fis.evaluate(
            diversity_index=0.05,
            stagnation_counter=15,
            max_stagnation=15,
            improvement_rate=0.0,
        )
        pc_healthy, pm_healthy = self.fis.evaluate(
            diversity_index=0.85,
            stagnation_counter=0,
            max_stagnation=15,
            improvement_rate=0.8,
        )
        # Stagnant scenario should have higher mutation to escape
        self.assertGreater(pm_stagnant, pm_healthy)
        # Healthy improving scenario should have higher crossover
        self.assertGreater(pc_healthy, pc_stagnant)

    def test_robustness_against_out_of_bound_inputs(self):
        """Ensure invalid/extreme inputs are safely handled without exception."""
        # Extreme negative and excessive positive
        pc1, pm1 = self.fis.evaluate(-1.0, -5, max_stagnation=10, improvement_rate=-0.5)
        self.assertGreaterEqual(pc1, 0.40)
        self.assertLessEqual(pc1, 0.95)
        self.assertGreaterEqual(pm1, 0.10)
        self.assertLessEqual(pm1, 0.90)

        pc2, pm2 = self.fis.evaluate(5.0, 100, max_stagnation=10, improvement_rate=10.0)
        self.assertGreaterEqual(pc2, 0.40)
        self.assertLessEqual(pc2, 0.95)
        self.assertGreaterEqual(pm2, 0.10)
        self.assertLessEqual(pm2, 0.90)

    def test_determinism(self):
        """Multiple evaluations with identical inputs yield identical outputs."""
        pc_a, pm_a = self.fis.evaluate(0.35, 7, 15, 0.12)
        pc_b, pm_b = self.fis.evaluate(0.35, 7, 15, 0.12)
        self.assertAlmostEqual(pc_a, pc_b, places=6)
        self.assertAlmostEqual(pm_a, pm_b, places=6)


if __name__ == "__main__":
    unittest.main()
