"""Tests for the Nemenyi critical-difference helper and ablation parameters."""

from __future__ import annotations

import math
import unittest
from argparse import Namespace

from src.stats import compute_nemenyi_cd


class NemenyiCdTests(unittest.TestCase):
    def test_alpha_05_known_value(self) -> None:
        self.assertAlmostEqual(compute_nemenyi_cd(4, 21, 0.05), 1.0235, places=4)

    def test_alpha_10_is_smaller(self) -> None:
        self.assertAlmostEqual(compute_nemenyi_cd(4, 21, 0.10), 0.9128, places=4)

    def test_deduplicated_n(self) -> None:
        self.assertAlmostEqual(compute_nemenyi_cd(4, 20, 0.05), 1.0488, places=4)

    def test_unsupported_alpha_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_nemenyi_cd(4, 21, 0.20)


class AblationParameterTests(unittest.TestCase):
    def test_random_els_control_is_configured(self) -> None:
        from src.runner import _parameters

        params = _parameters("GA_ELS_Random", 50, 50_000)
        self.assertFalse(params["enable_apc"])
        self.assertTrue(params["enable_els"])
        self.assertEqual(params["els_target"], "random")

    def test_canonical_ga_uses_080(self) -> None:
        from src.runner import _parameters

        self.assertEqual(_parameters("Canonical_GA", 50, 50_000)["pc"], 0.80)

    def test_haga_defaults(self) -> None:
        from src.runner import _parameters

        params = _parameters("HAGA", 50, 50_000)
        self.assertTrue(params["enable_apc"])
        self.assertTrue(params["enable_els"])
        self.assertEqual(params["els_target"], "violations")


if __name__ == "__main__":
    unittest.main()
