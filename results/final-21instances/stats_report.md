# HAGA benchmark — final statistics

- Runs analysed: **2520**
- Inferential instances: **20** (comp15 excluded as a duplicate of comp03)

## Mean ranks (20 unique instances)

| Algorithm | Mean rank |
|---|---|
| Simulated_Annealing | 1.000 |
| HAGA | 2.000 |
| Canonical_GA | 3.200 |
| Discrete_PSO | 3.800 |

Friedman chi2 = 56.1600, p = 3.883e-12, df = 3
Iman-Davenport F = 277.875
Nemenyi CD (alpha=0.05) = 1.0488

## Pairwise rank differences

| A | B | Δ rank | Significant (0.05) |
|---|---|---|---|
| Canonical_GA | Discrete_PSO | 0.600 | no |
| Canonical_GA | Simulated_Annealing | 2.200 | yes |
| Canonical_GA | HAGA | 1.200 | yes |
| Discrete_PSO | Simulated_Annealing | 2.800 | yes |
| Discrete_PSO | HAGA | 1.800 | yes |
| Simulated_Annealing | HAGA | 1.000 | no |

## Feasibility

| Algorithm | Feasible | Runs | Rate | 95% Wilson |
|---|---|---|---|---|
| Canonical_GA | 0 | 630 | 0.0% | [-0.0, 0.6] |
| Discrete_PSO | 0 | 630 | 0.0% | [-0.0, 0.6] |
| Simulated_Annealing | 365 | 630 | 57.9% | [54.0, 61.7] |
| HAGA | 182 | 630 | 28.9% | [25.5, 32.5] |

## HAGA vs SA, per-instance paired Wilcoxon (Holm)

| Instance | Median Δ SCV | p | p_Holm |
|---|---|---|---|
| comp01 | 1326.5 | 1.863e-09 | 0.000 |
| comp02 | 5203.0 | 1.863e-09 | 0.000 |
| comp03 | 4733.0 | 1.863e-09 | 0.000 |
| comp04 | 2245.5 | 1.863e-09 | 0.000 |
| comp05 | 5157.0 | 1.863e-09 | 0.000 |
| comp06 | 4375.5 | 1.863e-09 | 0.000 |
| comp07 | 3257.5 | 1.863e-09 | 0.000 |
| comp08 | 2611.0 | 1.863e-09 | 0.000 |
| comp09 | 3610.5 | 1.863e-09 | 0.000 |
| comp10 | 3354.0 | 1.863e-09 | 0.000 |
| comp11 | 432.0 | 1.732e-06 | 0.000 |
| comp12 | 2244.5 | 1.863e-09 | 0.000 |
| comp13 | 2749.0 | 1.863e-09 | 0.000 |
| comp14 | 3142.5 | 1.863e-09 | 0.000 |
| comp16 | 3986.5 | 1.732e-06 | 0.000 |
| comp17 | 4161.5 | 1.863e-09 | 0.000 |
| comp18 | 193.0 | 1.733e-06 | 0.000 |
| comp19 | 3969.0 | 1.733e-06 | 0.000 |
| comp20 | 4725.5 | 1.863e-09 | 0.000 |
| comp21 | 4231.5 | 1.863e-09 | 0.000 |
