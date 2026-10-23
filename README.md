# HAGA: Hybrid Adaptive Genetic Algorithm with Diversity-Guided Parameter Control and Elite Local Search for University Course Timetabling

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark](https://img.shields.io/badge/Benchmark-ITC%202007%20CB--CTT-green.svg)](http://www.cs.qub.ac.uk/itc2007/)
[![Status](https://img.shields.io/badge/Artifact-Reproducible-brightgreen.svg)](#reproduction-guide)

Open-source reference implementation and reproducible experimental harness for the Master's Thesis research:  
**"Pengembangan Model Algoritma Genetika Hibrida Adaptif dengan Pencarian Lokal Elit untuk Optimasi Penjadwalan Perkuliahan"**  
Department of Master of Computer Science, Universitas Teknokrat Indonesia (2026).

---

## 📌 Overview

The **University Course Timetabling Problem (UCTP)** is an NP-hard combinatorial optimization problem characterized by factorial search space complexity $\mathcal{O}(n!)$. Conventional Genetic Algorithms (GAs) with static parameterization suffer from premature convergence to infeasible local optima. 

**HAGA (Hybrid Adaptive Genetic Algorithm)** introduces a deterministic, computationally lightweight approach that balances global exploration and local exploitation through five architectural contracts:

1. **Event-Indexed 1D Chromosome Representation:** Genotype length strictly equals total lecture sessions $E$. Each gene encodes a bijectively mapped room and timeslot $(r_k, t_k)$, while lecturer and curriculum constraints are evaluated transparently via Hard Constraint Violations ($HCV$).
2. **Genotypic Population Diversity Metric ($H_{\text{norm}}$):** Population dispersion is measured every generation across all $\frac{N(N-1)}{2}$ individual pairs using normalized pairwise Hamming distance:
   $$H_{\text{norm}}(t) = \frac{2}{N(N-1)} \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} \left[ \frac{1}{E} \sum_{k=1}^{E} \mathbb{I}(c_{i,k} \neq c_{j,k}) \right]$$
3. **Linear Adaptive Parameter Control (APC):** Direct, closed-form feedback updates crossover ($P_c$) and mutation ($P_m$) rates with explicit clamping bounds ($P_c \in [0.40, 0.95], P_m \in [0.01, 0.30]$), eliminating the computational overhead of fuzzy rule bases.
4. **Conflict-Directed Elite Local Search (ELS):** Targeted stochastic first-improvement hill climbing applied exclusively to the top 10% elite individuals with a strict budget of $S_{\max} = 50$ steps per individual and Lamarckian inheritance.
5. **Fair Computational Budget Parity ($B_{\max}$):** Strict evaluation accounting where every objective evaluation (offspring and local search) decrements a shared global quota ($B_{\max} = 25{,}000$ in preliminary tests, $50{,}000$ in full benchmark), guaranteeing hardware-agnostic fair comparison against baselines.

---

## 🔬 Algorithm Architecture

```
                                [ START ]
                                    │
                                    ▼
                      [ Input CTT & Base Config ]
                                    │
                                    ▼
                      [ Initialize Population (N) ]
                                    │
                                    ▼
                    [ Evaluate Initial Fitness (HCV, SCV) ]
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                       ▼
        │                    [ Measure Genotypic Diversity H_norm ] ──┐
        │                               │                             │
        │                               ▼                             │ HAGA
        │                [ Adapt Parameters Pc & Pm via APC ] ────────┤ Components
        │                               │                             │
        │                               ▼                             │
        │                  [ Tournament Selection (k=3) ]             │
        │                               │                             │
        │                               ▼                             │
        │                [ Two-Point Crossover & Mutation ]           │
        │                               │                             │
        │                               ▼                             │
        │               [ Conflict-Directed ELS (Top 10%) ] ──────────┘
        │                               │
        │                               ▼
        │              < HCV == 0 OR Total Evals >= B_max? >
        │                               │
        └─── [ No (Next Gen) ] ─────────┤
                                        ▼ [ Yes ]
                                    [ END ]
```

---

## 📊 Preliminary Feasibility Study Results

Benchmarked on official competition instance **ITC 2007 CB-CTT `comp01.ctt`** (30 courses, 6 rooms, 53 constraints) across **10 independent random seeds** under shared budget $B_{\max} = 25{,}000$ objective evaluations:

| Evaluation Metric | Canonical GA (Static) | HAGA (Proposed) | Difference ($\Delta$) | Statistical Test |
| :--- | :---: | :---: | :---: | :---: |
| **Feasibility Rate ($HCV = 0$)** | **0% (0/10)** | **100% (10/10)** | **+100% (Absolute Feasibility)** | — |
| **Best Hard Constraint Violations ($HCV_{\min}$)** | 40 | **0** | **-40 (Conflict-free)** | — |
| **Mean Hard Violations ($HCV_{\text{mean}}$)** | $48.0 \pm 4.6$ | **0.0** | **-48.0** | $p < 0.001$ |
| **Best Soft Constraint Penalty ($SCV_{\min}$)** | 1,769 | **1,300** | **-469 (-26.5%)** | — |
| **Mean Soft Penalty ($SCV_{\text{mean}}$)** | $2,221.0 \pm 174.8$ | **1,748.0** | **-473.0 (-21.3%)** | **$p = 0.00195$** |
| **Mean Wall-Clock Runtime** | 107.4 s | **87.4 s** | **-20.0 s (-18.6%)** | — |

> **Non-parametric Statistical Validation:**  
> Wilcoxon Signed-Rank Test on paired runs ($N=10$) demonstrates that HAGA's soft constraint reduction is statistically significant at $\alpha = 0.01$ ($W = 55.0, p = 0.001953$).

---

## 📁 Repository Structure

```text
.
├── LICENSE                          # MIT License
├── README.md                        # Documentation & reproduction guide
├── requirements.txt                 # Pinned dependencies
├── data/
│   └── itc2007/
│       ├── comp01.ctt               # ITC 2007 CB-CTT Benchmark instance 01
│       └── comp02.ctt               # ITC 2007 CB-CTT Benchmark instance 02
├── figures/
│   ├── haga_algorithm_flowchart.png # Methodological flowchart
│   └── flowchart.png                # Research stage diagram
├── results/
│   ├── prapenelitian_summary.csv     # Raw 10-seed metrics (HCV, SCV, time)
│   ├── prapenelitian_convergence.csv # Full generational convergence logs
│   ├── boxplot_comparison.png        # Solution distribution boxplots
│   ├── convergence_comparison.png    # Fitness convergence trajectory
│   └── haga_diversity_trace.png      # H_norm, Pc, and Pm dynamics over time
└── src/
    ├── __init__.py                  # Package definition
    ├── loader.py                    # CTT instance parser
    ├── problem.py                   # Chromosome decoder and HCV/SCV evaluators
    ├── canonical_ga.py              # Canonical Genetic Algorithm (static baseline)
    ├── haga.py                      # HAGA engine (Hamming diversity + APC + ELS)
    └── runner.py                    # Multiprocess experiment orchestrator
```

---

## 🚀 Reproduction Guide

Install the dependencies, then check the pipeline before launching the full benchmark.

### 1. Clone the Repository
```bash
git clone https://github.com/AmZaDin14/haga-uctp.git
cd haga-uctp
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Check the Multi-Instance Pipeline

```bash
python scripts/run_benchmark_suite.py \
  --experiment-id pipeline-smoke --stage smoke \
  --evals 200 --pop 10 --seeds 3 --workers 3
```

This runs Canonical GA, Discrete PSO, SA and HAGA on `comp01`, `comp07` and
`comp12`: 36 runs with a small evaluation budget. Smoke and pilot reports are
descriptive only; they do not make inferential performance claims.

### 4. Pilot at the Final Evaluation Budget

```bash
python scripts/run_benchmark_suite.py \
  --experiment-id comparative-pilot --stage pilot \
  --evals 50000 --pop 50 --seeds 3 --workers 10
```

The same three instances exercise the pipeline at the final budget. Check
completion, budget accounting, per-run persistence and resume before proceeding.
Pilot success does not require HAGA to outperform the baselines.

### 5. Run All 21 Instances

```bash
python scripts/run_benchmark_suite.py \
  --experiment-id comparative-main --stage main \
  --evals 50000 --pop 50 --seeds 30 --workers 10
```

Main mode defaults to `comp01`–`comp21` and requires all 21 instance IDs and 30
seeds. Instances execute sequentially; the worker limit applies to the whole
experiment, not separately to every instance. This configuration has 2,520 runs.
Use `--mode ablation` for the four-stage ablation instead of the comparative suite.
For a single-instance experiment, use `python -m src.runner` with the same
required `--experiment-id` and an explicit `--instance data/itc2007/comp01.ctt`.
Custom subsets use `--instances <path> <path> ...` in smoke/pilot mode.

### Persistence, Resume and Reports

```text
results/<experiment-id>/
  manifest.json
  execution.log
  experiment_summary.csv
  per_instance_algorithm_summary.csv
  instance_fitness_medians.csv
  analysis.json
  comp01/
    runs/<algorithm>/<seed>.json
    experiment_summary.csv
    experiment_convergence.csv
    convergence.png
    boxplot.png
```

- Each completed run commits its summary and history together as an atomic JSON
  record, immediately when its worker finishes. CSVs and plots are rebuildable
  exports, not the source of completion status.
- Repeat the exact command to resume. Valid completed records are reused without
  modification; missing or corrupt records are recomputed. A failed attempt has
  a separate `<seed>.failed.json` marker and never counts as completion.
- The manifest freezes dataset paths/hashes, resolved algorithm parameters,
  seeds, budget, source hashes, Git revision, Python/dependency versions and
  machine identity. Configuration/provenance changes require a new experiment
  ID; worker count may change when resuming and is recorded in `execution.log`.
- One writer may own an experiment directory at a time (POSIX advisory lock).
  Ctrl-C or SIGTERM stops new submissions, finishes and saves active runs, then
  exits with status 130. Failed/incomplete execution exits nonzero.
- Add `--analyze-only` to the same command to regenerate CSVs, plots and analysis
  without executing missing runs. An incomplete grid is reported explicitly and
  returns nonzero. Stored run records are unchanged.
- Analysis uses one median internal fitness per instance–algorithm. Only a
  complete main experiment enables Friedman across instances and two-sided
  HAGA-versus-baseline Wilcoxon comparisons with Holm correction. Seeds are
  nested repetitions, not independent instance-level statistical units.
- HCV and feasibility are reported separately. SCV distributions and summaries
  use feasible runs only; algorithms without feasible runs are labelled rather
  than assigned a zero cost. Convergence medians align logged histories at
  common evaluation budgets using best-so-far step values.
- Existing files directly under `results/` are historical outputs and are not
  overwritten or automatically imported. Pilot and main IDs also remain
  separate: this runner does not silently pool results from different manifests.

Run the storage and analysis regressions with:

```bash
python -m unittest discover -s tests -v
```

---

## 👥 Authors & Affiliation

* **Researcher:** Amri Reza Wahyudin (NPM: 25321019)  
  *Email:* amri\_reza\_wahyudin@teknokrat.ac.id / amrireza97@gmail.com
* **Supervisor:** Dr. Rohmat Indra Borman, M.Kom.
* **Institution:**  
  Faculty of Engineering and Computer Science,  
  Universitas Teknokrat Indonesia, Bandar Lampung, Indonesia.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📑 Citation

If you use this codebase or methodology in your research, please cite:

```bibtex
@thesis{wahyudin2026haga,
  author      = {Amri Reza Wahyudin},
  title       = {Pengembangan Model Algoritma Genetika Hibrida Adaptif dengan Pencarian Lokal Elit untuk Optimasi Penjadwalan Perkuliahan},
  type        = {Master's thesis},
  institution = {Universitas Teknokrat Indonesia},
  year        = {2026},
  address     = {Bandar Lampung, Indonesia}
}
```
