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

Reproduce all experimental results and statistical figures in three simple steps:

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

### 3. Run the Benchmark Experiment
```bash
python3 -m src.runner
```

The script will:
* Run 10 independent seeds for both Canonical GA and HAGA under identical evaluation quotas ($B_{\max} = 25{,}000$).
* Execute the Wilcoxon signed-rank test.
* Output `prapenelitian_summary.csv`, `prapenelitian_convergence.csv`, and regenerate all comparison charts into `results/`.

---

## 👥 Authors & Affiliation

* **Researcher:** Amri Reza Wahyudin (NPM: 25321019)  
  *Email:* amrireza97@gmail.com
* **Supervisor:** Dr. Rohmat Indra Borman, M.Kom.
* **Institution:**  
  Program Studi Magister Ilmu Komputer, Fakultas Teknik dan Ilmu Komputer,  
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
