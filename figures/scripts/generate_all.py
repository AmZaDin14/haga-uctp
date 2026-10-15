"""Generate all diagrams for the proposal."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# FIGURE 1: Theoretical Framework
# ============================================================
def draw_theoretical_framework():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    blue = '#4472C4'
    orange = '#ED7D31'
    green = '#70AD47'
    red = '#C55A11'
    purple = '#7030A0'

    def box(x, y, w, h, text, color=blue, tc='white', fs=8):
        r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                     facecolor=color, edgecolor='black', lw=1)
        ax.add_patch(r)
        lines = text.split('\n')
        for i, ln in enumerate(lines):
            ax.text(x + w/2, y + h - 0.25 - i*0.22, ln, ha='center', va='top',
                    fontsize=fs, color=tc, fontweight='bold')

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

    # Title
    ax.text(5, 7.7, 'Theoretical Framework', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Top: UCTP Problem
    box(2.5, 6.2, 5, 0.8, 'UCTP: NP-Hard Combinatorial Optimization\nUniversity Course Timetabling Problem', purple, 'white', 8)

    # Left branch: Parameter Control Theory
    box(0.3, 4.5, 3.5, 1.2, 'Parameter Control Theory\n(Eiben, 1999; Karafotias, 2015)\n• Deterministic control\n• Adaptive control\n• Self-adaptive control\n• Gap: No diversity-guided\n  method for UCTP', blue, 'white', 7)

    # Center: Existing Adaptive GA Methods
    box(3.5, 4.5, 3.5, 1.2, 'Existing Adaptive GA Methods\n(Cicirello, 2016; SAGA, 2025;\n Ghaffar, 2021)\n• Fuzzy logic control\n• Cooperative coevolution\n• Hyper-heuristics\n• Limitation: Complex, slow,\n  high variability', orange, 'white', 7)

    # Right branch: Indonesian UCTP
    box(7, 4.5, 2.7, 1.2, 'Indonesian UCTP Studies\n(Sari, 2019; Kristiadi, 2019;\nFachrie, 2021; Ikhwani, 2022)\n• Single-institution data\n• No benchmark datasets\n• No statistical rigor\n• No adaptive mechanisms', blue, 'white', 7)

    arrow(5, 6.2, 5, 5.7)
    arrow(2.05, 5.7, 2.05, 5.7)
    ax.annotate('', xy=(2.05, 5.7), xytext=(5, 5.7),
                arrowprops=dict(arrowstyle='->', lw=1, color='gray'))
    ax.annotate('', xy=(5, 5.7), xytext=(8.35, 5.7),
                arrowprops=dict(arrowstyle='->', lw=1, color='gray'))

    # Research Gap
    box(2.5, 3.0, 5, 0.7, 'RESEARCH GAP\nNo diversity-guided GA for UCTP • No ITC 2007 in Indonesian studies\nNo rigorous statistics (30 runs, Wilcoxon, Friedman) in Indonesian studies',
         red, 'white', 7)
    arrow(5, 4.5, 5, 3.7)

    # Proposed HAGA
    box(1.5, 1.2, 7, 1.0, 'PROPOSED: HAGA — Hybrid Adaptive Genetic Algorithm\n1. Diversity-Guided Adaptive Parameter Control (Hamming distance)\n2. Elite Local Search (hill climbing on top 10%)\n3. Constraint-Aware Encoding',
         green, 'black', 7)
    arrow(5, 3.0, 5, 2.2)

    # Evaluation
    box(1.5, 0, 7, 0.6, 'EVALUATION: 3 datasets (UM Metro + ITC 2007 + Synthetic)\n4 algorithms (GA, PSO, SA, HAGA) | 30 runs | Wilcoxon + Friedman + Nemenyi',
         purple, 'white', 7)
    arrow(5, 1.2, 5, 0.6)

    plt.tight_layout()
    plt.savefig('proposal/figures/theoretical_framework.png', dpi=200, bbox_inches='tight')
    print("Theoretical framework PNG generated.")


# ============================================================
# FIGURE 2: Research Flowchart
# ============================================================
def draw_research_flowchart():
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    blue = '#4472C4'
    orange = '#ED7D31'
    green = '#70AD47'

    def box(x, y, w, h, text, color=blue, tc='white', fs=7):
        r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='black', lw=1)
        ax.add_patch(r)
        lines = text.split('\n')
        for i, ln in enumerate(lines):
            ax.text(x + w/2, y + h - 0.3 - i*0.22, ln, ha='center', va='top',
                    fontsize=fs, color=tc, fontweight='bold')

    def arr(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    ax.text(5, 13.5, 'Research Flowchart — HAGA for University Course Timetabling',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # INPUT
    box(1, 11.5, 8, 1.0, 'INPUT\n• UM Metro Dataset (~40 courses, 10 rooms, 20 lecturers)\n• ITC 2007 Benchmark (~100 courses)\n• Synthetic Dataset (~200 courses, 20 rooms)\n• Algorithm Parameters (N, G, Pc, Pm, alpha, beta)',
        green, 'black', 7)
    # Stage 1
    box(1, 9.5, 8, 1.0, 'STAGE 1: Literature Study & Gap Analysis\n• Systematic review of 30+ publications on adaptive GA, UCTP\n• Gap identification: no diversity-guided GA for UCTP,\n  no Indonesian UCTP study uses ITC 2007 benchmarks',
        blue, 'white', 7)
    arr(5, 11.5, 5, 10.5)
    # Stage 2
    box(1, 7.5, 8, 1.0, 'STAGE 2: HAGA Design & Implementation\n• Constraint-aware chromosome encoding\n• Diversity measurement (normalized Hamming distance)\n• Adaptive parameter control (Pc, Pm adjustment)\n• Elite local search (hill climbing on top 10%)',
        blue, 'white', 7)
    arr(5, 9.5, 5, 8.5)
    # Stage 3
    box(1, 5.7, 8, 0.8, 'STAGE 3: Baseline Implementation\n• Standard GA (fixed Pc=0.85, Pm=0.05)\n• PSO (inertia weight 0.7, c1=c2=1.5)\n• Simulated Annealing (T0=100, cooling=0.95)',
        blue, 'white', 7)
    arr(5, 7.5, 5, 6.5)
    # Stage 4
    box(1, 3.7, 8, 0.8, 'STAGE 4: Experimentation\n• 30 runs x 4 algorithms x 3 datasets = 360 executions\n• Ablation study: GA, GA+APC, GA+ELS, Full HAGA\n• Parameter sensitivity: N=50/100/200, alpha=0.1-0.5',
        orange, 'white', 7)
    arr(5, 5.7, 5, 4.5)
    # Stage 5
    box(1, 1.9, 8, 0.8, 'STAGE 5: Data Analysis\n• Descriptive statistics (mean, std, best, worst)\n• Wilcoxon signed-rank test (alpha=0.05)\n• Friedman + Nemenyi post-hoc test\n• Convergence & diversity trace visualization',
        orange, 'white', 7)
    arr(5, 3.7, 5, 2.7)
    # OUTPUT
    box(1, 0.2, 8, 0.8, 'OUTPUT\n• Optimized schedule for each dataset\n• Comparative performance tables & statistical results\n• Thesis BAB I-V & SINTA 2 journal paper',
        green, 'black', 7)
    arr(5, 1.9, 5, 1.0)

    # Legend
    ax.text(1, 12.8, '--- Input/Output', fontsize=8, color='green', fontweight='bold')
    ax.text(3.5, 12.8, '--- Research & Development', fontsize=8, color='blue', fontweight='bold')
    ax.text(6.5, 12.8, '--- Experimentation & Analysis', fontsize=8, color='orange', fontweight='bold')

    plt.tight_layout()
    plt.savefig('proposal/figures/research_flowchart.png', dpi=200, bbox_inches='tight')
    print("Research flowchart PNG generated.")


if __name__ == '__main__':
    draw_theoretical_framework()
    draw_research_flowchart()
    print("All diagrams generated successfully.")
