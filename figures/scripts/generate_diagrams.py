"""Generate research flowchart PNG for the proposal."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Colors
blue = '#4472C4'
orange = '#ED7D31'
green = '#70AD47'
gray = '#D9D9D9'

def draw_box(x, y, w, h, text, color=blue, text_color='white', fontsize=9):
    """Draw a rounded box with text."""
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    # Split text into lines
    lines = text.split('\n')
    line_h = 0.25
    start_y = y + h - 0.3 - (len(lines)-1) * line_h / 2
    for i, line in enumerate(lines):
        ax.text(x + w/2, start_y - i * line_h, line, ha='center', va='top',
                fontsize=fontsize, color=text_color, fontweight='bold')

def draw_arrow(x1, y1, x2, y2):
    """Draw arrow from (x1,y1) to (x2,y2)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# === Title ===
ax.text(5, 13.5, 'Research Flowchart — HAGA for University Course Timetabling',
        ha='center', va='center', fontsize=12, fontweight='bold')

# === INPUT ===
draw_box(1, 11.5, 8, 1.0, 'INPUT\n• UM Metro Dataset (~40 courses, 10 rooms, 20 lecturers)\n• ITC 2007 Benchmark (~100 courses)\n• Synthetic Dataset (~200 courses, 20 rooms)\n• Algorithm Parameters (N, G, Pc, Pm, α, β)',
         green, 'black', 7)

# === Stage 1 ===
draw_box(1, 9.5, 8, 1.0, 'STAGE 1: Literature Study & Gap Analysis\n• Systematic review of 30+ publications on adaptive GA, UCTP\n• Gap identification: no diversity-guided GA for UCTP,\n  no Indonesian UCTP with ITC 2007 benchmarks',
         blue, 'white', 7)
draw_arrow(5, 11.5, 5, 10.5)

# === Stage 2 ===
draw_box(1, 7.5, 8, 1.0, 'STAGE 2: HAGA Design & Implementation\n• Constraint-aware chromosome encoding\n• Diversity measurement (normalized Hamming distance)\n• Adaptive parameter control (Pc, Pm adjustment)\n• Elite local search (hill climbing on top 10%)\n• Repair mechanism',
         blue, 'white', 7)
draw_arrow(5, 9.5, 5, 8.5)

# === Stage 3 ===
draw_box(1, 5.7, 8, 0.8, 'STAGE 3: Baseline Implementation\n• Standard GA (fixed Pc=0.85, Pm=0.05)\n• PSO (inertia weight 0.7, c1=c2=1.5)\n• Simulated Annealing (T₀=100, cooling=0.95)',
         blue, 'white', 7)
draw_arrow(5, 7.5, 5, 6.5)

# === Stage 4 ===
draw_box(1, 3.7, 8, 0.8, 'STAGE 4: Experimentation\n• 30 independent runs × 4 algorithms × 3 datasets = 360 executions\n• Ablation study: GA → GA+APC → GA+ELS → Full HAGA\n• Parameter sensitivity: N=50/100/200, α=0.1-0.5, β=0.1-0.5',
         orange, 'white', 7)
draw_arrow(5, 5.7, 5, 4.5)

# === Stage 5 ===
draw_box(1, 1.9, 8, 0.8, 'STAGE 5: Data Analysis\n• Descriptive statistics (mean, std, best, worst)\n• Wilcoxon signed-rank test (α=0.05)\n• Friedman + Nemenyi post-hoc test\n• Convergence & diversity trace visualization',
         orange, 'white', 7)
draw_arrow(5, 3.7, 5, 2.7)

# === OUTPUT ===
draw_box(1, 0.2, 8, 0.8, 'OUTPUT\n• Optimized schedule for each dataset\n• Comparative performance tables & statistical results\n• Thesis BAB I-V & SINTA 2 journal paper',
         green, 'black', 7)
draw_arrow(5, 1.9, 5, 1.0)

# Arrow labels
ax.text(5.3, 11.0, 'Data & Parameters', fontsize=7, color='gray')
ax.text(5.3, 9.0, 'Research Foundation', fontsize=7, color='gray')
ax.text(5.3, 7.0, 'Algorithm Development', fontsize=7, color='gray')
ax.text(5.3, 5.3, 'Baseline Setup', fontsize=7, color='gray')
ax.text(5.3, 3.3, 'Data Collection', fontsize=7, color='gray')
ax.text(5.3, 1.5, 'Analysis & Writing', fontsize=7, color='gray')

# Legend
legend_y = 12.8
ax.text(1, legend_y, '▬ Input/Output', fontsize=8, color='green', fontweight='bold')
ax.text(3.5, legend_y, '▬ Research & Development', fontsize=8, color='blue', fontweight='bold')
ax.text(6.5, legend_y, '▬ Experimentation & Analysis', fontsize=8, color='orange', fontweight='bold')

plt.tight_layout()
plt.savefig('proposal/figures/research_flowchart.png', dpi=200, bbox_inches='tight')
print("Flowchart PNG generated successfully.")
