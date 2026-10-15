"""
Generate publication-quality diagram comparing Canonical GA workflow with HAGA modifications.
Visualizes:
1. Canonical GA pipeline backbone.
2. Three HAGA intervention points (Hamming diversity, linear APC, generational ELS).
3. Shared Evaluation Budget (B_max) tracking monitor ensuring computational parity.
Outputs: figures/haga_algorithm_flowchart.png (300 DPI)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, BoxStyle

# Set high DPI and font configurations
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

fig, ax = plt.subplots(figsize=(13, 17), dpi=300)
ax.set_xlim(0, 13)
ax.set_ylim(0, 17)
ax.axis('off')

# Color palette
C_CANONICAL = '#1E3A8A'    # Deep Blue (Canonical GA)
C_CANONICAL_BG = '#DBEAFE' # Light Blue Fill
C_HAGA = '#B45309'         # Deep Amber/Orange (HAGA Interventions)
C_HAGA_BG = '#FEF3C7'      # Amber Gold Fill
C_BUDGET = '#581C87'       # Deep Violet (Budget Monitor)
C_BUDGET_BG = '#F3E8FF'    # Soft Violet Fill
C_TERMINAL = '#065F46'     # Dark Emerald Green
C_TERMINAL_BG = '#D1FAE5' # Light Green
C_ARROW = '#374151'        # Dark Gray Arrow

def draw_node(x, y, w, h, title, subtitle="", category="canonical", badge=""):
    """Draw a styled process node with optional badge and subtitle."""
    if category == "canonical":
        edge, face = C_CANONICAL, C_CANONICAL_BG
    elif category == "haga":
        edge, face = C_HAGA, C_HAGA_BG
    elif category == "budget":
        edge, face = C_BUDGET, C_BUDGET_BG
    elif category == "terminal":
        edge, face = C_TERMINAL, C_TERMINAL_BG
    else:
        edge, face = '#4B5563', '#F3F4F6'

    # Box
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        edgecolor=edge, facecolor=face,
        linewidth=2.0 if category == "haga" else 1.5,
        linestyle='--' if category == "budget" else '-'
    )
    ax.add_patch(box)

    # Badge if provided
    if badge:
        b_box = FancyBboxPatch(
            (x + 0.15, y + h - 0.28), len(badge) * 0.16 + 0.25, 0.32,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            edgecolor=edge, facecolor=edge, linewidth=1
        )
        ax.add_patch(b_box)
        ax.text(x + 0.25, y + h - 0.12, badge, fontsize=7.5, fontweight='bold', color='white', va='center')

    # Title & Subtitle text positioning
    text_y = y + h / 2.0 + (0.15 if subtitle else 0.0)
    if badge:
        text_y -= 0.08
    ax.text(x + w / 2.0, text_y, title, ha='center', va='center',
            fontsize=9.2, fontweight='bold', color='#111827')

    if subtitle:
        ax.text(x + w / 2.0, text_y - 0.35, subtitle, ha='center', va='center',
                fontsize=7.8, style='italic', color='#4B5563')

def draw_arrow(x1, y1, x2, y2, label="", rad=0.0, color=C_ARROW, linestyle='-'):
    """Draw connecting arrow with optional text label."""
    connection = f"arc3,rad={rad}"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.6,
            linestyle=linestyle,
            connectionstyle=connection,
            mutation_scale=12
        )
    )
    if label:
        mid_x = (x1 + x2) / 2.0
        mid_y = (y1 + y2) / 2.0
        ax.text(mid_x + 0.15, mid_y, label, fontsize=7.5, fontweight='bold', color=color, va='center')

# ==============================================================================
# 1. HEADER & LEGEND
# ==============================================================================
ax.text(6.5, 16.5, "Diagram Alur Komparatif: Backbone Canonical GA dan Titik Intervensi HAGA",
        ha='center', va='center', fontsize=12.5, fontweight='bold', color='#111827')
ax.text(6.5, 16.15, "Perbandingan Siklus Generasi, Pengendalian Keragaman, Pencarian Lokal, dan Paritas Anggaran Evaluasi",
        ha='center', va='center', fontsize=9.0, style='italic', color='#4B5563')

# Legend items
ax.add_patch(FancyBboxPatch((1.0, 15.4), 3.2, 0.45, boxstyle="round,pad=0.05", facecolor=C_CANONICAL_BG, edgecolor=C_CANONICAL, lw=1.5))
ax.text(2.6, 15.62, "Alur Utama Canonical GA", ha='center', va='center', fontsize=8, fontweight='bold', color=C_CANONICAL)

ax.add_patch(FancyBboxPatch((4.7, 15.4), 3.6, 0.45, boxstyle="round,pad=0.05", facecolor=C_HAGA_BG, edgecolor=C_HAGA, lw=2.0))
ax.text(6.5, 15.62, "★ Titik Intervensi / Modifikasi HAGA", ha='center', va='center', fontsize=8, fontweight='bold', color=C_HAGA)

ax.add_patch(FancyBboxPatch((8.8, 15.4), 3.2, 0.45, boxstyle="round,pad=0.05", facecolor=C_BUDGET_BG, edgecolor=C_BUDGET, lw=1.5, linestyle='--'))
ax.text(10.4, 15.62, "Monitor Kuota Anggaran ($B_{max}$)", ha='center', va='center', fontsize=8, fontweight='bold', color=C_BUDGET)

# ==============================================================================
# 2. SHARED EVALUATION BUDGET MONITOR (Right Side Panel)
# ==============================================================================
draw_node(9.0, 8.5, 3.4, 4.5, "MONITOR ANGGARAN EVALUASI",
          category="budget", badge="PARITAS KOMPUTASI")

ax.text(10.7, 12.0, "Shared Budget Quota:\n$B_{max} = 25.000$ (Pra-Penelitian)\n$B_{max} = 50.000$ (Tesis Penuh)",
        ha='center', va='center', fontsize=8.0, fontweight='bold', color='#4C1D95')
ax.text(10.7, 10.6, "Sumber Konsumsi Anggaran:\n• Evaluasi Inisialisasi Pop (+N)\n• Evaluasi Offspring (+2 tiap xover)\n• Evaluasi Elite Local Search (+1 tiap step)",
        ha='center', va='center', fontsize=7.4, color='#374151')
ax.text(10.7, 9.2, "Aturan Penghentian:\nJika total evaluasi $\\geq B_{max}$,\nalgoritma berhenti seketika.",
        ha='center', va='center', fontsize=7.4, style='italic', color='#6B21A8')

# ==============================================================================
# 3. MAIN ALGORITHM PIPELINE (Center Column: x=1.5, w=6.8)
# ==============================================================================
W = 6.8
CX = 1.2

# Node 0: Start
draw_node(CX + 1.4, 14.4, 4.0, 0.65, "MULAI (START)",
          "Input instance CTT & parameter awal", category="terminal")

# Node 1: Initialization
draw_node(CX, 13.1, W, 0.85, "1. Inisialisasi Populasi Awal",
          "Representasi 1D Event-Indexed (N = 50 individu acak terdistribusi)", category="canonical")
draw_arrow(CX + W/2, 14.4, CX + W/2, 13.95)

# Node 2: Evaluation
draw_node(CX, 11.75, W, 0.90, "2. Evaluasi Fitness Populasi Awal",
          "Hitung HCV (Hard Constraints) & SCV (Soft Constraints); F(S) = 1000·HCV + SCV", category="canonical")
draw_arrow(CX + W/2, 13.1, CX + W/2, 12.65)
# Arrow to Budget Monitor
draw_arrow(CX + W, 12.2, 9.0, 12.2, label="+N Evals", color=C_BUDGET, linestyle='--')

# Node 3: HAGA Mod 1 (Diversity Measurement)
draw_node(CX, 10.2, W, 1.15, "3. Pengukuran Keragaman Genotipik Populasi",
          "Kalkulasi Normalized Pairwise Hamming Distance H_norm di seluruh pasangan individu",
          category="haga", badge="★ TITIK MODIFIKASI 1 (HAGA)")
draw_arrow(CX + W/2, 11.75, CX + W/2, 11.35)

# Note for Canonical GA bypass on Mod 1 & 2
ax.text(CX - 0.1, 10.15, "Canonical GA:\nMelewati tahap 3 & 4;\nparameter Pc & Pm statis",
        ha='right', va='center', fontsize=7.2, fontweight='bold', color=C_CANONICAL, style='italic')

# Node 4: HAGA Mod 2 (Linear APC)
draw_node(CX, 8.6, W, 1.15, "4. Kontrol Parameter Adaptif Linier (APC)",
          "Penyesuaian tertutup Pc & Pm secara proporsional terhadap H_norm dengan batas klem",
          category="haga", badge="★ TITIK MODIFIKASI 2 (HAGA)")
draw_arrow(CX + W/2, 10.2, CX + W/2, 9.75)

# Node 5: Tournament Selection & Elitism
draw_node(CX, 7.25, W, 0.90, "5. Seleksi Turnamen & Elitisme",
          "Tournament Selection (size k = 3) memilih induk; Elitisme (e = 2) mempertahankan solusi terbaik",
          category="canonical")
draw_arrow(CX + W/2, 8.6, CX + W/2, 8.15)

# Node 6: Crossover & Mutation
draw_node(CX, 5.85, W, 0.95, "6. Rekombinasi Genetik (Crossover & Mutasi)",
          "Two-Point Crossover (peluang Pc) & Slot Swap/Reassignment Mutation (peluang Pm)",
          category="canonical")
draw_arrow(CX + W/2, 7.25, CX + W/2, 6.80)
# Arrow to Budget Monitor from Offspring evaluation
draw_arrow(CX + W, 6.32, 9.0, 8.8, label="+Offspring Evals", color=C_BUDGET, rad=-0.2, linestyle='--')

# Node 7: HAGA Mod 3 (ELS)
draw_node(CX, 4.15, W, 1.25, "7. Elite Local Search (ELS) Generasional",
          "Stochastic First-Improvement Hill Climbing pada 10% individu elit (target jadwal bentrok)",
          category="haga", badge="★ TITIK MODIFIKASI 3 (HAGA)")
draw_arrow(CX + W/2, 5.85, CX + W/2, 5.40)
# Arrow to Budget Monitor from ELS
draw_arrow(CX + W, 4.8, 9.0, 8.5, label="+ELS Steps Evals", color=C_HAGA, rad=-0.25, linestyle='--')

# Note for Canonical GA bypass on Mod 3
ax.text(CX - 0.1, 4.75, "Canonical GA:\nTidak memiliki ELS;\noffspring langsung masuk pop",
        ha='right', va='center', fontsize=7.2, fontweight='bold', color=C_CANONICAL, style='italic')

# Node 8: Termination Check (Diamond-like Box)
draw_node(CX + 0.6, 2.5, W - 1.2, 1.15, "8. Kriteria Penghentian (Stopping Criteria)",
          "Apakah Total Evaluasi $\\geq B_{max}$ ATAU HCV=0 dan konvergen?", category="canonical")

# Loop back arrow (BELUM TERCAPAI)
# Arrow leaves left of node 8, goes up along left side, enters node 3
ax.annotate(
    "", xy=(CX, 10.75), xytext=(CX + 0.6, 3.1),
    arrowprops=dict(
        arrowstyle="-|>",
        color='#DC2626',
        lw=1.8,
        linestyle='-',
        connectionstyle="arc3,rad=-0.4",
        mutation_scale=12
    )
)
ax.text(CX - 0.6, 7.5, "BELUM TERPENUHI\n(Generasi Berikutnya)",
        ha='center', va='center', fontsize=7.8, fontweight='bold', color='#DC2626', rotation=90)

# Final Node: Success Output
draw_node(CX + 1.2, 0.7, 4.4, 0.95, "JADWAL OPTIMAL SELESAI",
          "Solusi Layak (HCV = 0) dengan Pelanggaran Preferensi (SCV) Minimum",
          category="terminal")
draw_arrow(CX + W/2, 2.5, CX + W/2, 1.65, label="TERPENUHI", color=C_TERMINAL)

# Bottom Note
ax.text(6.5, 0.25, "* HAGA secara ketat mematuhi paritas komputasi: setiap evaluasi pada tahap 2, 6, dan 7 mengurangi kuota anggaran bersama B_max.",
        ha='center', va='center', fontsize=7.8, style='italic', color='#4B5563')

plt.tight_layout()
output_path = os.path.join("figures", "haga_algorithm_flowchart.png")
os.makedirs("figures", exist_ok=True)
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Flowchart successfully generated at {output_path} (300 DPI)!")
