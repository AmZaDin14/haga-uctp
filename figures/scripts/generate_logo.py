"""Create placeholder university logo for proposal."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(2, 2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Simple crest-like shape
circle = plt.Circle((0.5, 0.5), 0.4, fill=False, edgecolor='#003366', linewidth=3)
ax.add_patch(circle)

# Inner circle
inner = plt.Circle((0.5, 0.5), 0.25, fill=False, edgecolor='#003366', linewidth=1.5)
ax.add_patch(inner)

# Text
ax.text(0.5, 0.5, 'UTI', ha='center', va='center', fontsize=16, fontweight='bold', color='#003366')
ax.text(0.5, 0.12, 'Universitas Teknokrat Indonesia', ha='center', va='center',
        fontsize=4, color='#003366', fontweight='bold')

plt.tight_layout()
plt.savefig('proposal/figures/logo_ut.png', dpi=150, bbox_inches='tight')
print("Logo placeholder generated.")
