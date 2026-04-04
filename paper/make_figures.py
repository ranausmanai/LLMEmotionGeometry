"""
Generate all figures for the arXiv paper.
Outputs: figures/fig1_behavior.pdf
         figures/fig2_layers.pdf
         figures/fig3_emotion_map.pdf
         figures/fig4_cosine.pdf
"""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

BASE = Path(__file__).parent.parent / "results"

# ── color palette ──────────────────────────────────────────────────────────
NEG  = "#e05c5c"   # negative / high-pressure conditions
POS  = "#5c9ee0"   # positive / exploratory
NEUT = "#a0a0a0"   # neutral / calm
CMAP_HEAT = "RdBu_r"

COND_COLOR = {
    "pressure":    "#d63031",
    "urgency":     "#e17055",
    "threat":      "#b2184a",
    "approval":    "#e07a5f",
    "shame":       "#6c757d",
    "curiosity":   "#0077b6",
    "encouragement": "#2ecc71",
    "calm":        "#636e72",
}

COND_ORDER = ["pressure","urgency","threat","approval","shame","encouragement","curiosity","calm"]

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── load data ──────────────────────────────────────────────────────────────
with open(BASE / "benchmark_emotion_analysis_0.8b.json") as f:
    beh = json.load(f)["summary"]["conditions"]

with open(BASE / "emotion_map_0.8b.json") as f:
    emap = json.load(f)

with open(BASE / "emotion_vector_summary_0.8b.json") as f:
    vsummary = json.load(f)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 1 — Behavioral results: hack rate & honest rate per condition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)

conds_plot = [c for c in COND_ORDER if c != "calm"] + ["calm"]
hack_rates  = [beh[c]["hack_rate"]  for c in conds_plot]
honest_rates= [beh[c]["honest_rate"] for c in conds_plot]
colors_plot = [COND_COLOR[c] for c in conds_plot]
labels_nice = [c.capitalize() for c in conds_plot]

x = np.arange(len(conds_plot))

ax = axes[0]
bars = ax.barh(x, hack_rates, color=colors_plot, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Hack Rate", fontsize=9)
ax.set_title("Shortcut-taking Rate", fontsize=10, fontweight="bold")
ax.set_yticks(x)
ax.set_yticklabels(labels_nice, fontsize=8.5)
ax.set_xlim(0, 0.75)
ax.axvline(0, color="#333", linewidth=0.5)
for i, v in enumerate(hack_rates):
    if v > 0:
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=7.5, color="#333")

ax = axes[1]
ax.barh(x, honest_rates, color=colors_plot, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Honest Rate", fontsize=9)
ax.set_title("Explicit Honesty Rate", fontsize=10, fontweight="bold")
ax.set_xlim(0, 0.75)
ax.axvline(0, color="#333", linewidth=0.5)
for i, v in enumerate(honest_rates):
    if v > 0:
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=7.5, color="#333")

fig.suptitle("Behavioral Results by Emotional Condition (Qwen 3.5 0.8B, n=20 per condition)",
             fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "fig1_behavior.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig1_behavior.png", bbox_inches="tight", dpi=150)
print("Saved fig1_behavior")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 2 — Layer-wise separation scores for all conditions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# vsummary has per-layer norms per condition
# structure: vsummary["conditions"][cond]["layer_norms"] = [float x 24]
fig, ax = plt.subplots(figsize=(7, 3.8))

layers = list(range(24))
highlight_conds = ["pressure", "urgency", "curiosity", "shame"]

for cond in COND_ORDER:
    if cond == "calm":
        continue
    layer_data = vsummary.get("layers_per_condition", {}).get(cond, None)
    if layer_data is None:
        continue
    norms = [entry["separation"] for entry in layer_data]
    lw  = 2.0 if cond in highlight_conds else 1.0
    alp = 1.0 if cond in highlight_conds else 0.45
    ls  = "-"
    ax.plot(layers, norms, color=COND_COLOR[cond], lw=lw, alpha=alp,
            label=cond.capitalize(), linestyle=ls)

ax.set_xlabel("Transformer Layer", fontsize=9)
ax.set_ylabel("Separation Score\n(||condition − calm|| in hidden space)", fontsize=8.5)
ax.set_title("Layer-wise Activation Separation from Calm Baseline\n(Qwen 3.5 0.8B)",
             fontsize=10, fontweight="bold")
ax.set_xticks(layers[::2])
ax.axvline(22, color="#bbb", lw=0.8, linestyle="--")
ax.annotate("Layer 22–23\npeak", xy=(22.1, ax.get_ylim()[1]*0.92),
            fontsize=7.5, color="#666")
ax.legend(fontsize=7.5, ncol=2, framealpha=0.7, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig2_layers.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig2_layers.png", bbox_inches="tight", dpi=150)
print("Saved fig2_layers")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 3 — 2D Emotion Map (PCA)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, ax = plt.subplots(figsize=(6, 5.2))

coords = emap["pca_coords"]
ev     = emap["explained_variance"]

# draw light axes
ax.axhline(0, color="#ccc", lw=0.8, zorder=0)
ax.axvline(0, color="#ccc", lw=0.8, zorder=0)

# valence arrow annotation
ax.annotate("", xy=(1.15, 0), xytext=(-0.65, 0),
            arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.2))
ax.text(1.15, 0.06, "Positive\nValence", fontsize=7, color="#888", ha="right")
ax.text(-0.65, 0.06, "Negative\nValence", fontsize=7, color="#888", ha="left")

offsets = {
    "approval":     (-0.17, -0.11),
    "curiosity":    ( 0.04,  0.06),
    "encouragement":( 0.04,  0.06),
    "pressure":     ( 0.04,  0.06),
    "shame":        ( 0.04,  0.06),
    "threat":       ( 0.04, -0.10),
    "urgency":      ( 0.09, -0.02),
    "calm":         ( 0.04,  0.06),
}

for cond, (px, py) in coords.items():
    c = COND_COLOR[cond]
    sz = 120 if cond != "calm" else 80
    marker = "o" if cond != "calm" else "D"
    ax.scatter(px, py, s=sz, color=c, zorder=3, edgecolors="white", linewidths=0.8,
               marker=marker)
    dx, dy = offsets.get(cond, (0.04, 0.06))
    ax.annotate(cond.capitalize(), (px, py), xytext=(px+dx, py+dy),
                fontsize=8, color=c, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=c, lw=0.5) if abs(dx)>0.03 or abs(dy)>0.05 else None)

ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% variance) — Valence axis", fontsize=9)
ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% variance)", fontsize=9)
ax.set_title("2D Emotion Map: PCA of Condition Direction Vectors\n(Qwen 3.5 0.8B, Layer 23)",
             fontsize=10, fontweight="bold")
ax.set_xlim(-0.58, 1.10)
ax.set_ylim(-0.48, 0.52)

# add PC1 explained variance box
props = dict(boxstyle="round,pad=0.3", facecolor="#f0f4f8", alpha=0.8, edgecolor="#ccc")
ax.text(0.97, 0.97,
        f"PC1: {ev[0]*100:.1f}%\nPC2: {ev[1]*100:.1f}%\nValence align: 0.951",
        transform=ax.transAxes, fontsize=8, va="top", ha="right", bbox=props)

fig.tight_layout()
fig.savefig(OUT / "fig3_emotion_map.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig3_emotion_map.png", bbox_inches="tight", dpi=150)
print("Saved fig3_emotion_map")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIG 4 — Cosine similarity heatmap between all condition vectors
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
csm = emap["cosine_similarity_matrix"]
conds_heat = [c for c in COND_ORDER if c != "calm"]

matrix = np.array([[csm[r][c] for c in conds_heat] for r in conds_heat])

fig, ax = plt.subplots(figsize=(5.5, 4.8))
im = ax.imshow(matrix, cmap=CMAP_HEAT, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.82, label="Cosine Similarity")

tick_labels = [c.capitalize() for c in conds_heat]
ax.set_xticks(range(len(conds_heat)))
ax.set_yticks(range(len(conds_heat)))
ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=8.5)
ax.set_yticklabels(tick_labels, fontsize=8.5)

for i in range(len(conds_heat)):
    for j in range(len(conds_heat)):
        val = matrix[i, j]
        color = "white" if abs(val) > 0.6 else "#222"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

ax.set_title("Pairwise Cosine Similarity of Condition Direction Vectors\n(Layer 23 unit vectors)",
             fontsize=9.5, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig4_cosine.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig4_cosine.png", bbox_inches="tight", dpi=150)
print("Saved fig4_cosine")

print("\nAll figures saved to", OUT)
