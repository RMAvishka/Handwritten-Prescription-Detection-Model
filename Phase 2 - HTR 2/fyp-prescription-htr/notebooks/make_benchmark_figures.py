import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path("../reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi":120,"savefig.dpi":300,"font.size":11})
INK=["#2b2b2b","#6f6f6f","#a3a3a3"]; BLUE="#1a4f8a"; RED="#8a1a1a"

# ---------------------------------------------------------------
# FIGURE A — BD BENCHMARK (closed-vocabulary public dataset)
# Two panels: EM and CER side by side, three models.
# ---------------------------------------------------------------
models = ["CRNN\nbaseline", "CRNN +\nlexicon", "Domain Aware TrOCR"]
bd_em  = [0.636, 0.845, 0.941]
bd_cer = [0.150, 0.112, 0.046]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))

bars1 = a1.bar(models, bd_em, color=[INK[2], INK[1], BLUE], edgecolor="black")
for b, v in zip(bars1, bd_em):
    a1.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.3f}", ha="center", fontsize=10, weight="bold")
a1.set_ylabel("Exact-Match Accuracy"); a1.set_ylim(0, 1.05)
a1.set_title("(a) Exact-match accuracy")
a1.spines[["top","right"]].set_visible(False)

bars2 = a2.bar(models, bd_cer, color=[INK[2], INK[1], BLUE], edgecolor="black")
for b, v in zip(bars2, bd_cer):
    a2.text(b.get_x()+b.get_width()/2, v+0.004, f"{v:.3f}", ha="center", fontsize=10, weight="bold")
a2.set_ylabel("Character Error Rate (lower = better)"); a2.set_ylim(0, 0.18)
a2.set_title("(b) Character error rate")
a2.spines[["top","right"]].set_visible(False)

fig.suptitle("BD benchmark (public closed-vocabulary prescription dataset)", fontsize=12, weight="bold")
plt.tight_layout(); plt.savefig(FIG/"bench_bd.png"); plt.show()
print("saved bench_bd.png")

# ---------------------------------------------------------------
# FIGURE B — IAM DOMAIN-GAP (same CRNN, three datasets, by CER)
# The key framing: same architecture, prescriptions are harder.
# ---------------------------------------------------------------
datasets = ["IAM\n(general English)", "BD\n(closed-vocab Rx)", "Custom\n(open-vocab Rx)"]
cer_vals = [0.189, 0.150, 0.481]
colors   = [INK[1], INK[1], RED]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(datasets, cer_vals, color=colors, edgecolor="black", width=0.6)
for b, v in zip(bars, cer_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=11, weight="bold")
ax.set_ylabel("Character Error Rate (lower = better)")
ax.set_ylim(0, 0.55)
ax.set_title("Domain gap: the same CRNN architecture across three datasets\n"
             "open-vocabulary prescriptions are markedly harder than general handwriting")
ax.spines[["top","right"]].set_visible(False)
# annotation arrow highlighting the gap
ax.annotate("", xy=(2, 0.481), xytext=(0, 0.189),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, ls="--"))
ax.text(1.0, 0.34, "~2.5x harder", ha="center", fontsize=10, color="gray", style="italic")
plt.tight_layout(); plt.savefig(FIG/"bench_iam_domaingap.png"); plt.show()
print("saved bench_iam_domaingap.png")
