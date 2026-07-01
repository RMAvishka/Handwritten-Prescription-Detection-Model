import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CFG_train = Path("../data/pharmacy_lk/splits/train.csv")
LABEL_COL = "medicine_name"

df = pd.read_csv(CFG_train)
counts = df[LABEL_COL].astype(str).str.strip().str.lower().value_counts()

singletons = (counts == 1).sum()
total_unique = len(counts)
print(f"unique names: {total_unique} | singletons: {singletons} ({singletons/total_unique:.1%})")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) rank-frequency (long tail)
axes[0].plot(range(1, len(counts)+1), counts.values, color="#2b2b2b")
axes[0].set_xlabel("Medicine name rank (most to least frequent)")
axes[0].set_ylabel("Occurrences in training set")
axes[0].set_title("(a) Long-tailed frequency distribution")
axes[0].spines[["top","right"]].set_visible(False)

# (b) histogram of how many names occur k times (capped for readability)
cap = 10
binned = counts.clip(upper=cap).value_counts().sort_index()
axes[1].bar(binned.index.astype(str), binned.values, color="#6f6f6f", edgecolor="black")
axes[1].set_xlabel(f"Occurrences (capped at {cap}+)")
axes[1].set_ylabel("Number of unique names")
axes[1].set_title(f"(b) {singletons/total_unique:.0%} of names appear only once")
axes[1].spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig("../reports/figures/htr_label_frequency.png", dpi=300, bbox_inches="tight")
plt.show()
print("saved -> reports/figures/htr_label_frequency.png")
