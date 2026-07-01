import pandas as pd
from pathlib import Path

LOG = Path("../reports/tables/cleaning_log.csv")   # adjust path if yours differs
# Expected columns (adapt to your actual log): raw_label, cleaned_label, reason

if LOG.exists():
    df = pd.read_csv(LOG)
    # try to pick a few illustrative rows: one per cleaning reason if available
    cols = [c for c in df.columns]
    print("columns in your log:", cols)
    # show a handful of examples where the label actually changed
    if {"raw_label","cleaned_label"}.issubset(df.columns):
        changed = df[df["raw_label"].astype(str) != df["cleaned_label"].astype(str)]
        sample = changed.head(8)
        print("\nExample before -> after rows:")
        print(sample.to_string(index=False))
        sample.to_csv("../reports/tables/cleaning_examples.csv", index=False)
        print("\nsaved -> reports/tables/cleaning_examples.csv")
else:
    print(f"cleaning_log.csv not found at {LOG}")
    print("Edit the LOG path to point at your actual cleaning log, then re-run.")
