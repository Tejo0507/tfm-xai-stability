"""Compare the runs of configs/repro_check.yaml with the stored main-grid runs.

    python scripts/compare_repro.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from tfmxai.metrics import kendall_rows

REPRO, MAIN = Path("results/repro_check/raw"), Path("results/main/raw")

rows = []
for f in sorted(REPRO.glob("*/*/k*/c*.npz")):
    rel = f.relative_to(REPRO)
    new, old = np.load(f), np.load(MAIN / rel)
    rows.append({"dataset": rel.parts[0], "model": rel.parts[1], "k": rel.parts[2],
                 "same_context": np.array_equal(new["ctx_idx"], old["ctx_idx"]),
                 "bit_identical": np.array_equal(new["phi"], old["phi"]),
                 "max_abs_diff": float(np.abs(new["phi"] - old["phi"]).max()),
                 "tau": float(np.nanmean(kendall_rows(np.abs(new["phi"]), np.abs(old["phi"]))))})
table = pd.DataFrame(rows)
table.to_csv(REPRO.parent / "comparison.csv", index=False)
print(table.to_string(index=False))
print(f"\n{len(table)} runs, {int(table.bit_identical.sum())} bit-identical, "
      f"largest difference {table.max_abs_diff.max():.2e}, lowest tau {table.tau.min():.6f}")
