"""Aggregate raw runs into one row per (dataset, model, k).

    python -m tfmxai.summarize --results results
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from tfmxai import metrics
from tfmxai.values import EPS


def collect(root: Path) -> pd.DataFrame:
    groups = defaultdict(list)
    for f in (root / "raw").glob("*/*/k*/c*_s*_e*.npz"):
        ds, model, kdir = f.parts[-4], f.parts[-3], f.parts[-2]
        ctx, seed, eseed = (int(x[1:]) for x in f.stem.split("_"))
        groups[(ds, model, int(kdir[1:]))].append((ctx, seed, eseed, f))

    rows = []
    for (ds, model, k), items in sorted(groups.items()):
        ctxs = sorted({c for c, _, _, _ in items})
        seeds = sorted({s for _, s, _, _ in items})
        by_key = {(c, s): f for c, s, e, f in items if e == 0}
        complete = [c for c in ctxs if all((c, s) in by_key for s in seeds)]
        if len(complete) < 2:
            continue
        load = {key: np.load(f) for key, f in by_key.items()}
        phis_seeds = np.stack([[load[(c, s)]["phi"] for c in complete] for s in seeds])
        phis = phis_seeds[0]
        p_true = np.stack([1 / (1 + np.exp(-load[(c, seeds[0])]["v_full"])) for c in complete])
        row = {
            "dataset": ds, "model": model, "k": k, "n_contexts": len(complete),
            "n_seeds": len(seeds), "d": phis.shape[2],
            "tau_local": metrics.local_kendall(phis),
            "tau_global": metrics.global_kendall(phis),
            "jaccard_top3": metrics.topk_jaccard(phis, 3),
            "sign_flip": metrics.sign_flip_rate(phis),
            **metrics.variance_components(phis_seeds),
            "logloss_mean": float(-np.log(np.clip(p_true, EPS, 1)).mean()),
            "logloss_sd_ctx": float((-np.log(np.clip(p_true, EPS, 1))).mean(1).std(ddof=1)),
            "p_true_sd_point": float(p_true.std(0, ddof=1).mean()),
            "sec_per_run": float(np.mean([load[(c, seeds[0])]["seconds"] for c in complete])),
        }
        oracle = root / "raw" / ds / "oracle.npz"
        if oracle.exists():
            row.update(metrics.oracle_error(phis, np.load(oracle)["phi"]))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    root = Path(args.results)
    df = collect(root)
    df.to_csv(root / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print(df.drop(columns=["n_seeds"], errors="ignore").to_string(index=False))


if __name__ == "__main__":
    main()
