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


def replicates(runs: dict, load, ctxs: list[int], pairs: list[tuple[int, int]]):
    """[R, C, n, d] phis for the contexts that have every (seed, eseed) pair, or None."""
    reps = [c for c in ctxs if all((c, s, e) in runs for s, e in pairs)]
    if len(pairs) < 2 or not reps:
        return None
    return np.stack([[load(c, s, e)["phi"] for c in reps] for s, e in pairs])


def collect(root: Path) -> pd.DataFrame:
    groups = defaultdict(list)
    for f in (root / "raw").glob("*/*/k*/c*_s*_e*.npz"):
        ds, model, kdir = f.parts[-4], f.parts[-3], f.parts[-2]
        ctx, seed, eseed = (int(x[1:]) for x in f.stem.split("_"))
        groups[(ds, model, int(kdir[1:]))].append((ctx, seed, eseed, f))

    rows = []
    for (ds, model, k), items in sorted(groups.items()):
        runs = {(c, s, e): f for c, s, e, f in items}
        seeds = sorted({s for _, s, _ in runs})
        eseeds = sorted({e for _, _, e in runs})
        s0, e0 = seeds[0], eseeds[0]
        ctxs = sorted(c for c, s, e in runs if (s, e) == (s0, e0))
        if len(ctxs) < 2:
            continue
        load = lambda c, s, e: np.load(runs[(c, s, e)])  # noqa: E731
        base = [load(c, s0, e0) for c in ctxs]
        phis = np.stack([r["phi"] for r in base])
        p_true = 1 / (1 + np.exp(-np.stack([r["v_full"] for r in base])))
        row = {
            "dataset": ds, "model": model, "k": k, "n_contexts": len(ctxs), "d": phis.shape[2],
            "tau_local": metrics.local_kendall(phis),
            "tau_global": metrics.global_kendall(phis),
            "jaccard_top3": metrics.topk_jaccard(phis, 3),
            "sign_flip": metrics.sign_flip_rate(phis),
            **metrics.variance_components(phis, replicates(runs, load, ctxs, [(s, e0) for s in seeds]),
                                          replicates(runs, load, ctxs, [(s0, e) for e in eseeds])),
            "logloss_mean": float(-np.log(np.clip(p_true, EPS, 1)).mean()),
            "logloss_sd_ctx": float((-np.log(np.clip(p_true, EPS, 1))).mean(1).std(ddof=1)),
            "p_true_sd_point": float(p_true.std(0, ddof=1).mean()),
            "sec_per_run": float(np.mean([float(r["seconds"]) for r in base])),
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
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
