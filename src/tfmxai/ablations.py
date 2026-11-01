"""Compare each ablation against the main grid on matching (dataset, model, k) cells.

    python -m tfmxai.ablations --main results/main

Ablation result directories that exist are picked up automatically:
  results/ablation_ensemble1, results/ablation_ensemble16   n_estimators 1 / 16 vs 4
  results/ablation_version                                  TabPFN v2 vs TabPFN-2.5
  results/ablation_stratified                               stratified vs uniform contexts
  results/ablation_grouped                                  grouped vs per-feature Shapley
Writes <main>/analysis/ablations.csv, remedy.csv and ablations.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from tfmxai import metrics
from tfmxai.analysis import LABEL, _md, bagging_curve, load_cell
from tfmxai.summarize import collect

ABLATIONS = {"ensemble1": "n_estimators = 1", "ensemble16": "n_estimators = 16",
             "version": "TabPFN v2", "stratified": "stratified contexts",
             "grouped": "grouped Shapley (|rho| > 0.7)"}
METRICS = ["tau_local", "jaccard_top3", "sign_flip", "tau_global", "vr", "logloss_mean"]


def posthoc_grouped(main_root: Path, abl_root: Path, cells: pd.DataFrame) -> pd.DataFrame:
    """Main-grid per-feature phi summed within the ablation's groups: the cheap alternative
    to grouped Shapley (same players, no extra model evaluations)."""
    rows = []
    for r in cells.itertuples():
        f = next((abl_root / "raw" / r.dataset / r.model / f"k{r.k}").glob("c*_s0_e0.npz"))
        g = np.load(f)["groups"]
        base, _ = load_cell(main_root, r.dataset, r.model, r.k)
        summed = np.stack([base[..., g == j].sum(-1) for j in range(g.max() + 1)], -1)
        rows.append({"dataset": r.dataset, "model": r.model, "k": r.k,
                     "tau_local": metrics.local_kendall(summed),
                     "jaccard_top3": metrics.topk_jaccard(summed, 3),
                     "sign_flip": metrics.sign_flip_rate(summed),
                     "tau_global": metrics.global_kendall(summed)})
    return pd.DataFrame(rows)


def compare(main: pd.DataFrame, abl: pd.DataFrame, name: str, against: str | None = None):
    """Paired differences ablation - reference, per model and k, across datasets."""
    ref = main.copy()
    if against:  # e.g. TabPFN v2 is compared with TabPFN-2.5 in the main grid
        ref = ref[ref.model == against].assign(model=abl.model.iloc[0])
    m = abl.merge(ref, on=["dataset", "model", "k"], suffixes=("", "_ref"))
    rows = []
    for (model, k), g in m.groupby(["model", "k"]):
        for col in [c for c in METRICS if c in abl and f"{c}_ref" in g]:
            diff = (g[col] - g[f"{col}_ref"]).dropna().to_numpy()
            if not len(diff):
                continue
            p = wilcoxon(diff).pvalue if len(diff) >= 5 and np.any(diff != 0) else np.nan
            rows.append({"ablation": name, "model": model, "k": k, "metric": col,
                         "n_datasets": len(diff), "ref_median": float(g[f"{col}_ref"].median()),
                         "abl_median": float(g[col].median()),
                         "median_diff": float(np.median(diff)), "p": float(p)})
    return pd.DataFrame(rows)


def remedy(main_root: Path, main: pd.DataFrame, ens16: pd.DataFrame, rng) -> pd.DataFrame:
    """Same GPU budget, two ways to spend it: one context with a 4x larger internal ensemble
    (n_estimators 16) vs averaging attributions over 4 contexts (n_estimators 4). Stability
    is local tau between two independent estimates of each kind."""
    rows = []
    for r in ens16.itertuples():
        bag = bagging_curve(load_cell(main_root, r.dataset, r.model, r.k)[0], 50, rng)
        ref = main[(main.dataset == r.dataset) & (main.model == r.model) & (main.k == r.k)]
        rows.append({"dataset": r.dataset, "model": r.model, "k": r.k,
                     "single_ctx_nest4": float(ref.tau_local.iloc[0]),
                     "single_ctx_nest16": r.tau_local,
                     "bag4_ctx_nest4": float(bag.loc[bag.M == 4, "tau"].iloc[0]),
                     "sec_nest16": r.sec_per_run,
                     "sec_bag4": 4 * float(ref.sec_per_run.iloc[0])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/main")
    ap.add_argument("--summary", help="main-grid summary CSV (default: <main>/summary.csv)")
    ap.add_argument("--out", help="output directory (default: <main>/analysis)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main_root = Path(args.main)
    out = Path(args.out) if args.out else main_root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    main_sum = pd.read_csv(args.summary or main_root / "summary.csv")

    tables, lines = [], ["# Ablations", ""]
    for key, title in ABLATIONS.items():
        root = main_root.parent / f"ablation_{key}"
        if not (root / "raw").exists():
            lines += [f"## {title}", "", "Not run yet.", ""]
            continue
        abl = collect(root)
        abl.to_csv(root / "summary.csv", index=False)
        t = compare(main_sum, abl, key, against="tabpfn_v2_5" if key == "version" else None)
        if key == "grouped":
            post = posthoc_grouped(main_root, root, abl)
            post.to_csv(root / "posthoc_grouped.csv", index=False)
            t = pd.concat([t, compare(post, abl, "grouped_vs_posthoc_sum")])
        tables.append(t)
        show = t[t.metric.isin(["tau_local", "sign_flip", "vr"])]
        note = ("Reference: the main grid on the same cells. For grouping, τ over fewer players "
                "is not comparable with per-feature τ; `grouped_vs_posthoc_sum` compares like "
                "with like (main-grid φ summed within the same groups)."
                if key == "grouped" else "Reference: the main grid on the same cells.")
        lines += [f"## {title}", "", note, "", _md(show), ""]
        if key == "ensemble16":
            rem = remedy(main_root, main_sum, abl, rng)
            rem.to_csv(out / "remedy.csv", index=False)
            agg = rem.groupby(["model", "k"])[["single_ctx_nest4", "single_ctx_nest16",
                                               "bag4_ctx_nest4", "sec_nest16",
                                               "sec_bag4"]].median().reset_index()
            lines += ["### Remedy: bigger ensemble vs context bagging (median local τ)", "",
                      _md(agg), ""]
    if tables:
        pd.concat(tables).to_csv(out / "ablations.csv", index=False)
    (out / "ablations.md").write_text("\n".join(lines), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
