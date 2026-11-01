"""Synthetic ground truth: are context-specific attributions noisy estimates of a stable truth,
and do context-bagged confidence intervals cover it? (docs/study-design.md, §7)

    python -m tfmxai.oracle_analysis --results results/synthetic

The oracle Shapley values use the true conditional value function logit P(y | x_S), which is
what every refit explainer estimates, so all error below is estimation error. Writes
<results>/analysis/{oracle.csv, oracle.md, fig_oracle.png}.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from tfmxai import metrics
from tfmxai.analysis import COLOR, LABEL, _md, _models, load_cell

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LEVEL = 0.9


def bias_variance(phis: np.ndarray, truth: np.ndarray) -> dict:
    """Split the mean squared error of single-context attributions into squared bias of the
    across-context mean and across-context variance (MSE = bias2 + (M-1)/M var)."""
    bias2 = float(((phis.mean(0) - truth) ** 2).mean())
    var = float(phis.var(0, ddof=1).mean())
    return {"mse": float(((phis - truth) ** 2).mean()), "bias2": bias2, "var": var,
            "var_share": var / (bias2 + var) if bias2 + var > 0 else np.nan}


def coverage(phis: np.ndarray, truth: np.ndarray, n_boot: int, rng, level: float = LEVEL) -> dict:
    """Coverage of the oracle by (a) the percentile-bootstrap CI of the context-bagged mean
    (resampling contexts) and (b) the central range of the single-context attributions.
    Only (point, feature) pairs whose oracle |phi| is at least 5% of the point's total count."""
    M = len(phis)
    a = (1 - level) / 2
    boot = phis[rng.integers(0, M, (n_boot, M))].mean(1)          # [B, n, d]
    lo, hi = np.quantile(boot, [a, 1 - a], axis=0)
    s_lo, s_hi = np.quantile(phis, [a, 1 - a], axis=0)
    share = np.abs(truth) / np.maximum(np.abs(truth).sum(1, keepdims=True), 1e-12)
    rel = share >= 0.05
    inside = lambda l, h: float(((truth >= l) & (truth <= h))[rel].mean())  # noqa: E731
    return {"cover_bagged_ci": inside(lo, hi), "cover_single_range": inside(s_lo, s_hi),
            "ci_width": float((hi - lo)[rel].mean())}


def oracle_tau(phis: np.ndarray, truth: np.ndarray) -> dict:
    """Local Kendall tau of |phi| against the oracle: one context vs the mean of all contexts."""
    single = np.nanmean([metrics.kendall_rows(np.abs(p), np.abs(truth)) for p in phis])
    bagged = np.nanmean(metrics.kendall_rows(np.abs(phis.mean(0)), np.abs(truth)))
    return {"oracle_tau_single": float(single), "oracle_tau_bagged": float(bagged)}


def collect(root: Path, n_boot: int, rng) -> pd.DataFrame:
    rows = []
    for ds_dir in sorted((root / "raw").iterdir()):
        oracle = ds_dir / "oracle.npz"
        m = re.fullmatch(r"syn-d(\d+)-rho([\d.]+)", ds_dir.name)
        if not oracle.exists() or not m:
            continue
        truth = np.load(oracle)["phi"]
        for model_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            for k_dir in sorted(model_dir.glob("k*"), key=lambda p: int(p.name[1:])):
                if len(list(k_dir.glob("c*_s0_e0.npz"))) < 2:
                    continue
                phis, _ = load_cell(root, ds_dir.name, model_dir.name, int(k_dir.name[1:]))
                rows.append({"dataset": ds_dir.name, "d": int(m[1]), "rho": float(m[2]),
                             "model": model_dir.name, "k": int(k_dir.name[1:]),
                             "n_contexts": len(phis),
                             "tau_local": metrics.local_kendall(phis),
                             **oracle_tau(phis, truth), **bias_variance(phis, truth),
                             **coverage(phis, truth, n_boot, rng)})
    return pd.DataFrame(rows)


def fig_oracle(df: pd.DataFrame, out: Path) -> None:
    ds = sorted(df.d.unique())
    panels = [("oracle_tau_single", "agreement with oracle, one context (τ)"),
              ("tau_local", "stability across contexts (τ)"),
              ("var_share", "share of error from context variance")]
    fig, axes = plt.subplots(len(ds), len(panels), figsize=(12, 3.3 * len(ds)), squeeze=False)
    for i, d in enumerate(ds):
        for j, (col, title) in enumerate(panels):
            ax = axes[i, j]
            for m in _models(df):
                for k, ls in zip(sorted(df.k.unique()), ["--", "-"]):
                    g = df[(df.d == d) & (df.model == m) & (df.k == k)].sort_values("rho")
                    ax.plot(g.rho, g[col], ls, marker="o", ms=3.5, color=COLOR[m],
                            label=f"{LABEL[m]}, k={k}" if (i, j) == (0, 0) else None)
            ax.set_title(f"d = {d}: {title}", fontsize=9)
            ax.set_xlabel("feature correlation ρ")
    fig.legend(loc="upper center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/synthetic")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.results)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df = collect(root, args.n_boot, np.random.default_rng(args.seed))
    df.to_csv(out / "oracle.csv", index=False)
    fig_oracle(df, out / "fig_oracle.png")
    cols = ["d", "rho", "model", "k", "n_contexts", "tau_local", "oracle_tau_single",
            "oracle_tau_bagged", "var_share", "cover_bagged_ci", "cover_single_range"]
    text = "\n".join(["# Synthetic ground truth", "",
                      f"Coverage is for a nominal {LEVEL:.0%} interval, on (point, feature) pairs "
                      "carrying at least 5% of the point's oracle |φ|.", "",
                      _md(df[cols].sort_values(["d", "rho", "model", "k"])), ""])
    (out / "oracle.md").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
