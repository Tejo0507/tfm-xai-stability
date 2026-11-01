"""Hypothesis tests H1-H4 and figures for a finished grid (docs/study-design.md, §2 and §8).

    python -m tfmxai.analysis --results results/main --config configs/main.yaml

Reads the raw runs and `summary.csv` (run `tfmxai.summarize` first) and writes
<results>/analysis/: one CSV per hypothesis, report.md and fig_*.png. The dataset is the
unit of analysis throughout; explained points are averaged within a dataset.
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from tfmxai.data import load_dataset
from tfmxai.metrics import kendall_rows
from tfmxai.run import load_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

TFMS = ["tabpfn_v2_5", "tabicl_v2"]
BASELINES = ["xgboost", "logreg"]
LABEL = {"tabpfn_v2_5": "TabPFN-2.5", "tabicl_v2": "TabICLv2", "xgboost": "XGBoost",
         "logreg": "Logistic reg.", "tabpfn_v2": "TabPFN v2"}
COLOR = {"tabpfn_v2_5": "#2f6db5", "tabicl_v2": "#d9822b", "xgboost": "#3a9a5b",
         "logreg": "#7a7a7a", "tabpfn_v2": "#8fb4e3"}
TAU_TARGET = 0.9


# ---------------------------------------------------------------- raw runs

def load_cell(root: Path, ds: str, model: str, k: int):
    """Base phis [M, n, d] (seed 0, explainer seed 0) and model-seed replicates
    [R, C, n, d] on the contexts that have every seed (None if there are none)."""
    cell = root / "raw" / ds / model / f"k{k}"
    runs = {tuple(int(x[1:]) for x in f.stem.split("_")): f for f in cell.glob("c*_s*_e*.npz")}
    ctxs = sorted(c for c, s, e in runs if (s, e) == (0, 0))
    base = np.stack([np.load(runs[(c, 0, 0)])["phi"] for c in ctxs])
    seeds = sorted({s for _, s, e in runs if e == 0})
    rep_ctxs = [c for c in ctxs if all((c, s, 0) in runs for s in seeds)]
    reps = (np.stack([[np.load(runs[(c, s, 0)])["phi"] for c in rep_ctxs] for s in seeds])
            if len(seeds) > 1 and rep_ctxs else None)
    return base, reps


# ---------------------------------------------------------------- dataset traits

def dataset_traits(cfg: dict) -> pd.DataFrame:
    """Per dataset: d, classes, mean |Spearman| feature correlation and class imbalance
    (1 - normalised class entropy) on the context pool."""
    rows = []
    for spec in cfg["datasets"]:
        ds = load_dataset(spec, n_explain=cfg["n_explain"])
        corr = pd.DataFrame(ds.X_pool).corr(method="spearman").abs().to_numpy()
        off = corr[~np.eye(ds.d, dtype=bool)]
        freq = np.bincount(ds.y_pool) / len(ds.y_pool)
        freq = freq[freq > 0]
        entropy = -(freq * np.log(freq)).sum() / np.log(len(freq))
        rows.append({"dataset": ds.name, "d": ds.d, "n_classes": len(freq),
                     "n_pool": len(ds.y_pool), "mean_abs_corr": float(np.nanmean(off)),
                     "imbalance": float(1 - entropy)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H1

def vr_bootstrap(base: np.ndarray, reps: np.ndarray, n_boot: int, rng) -> tuple:
    """VR = context variance / model-seed variance with a percentile CI from resampling
    explained points."""
    vc = base.var(0, ddof=1).mean(-1)            # [n]
    vs = reps.var(0, ddof=1).mean((0, -1))       # [n]
    idx = rng.integers(0, len(vc), (n_boot, len(vc)))
    boot = vc[idx].mean(1) / vs[idx].mean(1)
    return vc.mean() / vs.mean(), *np.percentile(boot, [2.5, 97.5])


def h1(root: Path, summary: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    rows = []
    for r in summary.itertuples():
        if r.model == "logreg":
            continue  # deterministic: no seed variance
        base, reps = load_cell(root, r.dataset, r.model, r.k)
        if reps is None:
            continue
        vr, lo, hi = vr_bootstrap(base, reps, n_boot, rng)
        rows.append({"dataset": r.dataset, "model": r.model, "k": r.k, "vr": vr,
                     "vr_lo": lo, "vr_hi": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H2

def paired_tests(summary: pd.DataFrame, metric: str, n_boot: int, rng) -> pd.DataFrame:
    """Each TFM against each baseline at matched k: Wilcoxon signed-rank across datasets,
    Holm-corrected over k within each (TFM, baseline) pair."""
    piv = summary.pivot_table(index=["dataset", "k"], columns="model", values=metric)
    rows = []
    for tfm in [m for m in TFMS if m in piv]:
        for base in [m for m in BASELINES if m in piv]:
            block = []
            for k in sorted(piv.index.get_level_values("k").unique()):
                diff = (piv.xs(k, level="k")[tfm] - piv.xs(k, level="k")[base]).dropna().to_numpy()
                if len(diff) < 5:
                    continue
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                ranks = pd.Series(np.abs(diff)).rank().to_numpy()
                r_rb = (ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum()
                boot = np.median(diff[rng.integers(0, len(diff), (n_boot, len(diff)))], 1)
                block.append({"metric": metric, "tfm": tfm, "baseline": base, "k": k,
                              "n_datasets": len(diff), "median_diff": float(np.median(diff)),
                              "ci_lo": float(np.percentile(boot, 2.5)),
                              "ci_hi": float(np.percentile(boot, 97.5)),
                              "wins": int((diff > 0).sum()), "rank_biserial": float(r_rb),
                              "p": float(p)})
            if block:
                p_holm = multipletests([b["p"] for b in block], method="holm")[1]
                for b, ph in zip(block, p_holm):
                    rows.append({**b, "p_holm": float(ph)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H3

def drivers(summary: pd.DataFrame, traits: pd.DataFrame):
    """Mixed model tau_local ~ log2 k + d + corr + imbalance + model + (1 | dataset).
    Continuous dataset traits are standardised so their coefficients are comparable."""
    import statsmodels.formula.api as smf

    df = summary.drop(columns=["d"]).merge(traits, on="dataset")
    for c in ["d", "mean_abs_corr", "imbalance"]:
        df[f"z_{c}"] = (df[c] - df[c].mean()) / df[c].std()
    df["log2k"] = np.log2(df["k"])
    fit = smf.mixedlm("tau_local ~ log2k + z_d + z_mean_abs_corr + z_imbalance"
                      " + C(model, Treatment('xgboost'))", df, groups=df["dataset"]).fit()
    coef = pd.DataFrame({"coef": fit.params, "se": fit.bse, "p": fit.pvalues,
                         "ci_lo": fit.conf_int()[0], "ci_hi": fit.conf_int()[1]})
    return coef.drop(index="Group Var", errors="ignore"), fit


def dissociation(summary: pd.DataFrame) -> pd.DataFrame:
    """Do predictions stabilise faster with k than attributions? Per dataset and model,
    log ratio of instability at the largest vs the smallest k, for predictions
    (per-point SD of p(true class)) and attributions (1 - tau_local)."""
    ks = sorted(summary.k.unique())
    lo, hi = summary[summary.k == ks[0]], summary[summary.k == ks[-1]]
    m = lo.merge(hi, on=["dataset", "model"], suffixes=("_lo", "_hi"))
    m["pred_logratio"] = np.log(m.p_true_sd_point_hi / m.p_true_sd_point_lo)
    m["attr_logratio"] = np.log((1 - m.tau_local_hi) / (1 - m.tau_local_lo))
    rows = []
    for model, g in m.groupby("model"):
        gap = (g.pred_logratio - g.attr_logratio).to_numpy()
        rows.append({"model": model, "k_from": ks[0], "k_to": ks[-1],
                     "pred_instability_ratio": float(np.exp(g.pred_logratio.median())),
                     "attr_instability_ratio": float(np.exp(g.attr_logratio.median())),
                     "datasets_pred_faster": int((gap < 0).sum()), "n_datasets": len(gap),
                     "p": float(wilcoxon(gap).pvalue) if np.any(gap != 0) else 1.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- H4

def bagging_curve(base: np.ndarray, n_splits: int, rng) -> pd.DataFrame:
    """Context-bagged Shapley values: tau_local between the mean phi of two disjoint
    random sets of M contexts, for M = 1 .. M_total // 2."""
    rows = []
    for M in range(1, len(base) // 2 + 1):
        taus = []
        for _ in range(n_splits):
            perm = rng.permutation(len(base))
            a, b = base[perm[:M]].mean(0), base[perm[M:2 * M]].mean(0)
            taus.append(np.nanmean(kendall_rows(np.abs(a), np.abs(b))))
        rows.append({"M": M, "tau": float(np.mean(taus))})
    return pd.DataFrame(rows)


def contexts_needed(curve: pd.DataFrame, target: float = TAU_TARGET) -> tuple[float, bool]:
    """Smallest observed M reaching the target, else the M where a power-law fit of
    1 - tau against M reaches it (flagged as extrapolated)."""
    hit = curve[curve.tau >= target]
    if len(hit):
        return float(hit.M.min()), False
    y = np.log(np.clip(1 - curve.tau.to_numpy(), 1e-6, None))
    slope, icpt = np.polyfit(np.log(curve.M.to_numpy()), y, 1)
    if slope >= 0:
        return float("inf"), True
    return float(np.exp((np.log(1 - target) - icpt) / slope)), True


def h4(root: Path, summary: pd.DataFrame, n_splits: int, rng):
    curves, need = [], []
    for r in summary.itertuples():
        base, _ = load_cell(root, r.dataset, r.model, r.k)
        c = bagging_curve(base, n_splits, rng).assign(dataset=r.dataset, model=r.model, k=r.k)
        curves.append(c)
        M, extrap = contexts_needed(c)
        need.append({"dataset": r.dataset, "model": r.model, "k": r.k, "M_needed": M,
                     "extrapolated": extrap, "gpu_seconds": M * r.sec_per_run})
    return pd.concat(curves, ignore_index=True), pd.DataFrame(need)


# ---------------------------------------------------------------- figures

def _models(df: pd.DataFrame) -> list[str]:
    order = TFMS + BASELINES + ["tabpfn_v2"]
    return [m for m in order if m in set(df.model)]


def fig_vr(vr: pd.DataFrame, out: Path) -> None:
    ks = sorted(vr.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 4.2), sharey=True)
    order = vr.groupby("dataset").vr.median().sort_values().index
    for ax, k in zip(np.atleast_1d(axes), ks):
        for j, m in enumerate(_models(vr)):
            g = vr[(vr.k == k) & (vr.model == m)].set_index("dataset").reindex(order)
            y = np.arange(len(order)) + (j - 1) * 0.22
            ax.errorbar(g.vr, y, xerr=[g.vr - g.vr_lo, g.vr_hi - g.vr], fmt="o", ms=3.5,
                        color=COLOR[m], label=LABEL[m], elinewidth=1)
        ax.axvline(1, color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_title(f"k = {k}")
        ax.set_xlabel("VR = context var / seed var")
        ax.set_yticks(range(len(order)), order, fontsize=8)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_stability_vs_k(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("tau_local", "Local attribution stability\n(Kendall τ, ↑)"),
              ("tau_global", "Global importance stability\n(Kendall τ, ↑)"),
              ("p_true_sd_point", "Prediction instability\n(SD of p(true class), ↓)")]
    for ax, (col, title) in zip(axes, panels):
        for m in _models(summary):
            g = summary[summary.model == m].groupby("k")[col]
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            ax.plot(med.index, med, "o-", color=COLOR[m], label=LABEL[m])
            ax.fill_between(med.index, q1, q3, color=COLOR[m], alpha=0.12)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(summary.k.unique()), [str(k) for k in sorted(summary.k.unique())])
        ax.set_xlabel("context size k")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_drivers(coef: pd.DataFrame, out: Path) -> None:
    c = coef.drop(index="Intercept", errors="ignore").iloc[::-1]
    names = [n.replace("C(model, Treatment('xgboost'))[T.", "model: ").rstrip("]")
             .replace("z_", "").replace("log2k", "log2 k") for n in c.index]
    names = [LABEL.get(n.split(": ")[-1], n) if n.startswith("model") else n for n in names]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.errorbar(c.coef, range(len(c)), xerr=[c.coef - c.ci_lo, c.ci_hi - c.coef], fmt="o",
                color="#2f6db5")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(range(len(c)), names, fontsize=9)
    ax.set_xlabel("effect on local Kendall τ (95% CI)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_bagging(curves: pd.DataFrame, out: Path) -> None:
    ks = sorted(curves.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 3.6), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), ks):
        for m in _models(curves):
            g = curves[(curves.k == k) & (curves.model == m)].groupby("M").tau
            ax.plot(g.median().index, g.median(), "o-", color=COLOR[m], label=LABEL[m])
            ax.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75),
                            color=COLOR[m], alpha=0.12)
        ax.axhline(TAU_TARGET, color="k", lw=0.8, ls="--")
        ax.set_title(f"k = {k}")
        ax.set_xlabel("contexts averaged, M")
    np.atleast_1d(axes)[0].set_ylabel("local τ between two bagged estimates")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- report

def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Markdown table without the optional `tabulate` dependency."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return format(v, floatfmt)
        return LABEL.get(v, str(v)) if isinstance(v, str) else str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def report(out: Path, summary, vr, tests, coef, diss, need) -> None:
    lines = ["# Analysis report", "",
             f"{summary.dataset.nunique()} datasets, models: "
             f"{', '.join(LABEL.get(m, m) for m in _models(summary))}, "
             f"k = {', '.join(str(int(k)) for k in sorted(summary.k.unique()))}, "
             f"{int(summary.n_contexts.max())} contexts per cell.", ""]

    lines += ["## H1: context variance vs model-seed variance", ""]
    if len(vr):
        h = vr.groupby(["model", "k"]).agg(median_vr=("vr", "median"), min_vr=("vr", "min"),
                                           datasets_ci_above_1=("vr_lo", lambda s: int((s > 1).sum())),
                                           n=("vr", "size")).reset_index()
        lines += [_md(h, ".1f"), ""]

    lines += ["## H2: TFM vs baseline at matched k (local Kendall τ)", "",
              "median_diff > 0 means the TFM is more stable. p_holm is Holm-corrected over k.", ""]
    lines += [_md(tests[tests.metric == "tau_local"].drop(columns="metric")), ""]

    lines += ["## H3: drivers of local stability (mixed model, random intercept per dataset)", ""]
    lines += [_md(coef.reset_index(names="term")), ""]
    lines += ["Prediction vs attribution instability, largest vs smallest k "
              "(ratio < 1 = more stable at large k):", "", _md(diss), ""]

    lines += [f"## H4: contexts needed for bagged local τ ≥ {TAU_TARGET}", ""]
    nd = need.groupby(["model", "k"]).agg(
        median_M=("M_needed", "median"),
        reached_within_half=("extrapolated", lambda s: int((~s).sum())),
        n=("M_needed", "size"),
        median_gpu_min=("gpu_seconds", lambda s: float(np.median(s) / 60))).reset_index()
    lines += [_md(nd, ".1f"), "",
              "M up to half the contexts is observed; larger M is a power-law extrapolation.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--summary", help="summary CSV (default: <results>/summary.csv)")
    ap.add_argument("--out", help="output directory (default: <results>/analysis)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.results)
    out = Path(args.out) if args.out else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summary = pd.read_csv(args.summary or root / "summary.csv")
    traits = dataset_traits(load_config(args.config))
    traits.to_csv(out / "dataset_traits.csv", index=False)

    vr = h1(root, summary, args.n_boot, rng)
    vr.to_csv(out / "h1_variance_ratio.csv", index=False)
    tests = pd.concat([paired_tests(summary, m, args.n_boot, rng)
                       for m in ["tau_local", "jaccard_top3", "sign_flip", "tau_global"]])
    tests.to_csv(out / "h2_matched_k_tests.csv", index=False)
    coef, _ = drivers(summary, traits)
    coef.to_csv(out / "h3_mixed_model.csv")
    diss = dissociation(summary)
    diss.to_csv(out / "h3_dissociation.csv", index=False)
    curves, need = h4(root, summary, args.n_splits, rng)
    curves.to_csv(out / "h4_bagging_curves.csv", index=False)
    need.to_csv(out / "h4_contexts_needed.csv", index=False)

    if len(vr):
        fig_vr(vr, out / "fig_vr.png")
    fig_stability_vs_k(summary, out / "fig_stability_vs_k.png")
    fig_drivers(coef, out / "fig_drivers.png")
    fig_bagging(curves, out / "fig_bagging.png")
    report(out, summary, vr, tests, coef, diss, need)
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console defaults to cp1252
    print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
