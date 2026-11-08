"""Robustness checks of the main results; nothing here changes a run.

    python -m tfmxai.robustness --main results/main --synthetic results/synthetic

Writes <main>/analysis/robustness/: one CSV per check and robustness.md. All checks read
finished runs. They ask how far the main results survive estimator noise, overlap between
contexts, model-structure choices, baseline hyperparameters and the extrapolation in H4.
"""

import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from tfmxai.analysis import BASELINES, TFMS, _md, paired_tests
from tfmxai.metrics import kendall_rows
from tfmxai.summarize import collect

warnings.filterwarnings("ignore")

CLOSE_LOGIT = 0.1


# ---------------------------------------------------------------- noise ceilings

def _pair_tau(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmean(kendall_rows(np.abs(a), np.abs(b))))


def noise_ceilings(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    """Local tau between reruns on one context: other model seed, other KernelSHAP seed.
    These bound what any between-context tau can reach; var ratios put noise on one scale."""
    rows = []
    for r in summary.itertuples():
        cell = root / "raw" / r.dataset / r.model / f"k{r.k}"
        tau_s, tau_e = [], []
        for c in range(3):
            base = cell / f"c{c:02d}_s0_e0.npz"
            seed, expl = cell / f"c{c:02d}_s1_e0.npz", cell / f"c{c:02d}_s0_e1.npz"
            if seed.exists():
                tau_s.append(_pair_tau(np.load(base)["phi"], np.load(seed)["phi"]))
            if expl.exists():
                tau_e.append(_pair_tau(np.load(base)["phi"], np.load(expl)["phi"]))
        rows.append({"dataset": r.dataset, "model": r.model, "k": r.k, "d": r.d,
                     "exact": r.d <= 8, "tau_context": r.tau_local,
                     "tau_seed": np.mean(tau_s) if tau_s else np.nan,
                     "tau_explainer": np.mean(tau_e) if tau_e else np.nan,
                     "var_context": r.var_context, "var_seed": r.var_seed,
                     "var_explainer": r.var_explainer,
                     "explainer_share": r.var_explainer / r.var_context})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- same prediction, other story

def close_predictions(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    """Top-feature agreement over all context pairs, and over pairs whose full-model logits
    differ by at most CLOSE_LOGIT. Quantifies what Figure 1 shows for one selected pair."""
    rows = []
    for r in summary.itertuples():
        cell = root / "raw" / r.dataset / r.model / f"k{r.k}"
        runs = [np.load(cell / f"c{c:02d}_s0_e0.npz") for c in range(10)]
        phis = np.stack([z["phi"] for z in runs])
        v = np.stack([z["v_full"] for z in runs])
        top = np.abs(phis).argmax(2)
        constant = (np.ptp(np.abs(phis), axis=2) == 0).mean()
        same, close, same_close = [], [], []
        for a, b in combinations(range(len(phis)), 2):
            s, c = top[a] == top[b], np.abs(v[a] - v[b]) <= CLOSE_LOGIT
            same.append(s), close.append(c), same_close.append(s[c])
        rows.append({"dataset": r.dataset, "model": r.model, "k": r.k,
                     "top1_agree": float(np.concatenate(same).mean()),
                     "close_share": float(np.concatenate(close).mean()),
                     "top1_agree_close": float(np.concatenate(same_close).mean()),
                     "constant_phi_share": float(constant)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- model structure

def _fit_mixed(df: pd.DataFrame, formula: str, label: str, re_formula: str | None = None) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    fit = smf.mixedlm(formula, df, groups=df["dataset"], re_formula=re_formula).fit()
    ci = fit.conf_int()
    t = pd.DataFrame({"spec": label, "term": fit.params.index, "coef": fit.params.values,
                      "se": fit.bse.values, "p": fit.pvalues.values,
                      "ci_lo": ci[0].values, "ci_hi": ci[1].values})
    return t[~t.term.str.contains("Var|Cov", regex=True)]


def cluster_bootstrap(df: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    """Percentile CI of the fixed effects from resampling whole datasets (OLS point fits)."""
    cols = ["log2k", "z_d", "z_mean_abs_corr", "z_imbalance"]
    X = pd.concat([df[cols], pd.get_dummies(df.model).astype(float).drop(columns="xgboost")], axis=1)
    X.insert(0, "Intercept", 1.0)
    names, Xv, y = list(X.columns), X.to_numpy(), df.tau_local.to_numpy()
    groups = [np.flatnonzero(df.dataset == d) for d in df.dataset.unique()]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boots.append(np.linalg.lstsq(Xv[idx], y[idx], rcond=None)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    point = np.linalg.lstsq(Xv, y, rcond=None)[0]
    return pd.DataFrame({"term": names, "ols": point, "boot_lo": lo, "boot_hi": hi})


def model_specs(summary: pd.DataFrame, traits: pd.DataFrame, n_boot: int, rng):
    df = summary.drop(columns=["d"]).merge(traits, on="dataset")
    for c in ["d", "mean_abs_corr", "imbalance"]:
        df[f"z_{c}"] = (df[c] - df[c].mean()) / df[c].std()
    df["log2k"] = np.log2(df["k"])
    df["overlap"] = df["k"] / df["n_pool"]
    df["kernelshap"] = (df["d"] > 8).astype(float)
    base = ("tau_local ~ log2k + z_d + z_mean_abs_corr + z_imbalance"
            " + C(model, Treatment('xgboost'))")
    specs = pd.concat([_fit_mixed(df, base, "base"),
                       _fit_mixed(df, base, "random slope for log2k", "~log2k"),
                       _fit_mixed(df, base + " + overlap", "adds k/n_pool"),
                       _fit_mixed(df, base + " + kernelshap", "adds KernelSHAP indicator")])
    return specs, cluster_bootstrap(df, n_boot, rng), df


# ---------------------------------------------------------------- tests on subsets

def tfm_vs_tfm(summary: pd.DataFrame, metric: str = "tau_local") -> pd.DataFrame:
    piv = summary.pivot_table(index=["dataset", "k"], columns="model", values=metric)
    rows = []
    for k in sorted(summary.k.unique()):
        diff = (piv.xs(k, level="k")["tabpfn_v2_5"] - piv.xs(k, level="k")["tabicl_v2"]).dropna()
        rows.append({"metric": metric, "k": k, "median_diff": diff.median(),
                     "wins_tabpfn": int((diff > 0).sum()), "n": len(diff),
                     "p": wilcoxon(diff).pvalue})
    t = pd.DataFrame(rows)
    t["p_holm"] = multipletests(t.p, method="holm")[1]
    return t


def predictive(summary: pd.DataFrame) -> pd.DataFrame:
    """Log-loss on the explained points, against the best-known comparison per k."""
    piv = summary.pivot_table(index=["dataset", "k"], columns="model", values="logloss_mean")
    rows = []
    for k in sorted(summary.k.unique()):
        p = piv.xs(k, level="k")
        for m in p.columns:
            rows.append({"k": k, "model": m, "median_logloss": p[m].median(),
                         "best_on": int((p.idxmin(axis=1) == m).sum())})
    return pd.DataFrame(rows)


def stability_vs_loss(summary: pd.DataFrame) -> pd.DataFrame:
    """Does the more stable model of a TFM-baseline pair also predict better?"""
    rows = []
    piv_t = summary.pivot_table(index=["dataset", "k"], columns="model", values="tau_local")
    piv_l = summary.pivot_table(index=["dataset", "k"], columns="model", values="logloss_mean")
    for tfm in TFMS:
        for base in BASELINES:
            dl = piv_l[tfm] - piv_l[base]
            rows.append({"tfm": tfm, "baseline": base,
                         "median_logloss_diff": dl.median(),
                         "cells_tfm_lower_loss": int((dl < 0).sum()), "cells": len(dl),
                         "cells_tfm_stabler": int((piv_t[tfm] > piv_t[base]).sum()),
                         "cells_tfm_stabler_and_lower_loss": int(((piv_t[tfm] > piv_t[base]) & (dl < 0)).sum())})
    return pd.DataFrame(rows)


def dissociation_logit(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    """The dissociation on one scale: per-point SD across contexts of the full-model logit and
    of phi, at the largest vs the smallest k. p(true class) saturates and tau is rank-based,
    so this checks that the headline dissociation is not an artefact of those two scales."""
    ks = sorted(summary.k.unique())
    rows = {}
    for r in summary[summary.k.isin([ks[0], ks[-1]])].itertuples():
        cell = root / "raw" / r.dataset / r.model / f"k{r.k}"
        runs = [np.load(cell / f"c{c:02d}_s0_e0.npz") for c in range(10)]
        rows[(r.dataset, r.model, r.k)] = (np.stack([z["v_full"] for z in runs]).std(0, ddof=1).mean(),
                                           np.stack([z["phi"] for z in runs]).std(0, ddof=1).mean())
    out = []
    for model in summary.model.unique():
        lv, lp = [], []
        for ds in summary.dataset.unique():
            lo, hi = rows[(ds, model, ks[0])], rows[(ds, model, ks[-1])]
            lv.append(np.log(hi[0] / lo[0]))
            lp.append(np.log(hi[1] / lo[1]))
        gap = np.array(lv) - np.array(lp)
        out.append({"model": model, "logit_sd_ratio": np.exp(np.median(lv)),
                    "phi_sd_ratio": np.exp(np.median(lp)), "datasets_logit_faster": int((gap < 0).sum()),
                    "n": len(gap), "p": wilcoxon(gap).pvalue})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- H4 extrapolation

def _powerlaw_tau(curve: pd.DataFrame, at: int) -> float:
    slope, icpt = np.polyfit(np.log(curve.M), np.log(np.clip(1 - curve.tau, 1e-6, None)), 1)
    return float(1 - np.exp(icpt + slope * np.log(at)))


def _needed(curve: pd.DataFrame, target: float = 0.9) -> float:
    slope, icpt = np.polyfit(np.log(curve.M), np.log(np.clip(1 - curve.tau, 1e-6, None)), 1)
    if slope >= 0:
        return np.inf
    with np.errstate(over="ignore"):  # a nearly flat decline overflows to inf, which is the answer
        return float(np.exp((np.log(1 - target) - icpt) / slope))


def bagging_backtest(curves: pd.DataFrame) -> pd.DataFrame:
    """Fit the power law on M = 1..3, predict the observed tau at M = 5, and compare the M
    needed for tau = 0.9 from the short and the full fit."""
    rows = []
    for (ds, model, k), g in curves.groupby(["dataset", "model", "k"]):
        short, obs = g[g.M <= 3], g[g.M == 5].tau.iloc[0]
        rows.append({"dataset": ds, "model": model, "k": k,
                     "tau_m1": g[g.M == 1].tau.iloc[0], "tau_m4": g[g.M == 4].tau.iloc[0],
                     "tau_m5": obs, "pred_m5": _powerlaw_tau(short, 5),
                     "err_m5": _powerlaw_tau(short, 5) - obs,
                     "needed_short": _needed(short), "needed_full": _needed(g)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- oracle

def percentile_coverage_reference(M: int = 10, level: float = 0.9, n_sim: int = 4000,
                                  n_boot: int = 1000, seed: int = 0) -> dict:
    """Coverage of both interval types for unbiased Gaussian attributions with M contexts."""
    rng = np.random.default_rng(seed)
    a = (1 - level) / 2
    x = rng.normal(size=(n_sim, M))
    idx = rng.integers(0, M, (n_boot, n_sim, M))
    boot = x[np.arange(n_sim)[None, :, None], idx].mean(2)
    lo, hi = np.quantile(boot, [a, 1 - a], axis=0)
    s_lo, s_hi = np.quantile(x, [a, 1 - a], axis=1)
    return {"cover_bagged_ci": float(((lo <= 0) & (hi >= 0)).mean()),
            "cover_single_range": float(((s_lo <= 0) & (s_hi >= 0)).mean())}


def oracle_summary(synth: Path) -> pd.DataFrame:
    o = pd.read_csv(synth / "analysis" / "oracle.csv")
    # bias2 of the mean carries var/M of noise; remove it for an unbiased split.
    o["bias2_debiased"] = (o.bias2 - o["var"] / o.n_contexts).clip(lower=0)
    o["var_share_debiased"] = o["var"] / (o["var"] + o.bias2_debiased)
    return o


# ---------------------------------------------------------------- grouped vs summed

def grouped_agreement(main: Path, grouped: Path) -> pd.DataFrame:
    """Per point: do grouped Shapley values match main-grid values summed within groups?"""
    rows = []
    for f in sorted((grouped / "raw").glob("*/*/k*")):
        ds, model, k = f.parts[-3], f.parts[-2], int(f.parts[-1][1:])
        rs, ts = [], []
        for c in range(10):
            z = np.load(f / f"c{c:02d}_s0_e0.npz")
            phi_g, g = z["phi"], z["groups"]
            phi_m = np.load(main / "raw" / ds / model / f"k{k}" / f"c{c:02d}_s0_e0.npz")["phi"]
            summed = np.stack([phi_m[:, g == j].sum(1) for j in range(g.max() + 1)], axis=1)
            a, b = phi_g - phi_g.mean(1, keepdims=True), summed - summed.mean(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = (a * b).sum(1) / np.sqrt((a**2).sum(1) * (b**2).sum(1))
            rs.append(np.nanmean(r))
            ts.append(_pair_tau(phi_g, summed))
        rows.append({"dataset": ds, "model": model, "k": k, "players": int(g.max()) + 1,
                     "d": len(g), "pearson": float(np.mean(rs)), "tau": float(np.mean(ts))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- baseline hyperparameters

def shallow_xgb(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    """TFMs against strongly regularised XGBoost on the same cells (6 exact-Shapley datasets)."""
    if not (root / "raw").exists():
        return pd.DataFrame()
    abl = collect(root)
    if abl.empty:
        return pd.DataFrame()
    abl.to_csv(root / "summary.csv", index=False)
    both = pd.concat([summary, abl]).pivot_table(index=["dataset", "k"], columns="model",
                                                 values=["tau_local", "logloss_mean"])
    both = both.dropna(subset=[("tau_local", "xgboost_shallow")])
    rows = []
    for k in sorted(abl.k.unique()):
        t, ll = both["tau_local"].xs(k, level="k"), both["logloss_mean"].xs(k, level="k")
        for m in TFMS + ["xgboost", "logreg"]:
            diff = (t[m] - t["xgboost_shallow"]).to_numpy()
            rows.append({"k": k, "model": m, "tau_model": np.median(t[m]),
                         "tau_shallow": np.median(t["xgboost_shallow"]),
                         "median_diff": np.median(diff), "wins": int((diff > 0).sum()),
                         "n": len(diff), "p": wilcoxon(diff).pvalue,
                         "logloss_model": np.median(ll[m]),
                         "logloss_shallow": np.median(ll["xgboost_shallow"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def _agg(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    return df.groupby(by)[cols].median().reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/main")
    ap.add_argument("--synthetic", default="results/synthetic")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root, synth = Path(args.main), Path(args.synthetic)
    out = root / "analysis" / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary = pd.read_csv(root / "summary.csv")
    traits = pd.read_csv(root / "analysis" / "dataset_traits.csv")
    lines = ["# Robustness checks", ""]

    def section(title, table, name, fmt=".3f"):
        table.to_csv(out / f"{name}.csv", index=False)
        lines.extend([f"## {title}", "", _md(table, fmt), ""])

    noise = noise_ceilings(root, summary)
    noise.to_csv(out / "noise_ceilings.csv", index=False)
    section("Between-context tau vs noise ceilings",
            _agg(noise, ["exact", "model", "k"],
                 ["tau_context", "tau_seed", "tau_explainer", "explainer_share"]), "noise_agg")
    section("Variance components by k",
            _agg(noise, ["model", "k"], ["var_context", "var_seed"]), "variance_by_k", ".5f")

    close = close_predictions(root, summary)
    close.to_csv(out / "close_predictions.csv", index=False)
    section("Top-feature agreement, all pairs vs near-identical predictions",
            _agg(close, ["model", "k"], ["top1_agree", "close_share", "top1_agree_close",
                                         "constant_phi_share"]), "close_agg")

    specs, boot, df = model_specs(summary, traits, args.n_boot, rng)
    section("Mixed-model specifications", specs, "model_specs")
    section("Cluster bootstrap over datasets", boot, "model_bootstrap")
    section("Dataset table", df.drop_duplicates("dataset")[
        ["dataset", "d", "n_classes", "n_pool", "mean_abs_corr", "imbalance"]], "datasets")

    tests = paired_tests(summary, "tau_local", args.n_boot, rng)
    tests["p_holm_all12"] = multipletests(tests.p, method="holm")[1]
    section("H2 with Holm over all 12 tests", tests[["tfm", "baseline", "k", "p", "p_holm", "p_holm_all12"]],
            "h2_all12")
    sub = []
    for name, part in [("exact", summary[summary.d <= 8]), ("kernelshap", summary[summary.d > 8])]:
        sub.append(paired_tests(part, "tau_local", args.n_boot, rng).assign(subset=name))
    sub = pd.concat(sub)
    section("H2 by estimator", sub[["subset", "tfm", "baseline", "k", "n_datasets", "median_diff",
                                    "wins", "p_holm"]], "h2_by_estimator")
    section("TabPFN-2.5 vs TabICLv2", tfm_vs_tfm(summary), "tfm_vs_tfm")
    section("Predictive log-loss", predictive(summary), "predictive")
    section("Stability against predictive loss", stability_vs_loss(summary), "stability_vs_loss")

    section("Dissociation on the logit scale", dissociation_logit(root, summary), "dissociation_logit")

    curves = pd.read_csv(root / "analysis" / "h4_bagging_curves.csv")
    back = bagging_backtest(curves)
    back.to_csv(out / "bagging_backtest.csv", index=False)
    section("Bagging: observed tau and extrapolation backtest",
            _agg(back, ["model", "k"], ["tau_m1", "tau_m4", "tau_m5", "pred_m5", "err_m5",
                                        "needed_short", "needed_full"]), "bagging_agg", ".2f")

    o = oracle_summary(synth)
    o.to_csv(out / "oracle_debiased.csv", index=False)
    section("Oracle by model", _agg(o, ["model"], ["oracle_tau_single", "oracle_tau_bagged", "var_share",
                                                   "var_share_debiased", "cover_bagged_ci",
                                                   "cover_single_range"]), "oracle_by_model")
    section("Oracle by model and k", _agg(o, ["model", "k"], ["oracle_tau_single", "oracle_tau_bagged",
                                                              "var_share_debiased"]), "oracle_by_model_k")
    tfm = o[o.model.isin(TFMS)]
    section("Oracle TFMs by rho", _agg(tfm, ["rho"], ["tau_local", "oracle_tau_single", "oracle_tau_bagged"]),
            "oracle_by_rho")
    ref = percentile_coverage_reference()
    section("Interval coverage for unbiased Gaussian attributions, M = 10",
            pd.DataFrame([ref]), "coverage_reference")

    group = grouped_agreement(root, root.parent / "ablation_grouped")
    group.to_csv(out / "grouped_agreement.csv", index=False)
    section("Grouped Shapley vs summed per-feature values", _agg(group, ["model", "k"], ["pearson", "tau"]),
            "grouped_agg")

    shallow = shallow_xgb(root.parent / "ablation_xgb_shallow", summary)
    if len(shallow):
        section("Regularised XGBoost sensitivity", shallow, "xgb_shallow")

    (out / "robustness.md").write_text("\n".join(lines), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
