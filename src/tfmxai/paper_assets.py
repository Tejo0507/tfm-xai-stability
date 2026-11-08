"""Figures and LaTeX tables of the manuscript, generated from the result files only.

    python -m tfmxai.paper_assets --out results/paper_assets

Run `tfmxai.analysis`, `tfmxai.ablations`, `tfmxai.oracle_analysis` and `tfmxai.robustness`
first. Nothing is typed by hand, so a table cannot drift from the outputs it reports.
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from tfmxai.analysis import BASELINES, TFMS
from tfmxai.data import load_dataset
from tfmxai.example_figure import pick
from tfmxai.metrics import kendall_rows
from tfmxai.run import load_config

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = TFMS + BASELINES
NAME = {"tabpfn_v2_5": "TabPFN-2.5", "tabicl_v2": "TabICLv2", "xgboost": "XGBoost", "logreg": "Log.\\ reg."}
PLOT_NAME = {**NAME, "logreg": "Logistic reg."}
# Okabe-Ito colours with a distinct marker per model: the figures also read in greyscale.
COLOR = {"tabpfn_v2_5": "#0072B2", "tabicl_v2": "#D55E00", "xgboost": "#009E73", "logreg": "#555555"}
MARKER = {"tabpfn_v2_5": "o", "tabicl_v2": "s", "xgboost": "^", "logreg": "D"}
WIDTH = 6.5

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
                     "lines.markersize": 4})


# ---------------------------------------------------------------- figures

def fig_example(root: Path, cfg_path: str, out: Path) -> dict:
    ds, model, k = "adult", "tabpfn_v2_5", 512
    cell = root / "raw" / ds / model / f"k{k}"
    runs = [np.load(f) for f in sorted(cell.glob("c*_s0_e0.npz"))]
    phis, v = np.stack([r["phi"] for r in runs]), np.stack([r["v_full"] for r in runs])
    p = 1 / (1 + np.exp(-v))
    i, a, b, tau = pick(phis, v, 0.1)
    pairs = combinations(range(len(phis)), 2)
    typical = np.nanmedian([np.nanmedian(kendall_rows(np.abs(phis[x]), np.abs(phis[y]))) for x, y in pairs])
    spec = next(s for s in load_config(cfg_path)["datasets"] if s["name"] == ds)
    names = np.array(load_dataset(spec).feature_names)
    order = np.argsort(-np.abs(phis[[a, b], i]).max(0))[:8][::-1]
    lim = np.abs(phis[[a, b], i][:, order]).max() * 1.1
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.35), sharey=True, layout="constrained")
    for ax, c, name in zip(axes, (a, b), ("A", "B")):
        vals = phis[c, i, order]
        ax.barh(range(len(order)), vals, color=COLOR[model])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_title(f"Context {name}: p(true class) = {p[c, i]:.3f}")
        ax.set_xlabel("Shapley value (logit of the true class)")
    axes[0].set_yticks(range(len(order)), names[order])
    fig.savefig(out / "fig_example.pdf")
    plt.close(fig)
    return {"point": i, "ctx_a": a, "ctx_b": b, "tau": tau, "median_tau": float(typical),
            "p_a": float(p[a, i]), "p_b": float(p[b, i]), "dlogit": float(abs(v[a, i] - v[b, i]))}


def fig_vr(vr: pd.DataFrame, out: Path) -> None:
    ks = sorted(vr.k.unique())
    order = vr.groupby("dataset").vr.median().sort_values().index
    fig, axes = plt.subplots(1, len(ks), figsize=(WIDTH, 3.25), sharey=True, sharex=True, layout="constrained")
    for ax, k in zip(axes, ks):
        for j, m in enumerate(["tabpfn_v2_5", "tabicl_v2", "xgboost"]):
            g = vr[(vr.k == k) & (vr.model == m)].set_index("dataset").reindex(order)
            ax.errorbar(g.vr, np.arange(len(order)) + (j - 1) * 0.24, xerr=[g.vr - g.vr_lo, g.vr_hi - g.vr],
                        fmt=MARKER[m], ms=3, color=COLOR[m], label=PLOT_NAME[m], elinewidth=0.8)
        ax.axvline(1, color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_title(f"k = {k}")
        ax.set_xlabel("variance ratio (log scale)")
    axes[0].set_yticks(range(len(order)), order)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(out / "fig_vr.pdf")
    plt.close(fig)


def fig_stability(summary: pd.DataFrame, close: pd.DataFrame, out: Path) -> None:
    df = summary.merge(close, on=["dataset", "model", "k"])
    panels = [("tau_local", "Local attribution stability", "Kendall $\\tau$ between contexts"),
              ("top1_agree", "Top-feature agreement", "share of context pairs"),
              ("p_true_sd_point", "Prediction instability", "per-point SD of p(true class)")]
    ks = sorted(df.k.unique())
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.25), layout="constrained")
    for ax, (col, title, ylabel) in zip(axes, panels):
        for m in MODELS:
            g = df[df.model == m].groupby("k")[col]
            ax.plot(ks, g.median(), marker=MARKER[m], color=COLOR[m], label=PLOT_NAME[m])
            ax.fill_between(ks, g.quantile(0.25), g.quantile(0.75), color=COLOR[m], alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlabel("context size k")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(out / "fig_stability.pdf")
    plt.close(fig)


def fig_bagging(curves: pd.DataFrame, out: Path) -> None:
    ks = sorted(curves.k.unique())
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.2), sharey=True, layout="constrained")
    for ax, k in zip(axes, ks):
        for m in MODELS:
            g = curves[(curves.k == k) & (curves.model == m)].groupby("M").tau
            ax.plot(g.median().index, g.median(), marker=MARKER[m], color=COLOR[m], label=PLOT_NAME[m])
            ax.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75), color=COLOR[m], alpha=0.12, lw=0)
        ax.axhline(0.9, color="k", lw=0.8, ls="--")
        ax.set_xticks(range(1, 6))
        ax.set_title(f"k = {k}")
        ax.set_xlabel("contexts averaged per bag, M")
    axes[0].set_ylabel("Kendall $\\tau$ between disjoint bags")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(out / "fig_bagging.pdf")
    plt.close(fig)


def fig_oracle(o: pd.DataFrame, out: Path) -> None:
    ds, ks = sorted(o.d.unique()), sorted(o.k.unique())
    fig, axes = plt.subplots(len(ds), len(ks), figsize=(WIDTH, 3.5), sharex=True, sharey=True, layout="constrained")
    for i, d in enumerate(ds):
        for j, k in enumerate(ks):
            ax = axes[i, j]
            for m in MODELS:
                g = o[(o.d == d) & (o.k == k) & (o.model == m)].sort_values("rho")
                ax.plot(g.rho, g.oracle_tau_bagged, marker=MARKER[m], color=COLOR[m], label=PLOT_NAME[m])
                ax.plot(g.rho, g.oracle_tau_single, marker=MARKER[m], color=COLOR[m], ls=":", mfc="none")
            ax.set_xticks(sorted(o.rho.unique()))
            ax.set_title(f"d = {d}, k = {k}")
            if i == len(ds) - 1:
                ax.set_xlabel("feature correlation $\\rho$")
            if j == 0:
                ax.set_ylabel("$\\tau$ with oracle")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(out / "fig_oracle.pdf")
    plt.close(fig)


def fig_datasets(summary: pd.DataFrame, traits: pd.DataFrame, out: Path, k: int = 1024) -> None:
    """Local tau of every model on every dataset at one context size, datasets ordered by d."""
    s = summary[summary.k == k]
    order = traits.sort_values(["d", "dataset"]).dataset.tolist()
    d = traits.set_index("dataset").d
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0), layout="constrained")
    y = np.arange(len(order))
    for j, m in enumerate(MODELS):
        g = s[s.model == m].set_index("dataset").reindex(order)
        ax.plot(g.tau_local, y + (j - 1.5) * 0.13, MARKER[m], color=COLOR[m], ms=4, label=PLOT_NAME[m],
                mfc=COLOR[m] if m in TFMS else "white")
    for i in range(len(order)):
        ax.axhline(i, color="#e6e6e6", lw=0.5, zorder=0)
    split = sum(d[n] <= 8 for n in order) - 0.5
    ax.axhline(split, color="k", lw=0.7, ls="--")
    ax.text(1.01, split + 0.15, "KernelSHAP\n($d > 8$)", transform=ax.get_yaxis_transform(), ha="left",
            va="bottom", fontsize=7)
    ax.text(1.01, split - 0.15, "exact Shapley\n($d \\leq 8$)", transform=ax.get_yaxis_transform(), ha="left",
            va="top", fontsize=7)
    ax.set_yticks(y, [f"{n} ($d$={d[n]})" for n in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(f"local $\\tau$ between contexts at $k = {k}$ (higher = more stable)")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(out / "fig_datasets.pdf")
    plt.close(fig)


def fig_tradeoff(summary: pd.DataFrame, shallow: pd.DataFrame, out: Path) -> None:
    """Stability against log-loss per dataset: all 16 datasets (left) and the six exact datasets with the
    regularised XGBoost added (right), both at k = 1024."""
    s = summary[summary.k == 1024]
    exact = sorted(shallow.dataset.unique()) if not shallow.empty else []
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True, layout="constrained")
    panels = ((axes[0], s, "16 datasets"), (axes[1], s[s.dataset.isin(exact)], "six exact datasets"))
    for ax, g, title in panels:
        for m in MODELS:
            h = g[g.model == m]
            ax.scatter(h.logloss_mean, h.tau_local, marker=MARKER[m], s=14, color=COLOR[m], alpha=0.35, lw=0)
            ax.scatter(h.logloss_mean.median(), h.tau_local.median(), marker=MARKER[m], s=55, color=COLOR[m],
                       edgecolor="k", lw=0.6, label=PLOT_NAME[m], zorder=3)
        ax.set_title(title)
        ax.set_xlabel("log-loss on explained points (lower = better)")
        ax.grid(alpha=0.3, lw=0.5)
    if not shallow.empty:
        h = shallow[shallow.k == 1024]
        axes[1].scatter(h.logloss_mean, h.tau_local, marker="v", s=14, color="#CC79A7", alpha=0.35, lw=0)
        axes[1].scatter(h.logloss_mean.median(), h.tau_local.median(), marker="v", s=55, color="#CC79A7",
                        edgecolor="k", lw=0.6, label="XGBoost, depth 2", zorder=3)
    axes[0].set_ylabel("local $\\tau$ (higher = more stable)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5, frameon=False)
    fig.savefig(out / "fig_tradeoff.pdf")
    plt.close(fig)


def fig_remedy(remedy: pd.DataFrame, out: Path) -> None:
    """Median local tau and cost of one context, a 16-member ensemble and a bag of four contexts."""
    arms = [("single_ctx_nest4", "one context", "#BBBBBB", None),
            ("single_ctx_nest16", "larger ensemble (16)", "#7F7F7F", "///"),
            ("bag4_ctx_nest4", "bag of 4 contexts", "#009E73", None)]
    cells = [(m, k) for m in TFMS for k in sorted(remedy.k.unique())]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3), layout="constrained",
                             gridspec_kw={"width_ratios": [1.9, 1]})
    ax = axes[0]
    x = np.arange(len(cells))
    for j, (col, lab, c, hatch) in enumerate(arms):
        vals = [remedy[(remedy.model == m) & (remedy.k == k)][col].median() for m, k in cells]
        ax.bar(x + (j - 1) * 0.26, vals, 0.25, color=c, hatch=hatch, edgecolor="k", lw=0.4, label=lab)
    ax.set_xticks(x, [f"{PLOT_NAME[m]}\n$k$ = {k}" for m, k in cells], fontsize=6.8)
    ax.set_ylabel("median local $\\tau$")
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=7)
    ax = axes[1]
    for m in TFMS:
        g = remedy[remedy.model == m]
        for k, mk in zip(sorted(g.k.unique()), ("o", "s")):
            h = g[g.k == k]
            ax.plot([h.sec_nest16.median(), h.sec_bag4.median()],
                    [h.single_ctx_nest16.median(), h.bag4_ctx_nest4.median()], color=COLOR[m], lw=0.8)
            ax.plot(h.sec_nest16.median(), h.single_ctx_nest16.median(), mk, color=COLOR[m], mfc="white")
            ax.plot(h.sec_bag4.median(), h.bag4_ctx_nest4.median(), mk, color=COLOR[m],
                    label=f"{PLOT_NAME[m]}, $k$ = {k}")
    ax.set_xlabel("seconds per estimate (median)")
    ax.set_ylabel("median local $\\tau$")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2,
              handlelength=1.0, columnspacing=0.8)
    fig.savefig(out / "fig_remedy.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- tables

def _p(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def _p2(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.2f}"


def _s(x: float, f: str = ".3f") -> str:
    return f"${x:+{f}}$"


def _write(out: Path, name: str, spec: str, head: list[str], rows: list[str]) -> None:
    body = "\n".join(rows)
    (out / f"{name}.tex").write_text(
        f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n" + "\n".join(head) + f"\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n",
        encoding="utf-8")


def tab_overview(s: pd.DataFrame, close: pd.DataFrame, out: Path) -> None:
    df = s.merge(close, on=["dataset", "model", "k"])
    med = df.groupby(["model", "k"])[["tau_local", "top1_agree", "top1_agree_close", "logloss_mean"]].median()
    rows = []
    for m in MODELS:
        c = [f"{med.loc[(m, k), col]:.2f}" for col in ["tau_local"] for k in (128, 512, 1024)]
        c += [f"{med.loc[(m, k), 'top1_agree']:.2f}" for k in (128, 512, 1024)]
        c += [f"{med.loc[(m, k), 'top1_agree_close']:.2f}" for k in (128, 1024)]
        c += [f"{med.loc[(m, k), 'logloss_mean']:.2f}" for k in (128, 512, 1024)]
        rows.append(NAME[m] + " & " + " & ".join(c) + " \\\\")
    _write(out, "tab_overview", "l" + "ccc" + "ccc" + "cc" + "ccc",
           ["& \\multicolumn{3}{c}{Local $\\tau$} & \\multicolumn{3}{c}{Top-feature agreement} & "
            "\\multicolumn{2}{c}{Agreement, close preds.} & \\multicolumn{3}{c}{Log-loss} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-12}",
            "Model & 128 & 512 & 1024 & 128 & 512 & 1024 & 128 & 1024 & 128 & 512 & 1024 \\\\"], rows)


def tab_vr(vr: pd.DataFrame, out: Path) -> None:
    g = vr.groupby(["model", "k"]).vr.agg(["median", "min", "max"])
    rows = []
    for m in ["tabpfn_v2_5", "tabicl_v2", "xgboost"]:
        cells = [f"{g.loc[(m, k), 'median']:.0f} & {g.loc[(m, k), 'min']:.0f}--{g.loc[(m, k), 'max']:.0f}"
                 for k in (128, 512, 1024)]
        rows.append(NAME[m] + " & " + " & ".join(cells) + " \\\\")
    _write(out, "tab_vr", "lcccccc",
           ["& \\multicolumn{2}{c}{$k=128$} & \\multicolumn{2}{c}{$k=512$} & \\multicolumn{2}{c}{$k=1024$} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
            "Model & median & range & median & range & median & range \\\\"], rows)


def tab_h2(t: pd.DataFrame, holm12: pd.DataFrame, out: Path) -> None:
    t = t[t.metric == "tau_local"].merge(holm12[["tfm", "baseline", "k", "p_holm_all12"]], on=["tfm", "baseline", "k"])
    rows = []
    for tfm in TFMS:
        for base in BASELINES:
            for k in (128, 512, 1024):
                r = t[(t.tfm == tfm) & (t.baseline == base) & (t.k == k)].iloc[0]
                lead = f"{NAME[tfm]} & {NAME[base]}" if k == 128 else "&"
                rows.append(f"{lead} & {k} & {_s(r.median_diff)} & $[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]$ & "
                            f"{r.wins}/16 & ${r.rank_biserial:+.2f}$ & {_p(r.p_holm_all12)} \\\\")
        if tfm != TFMS[-1]:
            rows.append("\\midrule")
    _write(out, "tab_h2", "llrrcccc",
           ["TFM & Baseline & $k$ & median $\\Delta\\tau$ & 95\\% CI & wins & $r$ & $p_\\text{Holm}$ \\\\"], rows)


def tab_noise(noise: pd.DataFrame, out: Path) -> None:
    rows = []
    for exact, label in [(True, "Exact ($d\\le 8$, 6 datasets)"), (False, "KernelSHAP ($d>8$, 10 datasets)")]:
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{label}}}}} \\\\")
        g = noise[noise.exact == exact].groupby(["model", "k"]).median(numeric_only=True)
        for m in MODELS:
            ctx = " & ".join(f"{g.loc[(m, k), 'tau_context']:.2f}" for k in (128, 512, 1024))
            seed = "n/a" if m == "logreg" else f"{g.loc[(m, 1024), 'tau_seed']:.2f}"
            expl = "n/a" if exact else f"{g.loc[(m, 1024), 'tau_explainer']:.2f}"
            share = "n/a" if exact else f"{100 * g.loc[(m, 1024), 'explainer_share']:.0f}\\%"
            rows.append(f"{NAME[m]} & {ctx} & {seed} & {expl} & {share} \\\\")
    _write(out, "tab_noise", "lcccccc",
           ["& \\multicolumn{3}{c}{Between contexts, $\\tau$} & \\multicolumn{2}{c}{Same context, $k=1024$} & "
            "$\\mathrm{Var}_{\\text{expl}}/\\mathrm{Var}_{\\text{ctx}}$ \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
            "Model & 128 & 512 & 1024 & other seed & other explainer seed & $k=1024$ \\\\"], rows)


def tab_specs(spec: pd.DataFrame, boot: pd.DataFrame, out: Path) -> None:
    terms = [("log2k", "$\\log_2 k$", "log2k"), ("z_d", "features $d$", "z_d"),
             ("z_mean_abs_corr", "mean $|\\rho|$", "z_mean_abs_corr"),
             ("z_imbalance", "imbalance", "z_imbalance"),
             ("C(model, Treatment('xgboost'))[T.tabpfn_v2_5]", "TabPFN-2.5", "tabpfn_v2_5"),
             ("C(model, Treatment('xgboost'))[T.tabicl_v2]", "TabICLv2", "tabicl_v2"),
             ("C(model, Treatment('xgboost'))[T.logreg]", "Log.\\ reg.", "logreg"),
             ("overlap", "overlap $k/n_{\\text{pool}}$", None),
             ("kernelshap", "KernelSHAP used", None)]
    specs = ["base", "adds k/n_pool", "adds KernelSHAP indicator"]
    rows = []
    for key, label, bkey in terms:
        cells = []
        for sp in specs:
            r = spec[(spec.spec == sp) & (spec.term == key)]
            cells.append(f"${r.coef.iloc[0]:+.3f}$ ({_p2(r.p.iloc[0])})" if len(r) else "--")
        if bkey:
            b = boot[boot.term == bkey].iloc[0]
            cells.append(f"$[{b.boot_lo:+.3f}, {b.boot_hi:+.3f}]$")
        else:
            cells.append("--")
        rows.append(label + " & " + " & ".join(cells) + " \\\\")
    _write(out, "tab_specs", "lcccc",
           ["Term & Base & Adds overlap & Adds estimator & Cluster bootstrap CI \\\\"], rows)


def tab_dissociation(diss: pd.DataFrame, logit: pd.DataFrame, out: Path) -> None:
    rows = []
    for m in MODELS:
        d, l = diss[diss.model == m].iloc[0], logit[logit.model == m].iloc[0]
        rows.append(f"{NAME[m]} & {d.pred_instability_ratio:.2f} & {d.attr_instability_ratio:.2f} & "
                    f"{d.datasets_pred_faster}/16 & {_p(d.p)} & {l.logit_sd_ratio:.2f} & {l.phi_sd_ratio:.2f} & "
                    f"{l.datasets_logit_faster}/16 & {_p(l.p)} \\\\")
    _write(out, "tab_dissociation", "lcccccccc",
           ["& \\multicolumn{4}{c}{Headline scales} & \\multicolumn{4}{c}{Common logit scale} \\\\",
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
            "Model & SD $p$ & $1-\\tau$ & faster & $p$ & SD logit & SD $\\phi$ & faster & $p$ \\\\"], rows)


def tab_bagging(back: pd.DataFrame, need: pd.DataFrame, out: Path) -> None:
    g = back.groupby(["model", "k"])[["tau_m1", "tau_m4", "tau_m5", "needed_short", "needed_full", "err_m5"]].median()
    n = need.groupby(["model", "k"]).agg(reached=("extrapolated", lambda s: int((~s).sum())),
                                         gpu=("gpu_seconds", lambda s: float(np.median(s)) / 60))
    rows = []
    for m in MODELS:
        for k in (128, 1024):
            r = g.loc[(m, k)]
            rows.append(f"{NAME[m] if k == 128 else ''} & {k} & {r.tau_m1:.2f} & {r.tau_m4:.2f} & {r.tau_m5:.2f} & "
                        f"${r.err_m5:+.2f}$ & {r.needed_full:,.0f} & {r.needed_short:,.0f} & {n.loc[(m, k), 'reached']}/16 \\\\")
    _write(out, "tab_bagging", "lrcccccccc"[:9],
           ["& & \\multicolumn{3}{c}{Observed $\\tau$} & Backtest & \\multicolumn{2}{c}{$M$ for $\\tau\\ge0.9$} & Reached \\\\",
            "\\cmidrule(lr){3-5}\\cmidrule(lr){7-8}",
            "Model & $k$ & $M{=}1$ & $M{=}4$ & $M{=}5$ & error at 5 & all fits & $M\\le3$ fit & by $M{=}5$ \\\\"], rows)


def tab_ablations(abl: pd.DataFrame, rem: pd.DataFrame, out: Path) -> None:
    t = abl[abl.metric == "tau_local"]
    rows = []
    def row(label, key, model, ks=(128, 1024)):
        cells = []
        for k in ks:
            r = t[(t.ablation == key) & (t.model == model) & (t.k == k)].iloc[0]
            cells.append(f"${r.ref_median:.2f}$ & ${r.abl_median:.2f}$ & {_s(r.median_diff)} & {_p(r.p)}")
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    for key, label in [("ensemble1", "$n_{\\text{est}}=1$"), ("ensemble16", "$n_{\\text{est}}=16$"),
                       ("stratified", "Stratified contexts")]:
        for m in TFMS:
            row(f"{label}, {NAME[m]}", key, m)
    row("TabPFN v2 vs 2.5", "version", "tabpfn_v2")
    for m in TFMS:
        row(f"Grouped vs summed, {NAME[m]}", "grouped_vs_posthoc_sum", m)
    _write(out, "tab_ablations", "lrrrrrrrr",
           ["& \\multicolumn{4}{c}{$k=128$} & \\multicolumn{4}{c}{$k=1024$} \\\\",
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
            "Condition & ref. & new & $\\Delta\\tau$ & $p$ & ref. & new & $\\Delta\\tau$ & $p$ \\\\"], rows)
    agg = rem.groupby(["model", "k"])[["single_ctx_nest4", "single_ctx_nest16", "bag4_ctx_nest4", "sec_nest16", "sec_bag4"]].median()
    rows = []
    for m in TFMS:
        for k in (128, 1024):
            r = agg.loc[(m, k)]
            rows.append(f"{NAME[m] if k == 128 else ''} & {k} & {r.single_ctx_nest4:.2f} & {r.single_ctx_nest16:.2f} & "
                        f"{r.bag4_ctx_nest4:.2f} & {r.sec_nest16:.0f} & {r.sec_bag4:.0f} \\\\")
    _write(out, "tab_remedy", "lrccccc",
           ["& & \\multicolumn{3}{c}{Local $\\tau$} & \\multicolumn{2}{c}{Seconds per estimate} \\\\",
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
            "Model & $k$ & one context & larger ensemble & bag of 4 & larger ensemble & bag of 4 \\\\"], rows)


def tab_oracle(o: pd.DataFrame, out: Path) -> None:
    g = o.groupby(["model", "k"])[["oracle_tau_single", "oracle_tau_bagged", "var_share_debiased",
                                   "cover_bagged_ci", "cover_single_range"]].median()
    rows = []
    for m in MODELS:
        for k in (128, 1024):
            r = g.loc[(m, k)]
            rows.append(f"{NAME[m] if k == 128 else ''} & {k} & {r.oracle_tau_single:.2f} & {r.oracle_tau_bagged:.2f} & "
                        f"{100 * r.var_share_debiased:.0f}\\% & {100 * r.cover_bagged_ci:.0f}\\% & {100 * r.cover_single_range:.0f}\\% \\\\")
    _write(out, "tab_oracle", "lrccccc",
           ["& & \\multicolumn{2}{c}{$\\tau$ with oracle} & Error from & \\multicolumn{2}{c}{Coverage of the truth} \\\\",
            "\\cmidrule(lr){3-4}\\cmidrule(lr){6-7}",
            "Model & $k$ & one context & mean of 10 & variance & bootstrap CI & range \\\\"], rows)


def tab_shallow(sh: pd.DataFrame, out: Path) -> None:
    if not len(sh):
        return
    rows = []
    for k in (128, 1024):
        for m in MODELS:
            r = sh[(sh.k == k) & (sh.model == m)].iloc[0]
            rows.append(f"{k if m == MODELS[0] else ''} & {NAME[m]} & {r.tau_model:.2f} & {_s(r.median_diff)} & {r.wins}/{r.n} & "
                        f"{_p(r.p)} & {r.logloss_model:.2f} \\\\")
        r = sh[sh.k == k].iloc[0]
        rows.append(f" & XGBoost, depth 2 & {r.tau_shallow:.2f} & -- & -- & -- & {r.logloss_shallow:.2f} \\\\")
        if k == 128:
            rows.append("\\midrule")
    _write(out, "tab_shallow", "rlccccc",
           ["$k$ & Model & Local $\\tau$ & $\\Delta\\tau$ vs shallow & wins & $p$ & Log-loss \\\\"], rows)


def tab_datasets(cfg: dict, traits: pd.DataFrame, s: pd.DataFrame, out: Path) -> None:
    ids = {d["name"]: d["openml_id"] for d in cfg["datasets"]}
    piv = s[s.k == 1024].pivot(index="dataset", columns="model", values="tau_local")
    rows = []
    for r in traits.sort_values(["d", "dataset"]).itertuples():
        est = "exact" if r.d <= 8 else "Kernel"
        taus = " & ".join(f"{piv.loc[r.dataset, m]:.2f}" for m in MODELS)
        rows.append(f"\\textsf{{{r.dataset}}} & {ids[r.dataset]} & {r.d} & {r.n_classes} & {r.n_pool:,} & "
                    f"{r.mean_abs_corr:.2f} & {r.imbalance:.2f} & {est} & {taus} \\\\")
    _write(out, "tab_datasets", "lrrrrrrl" + "cccc",
           ["& & & & & & & & \\multicolumn{4}{c}{Local $\\tau$, $k=1024$} \\\\",
            "\\cmidrule(lr){9-12}",
            "Dataset & ID & $d$ & cl. & $n_{\\text{pool}}$ & $|\\rho|$ & imb. & $\\phi$ & "
            "TabPFN & TabICL & XGB & LR \\\\"], rows)


def tab_secondary(t: pd.DataFrame, s: pd.DataFrame, out: Path) -> None:
    labels = {"tau_global": "Global $\\tau$", "jaccard_top3": "Top-3 Jaccard", "sign_flip": "Sign-flip rate"}
    rows = []
    med = s.groupby(["model", "k"])[list(labels)].median()
    for metric, label in labels.items():
        for m in MODELS:
            vals = " & ".join(f"{med.loc[(m, k), metric]:.2f}" for k in (128, 512, 1024))
            tests = ""
            if m in TFMS:
                g = t[(t.metric == metric) & (t.tfm == m)]
                wins = lambda b: "/".join(str(g[(g.baseline == b) & (g.k == k)].wins.iloc[0]) for k in (128, 512, 1024))  # noqa: E731
                tests = f"{wins('xgboost')} & {wins('logreg')}"
            else:
                tests = "& "
            rows.append(f"{label if m == MODELS[0] else ''} & {NAME[m]} & {vals} & {tests} \\\\")
        if metric != "sign_flip":
            rows.append("\\midrule")
    _write(out, "tab_secondary", "llccccc",
           ["Metric & Model & 128 & 512 & 1024 & higher than XGB & higher than LR \\\\"], rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/main")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--out", default="results/paper_assets")
    args = ap.parse_args()
    root = Path(args.main)
    res, rob = root / "analysis", root / "analysis" / "robustness"
    figs, tabs = Path(args.out) / "figures", Path(args.out) / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(root / "summary.csv")
    vr = pd.read_csv(res / "h1_variance_ratio.csv")
    close = pd.read_csv(rob / "close_predictions.csv")
    curves = pd.read_csv(res / "h4_bagging_curves.csv")
    o = pd.read_csv(rob / "oracle_debiased.csv")

    print(fig_example(root, args.config, figs))
    fig_vr(vr, figs)
    fig_stability(s, close, figs)
    fig_bagging(curves, figs)
    fig_oracle(o, figs)
    fig_datasets(s, pd.read_csv(res / "dataset_traits.csv"), figs)
    shallow_runs = root.parent / "ablation_xgb_shallow" / "summary.csv"
    fig_tradeoff(s, pd.read_csv(shallow_runs) if shallow_runs.exists() else pd.DataFrame(), figs)
    fig_remedy(pd.read_csv(res / "remedy.csv"), figs)

    tab_overview(s, close, tabs)
    tab_vr(vr, tabs)
    tests = pd.read_csv(res / "h2_matched_k_tests.csv")
    tab_h2(tests, pd.read_csv(rob / "h2_all12.csv"), tabs)
    tab_noise(pd.read_csv(rob / "noise_ceilings.csv"), tabs)
    tab_specs(pd.read_csv(rob / "model_specs.csv"), pd.read_csv(rob / "model_bootstrap.csv"), tabs)
    tab_dissociation(pd.read_csv(res / "h3_dissociation.csv"), pd.read_csv(rob / "dissociation_logit.csv"), tabs)
    tab_bagging(pd.read_csv(rob / "bagging_backtest.csv"), pd.read_csv(res / "h4_contexts_needed.csv"), tabs)
    tab_ablations(pd.read_csv(res / "ablations.csv"), pd.read_csv(res / "remedy.csv"), tabs)
    tab_oracle(o, tabs)
    shallow = rob / "xgb_shallow.csv"
    tab_shallow(pd.read_csv(shallow) if shallow.exists() else pd.DataFrame(), tabs)
    cfg = load_config(args.config)
    tab_datasets(cfg, pd.read_csv(res / "dataset_traits.csv"), s, tabs)
    tab_secondary(tests, s, tabs)


if __name__ == "__main__":
    main()
