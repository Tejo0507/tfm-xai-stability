"""Figure 1: one explained point, two contexts, near-identical predictions, different
attributions. The pair is chosen automatically and the choice rule is printed with it.

    python -m tfmxai.example_figure --results results/main --dataset adult --model tabpfn_v2_5 --k 512
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np

from tfmxai.analysis import COLOR, LABEL
from tfmxai.data import load_dataset
from tfmxai.metrics import kendall_rows
from tfmxai.run import load_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def pick(phis: np.ndarray, v: np.ndarray, max_dv: float) -> tuple[int, int, int, float]:
    """(point, context a, context b, tau) whose full-model logits differ by at most max_dv
    (the scale the attributions are on), with a different top feature and the lowest |phi|
    rank agreement among such pairs. This is the extreme case by construction."""
    best = None
    for a, b in combinations(range(len(phis)), 2):
        tau = kendall_rows(np.abs(phis[a]), np.abs(phis[b]))
        ok = ((np.abs(v[a] - v[b]) <= max_dv)
              & (np.abs(phis[a]).argmax(1) != np.abs(phis[b]).argmax(1)) & ~np.isnan(tau))
        for i in np.flatnonzero(ok):
            if best is None or tau[i] < best[3]:
                best = (int(i), a, b, float(tau[i]))
    if best is None:
        raise SystemExit(f"no pair with |delta logit| <= {max_dv} and different top features")
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--dataset", default="adult")
    ap.add_argument("--model", default="tabpfn_v2_5")
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--max-dv", type=float, default=0.1,
                    help="max difference in the logit of the true class")
    ap.add_argument("--top", type=int, default=8, help="features shown")
    args = ap.parse_args()

    cell = Path(args.results) / "raw" / args.dataset / args.model / f"k{args.k}"
    runs = [np.load(f) for f in sorted(cell.glob("c*_s0_e0.npz"))]
    phis = np.stack([r["phi"] for r in runs])
    v = np.stack([r["v_full"] for r in runs])
    p = 1 / (1 + np.exp(-v))
    i, a, b, tau = pick(phis, v, args.max_dv)
    typical = np.nanmedian([np.nanmedian(kendall_rows(np.abs(phis[x]), np.abs(phis[y])))
                            for x, y in combinations(range(len(phis)), 2)])
    spec = next(s for s in load_config(args.config)["datasets"] if s["name"] == args.dataset)
    names = np.array(load_dataset(spec).feature_names)

    order = np.argsort(-np.abs(phis[[a, b], i]).max(0))[: args.top][::-1]
    lim = np.abs(phis[[a, b], i][:, order]).max() * 1.1
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.42 * len(order) + 1.4), sharey=True)
    for ax, c in zip(axes, (a, b)):
        vals = phis[c, i, order]
        ax.barh(range(len(order)), vals, color=np.where(vals >= 0, COLOR[args.model], "#c0504d"))
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_title(f"context {c}: p(true class) = {p[c, i]:.3f}", fontsize=10)
        ax.set_xlabel("Shapley value (logit of true class)")
    axes[0].set_yticks(range(len(order)), names[order], fontsize=9)
    fig.suptitle(f"{LABEL[args.model]}, {args.dataset}, k = {args.k}: same point, two random "
                 f"contexts (Kendall τ of |φ| ranks = {tau:.2f})", fontsize=10)
    fig.tight_layout()
    out = Path(args.results) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"fig1_{args.dataset}_{args.model}_k{args.k}.png"
    fig.savefig(path, dpi=200)
    print(f"point {i}, contexts {a} vs {b}: |d logit| = {abs(v[a, i] - v[b, i]):.3f}, "
          f"tau = {tau:.3f} (lowest tau among {len(phis)} contexts with |d logit| <= "
          f"{args.max_dv} and different top features; median tau over all points and pairs "
          f"= {typical:.3f}) -> {path}")


if __name__ == "__main__":
    main()
