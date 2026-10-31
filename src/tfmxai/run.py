"""Grid runner. One run = (dataset, model, k, context, model seed, explainer seed).

Each run writes results/raw/<dataset>/<model>/k<k>/c<ctx>_s<seed>_e<eseed>.npz and is
skipped if that file exists, so the grid can be stopped (Ctrl+C, --hours) and resumed at
any point by rerunning the same command. Runs are ordered context-major: every dataset
finishes context c before any starts c+1, so a stopped grid is still balanced.

    python -m tfmxai.run --config configs/pilot.yaml
    python -m tfmxai.run --config configs/main.yaml --models xgboost logreg --hours 6
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

from tfmxai.contexts import context_rng, sample_context
from tfmxai.data import load_dataset
from tfmxai.models import make_model
from tfmxai.shapley import explain
from tfmxai.values import coalition_values

warnings.filterwarnings("ignore")


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["datasets"] = yaml.safe_load(Path(cfg["datasets"]).read_text())["datasets"]
    return cfg


def run_path(root: Path, ds: str, model: str, k: int, ctx: int, seed: int, eseed: int) -> Path:
    return root / "raw" / ds / model / f"k{k}" / f"c{ctx:02d}_s{seed}_e{eseed}.npz"


def replicates_for(cfg: dict, ctx: int, exact: bool) -> list[tuple[int, int]]:
    """(model seed, explainer seed) pairs for one context. Every context gets one run; the
    first `replicate_contexts` also get the other model seeds (model noise) and, for
    KernelSHAP, the other explainer seeds (estimator noise). Exact enumeration has none."""
    s0, e0 = cfg["model_seeds"][0], cfg["explainer_seeds"][0]
    pairs = [(s0, e0)]
    if ctx < cfg["replicate_contexts"]:
        pairs += [(s, e0) for s in cfg["model_seeds"][1:]]
        if not exact:
            pairs += [(s0, e) for e in cfg["explainer_seeds"][1:]]
    return pairs


def save_oracle(ds, cfg: dict, root: Path) -> None:
    path = root / "raw" / ds.name / "oracle.npz"
    if ds.oracle is None or path.exists():
        return
    rng = np.random.default_rng(0)
    phi, v_full, exact, _ = explain(lambda m: ds.oracle(ds.X_explain, ds.y_explain, m), ds.d,
                                    exact_if_d_le=cfg["sv"]["exact_if_d_le"],
                                    budget=cfg["sv"]["oracle_budget"], rng=rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, phi=phi, v_full=v_full, exact=exact)


def save_atomic(path: Path, **arrays) -> None:
    # A run killed mid-write must not leave a file that resume would treat as done.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def keep_awake() -> None:
    if sys.platform == "win32":  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED; released on exit
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", help="subset of config models")
    ap.add_argument("--datasets", nargs="*", help="subset of config dataset names")
    ap.add_argument("--hours", type=float, help="stop cleanly after this many hours")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["results_dir"])
    models = args.models or cfg["models"]
    specs = [s for s in cfg["datasets"] if not args.datasets or s["name"] in args.datasets]
    datasets = [load_dataset(s, n_explain=cfg["n_explain"]) for s in specs]
    root.mkdir(parents=True, exist_ok=True)
    (root / "last_config.json").write_text(json.dumps(cfg, indent=2))
    for ds in datasets:
        save_oracle(ds, cfg, root)

    plan = [(ctx, ds, k, name, seed, eseed)
            for ctx in range(cfg["n_contexts"])
            for ds in datasets
            for k in cfg["k"] if k <= len(ds.y_pool)
            for name in models
            for seed, eseed in replicates_for(cfg, ctx, ds.d <= cfg["sv"]["exact_if_d_le"])]
    todo = [(ctx, ds, k, name, seed, eseed) for ctx, ds, k, name, seed, eseed in plan
            if not run_path(root, ds.name, name, k, ctx, seed, eseed).exists()]
    print(f"{len(plan) - len(todo)}/{len(plan)} runs already done, {len(todo)} to go")

    timing_log = root / f"timing_{'+'.join(models)}.csv"
    new_log = not timing_log.exists()
    deadline = time.time() + args.hours * 3600 if args.hours else float("inf")
    keep_awake()
    done = 0
    try:
        with timing_log.open("a", newline="") as fh:
            log = csv.writer(fh)
            if new_log:
                log.writerow(["dataset", "model", "k", "context", "seed", "eseed", "d",
                              "n_coalitions", "exact", "seconds"])
            for ctx, ds, k, name, seed, eseed in todo:
                if time.time() > deadline:
                    print(f"--hours {args.hours} reached; rerun the same command to resume")
                    break
                idx = sample_context(ds.y_pool, k, cfg["sampling"], context_rng(ds.name, k, ctx))
                X_ctx, y_ctx = ds.X_pool[idx], ds.y_pool[idx]
                model = make_model(name, seed, device=cfg["device"],
                                   n_estimators=cfg["n_estimators"], cpu_jobs=cfg["cpu_jobs"])
                t0 = time.perf_counter()
                phi, v_full, is_exact, n_eval = explain(
                    lambda masks: coalition_values(model, X_ctx, y_ctx, ds.X_explain,
                                                   ds.y_explain, masks, ds.n_classes),
                    ds.d, exact_if_d_le=cfg["sv"]["exact_if_d_le"], budget=cfg["sv"]["budget"],
                    rng=np.random.default_rng([seed, eseed, ctx, k]))
                secs = time.perf_counter() - t0
                save_atomic(run_path(root, ds.name, name, k, ctx, seed, eseed), phi=phi,
                            v_full=v_full, ctx_idx=idx, exact=is_exact, seconds=secs)
                log.writerow([ds.name, name, k, ctx, seed, eseed, ds.d, n_eval, is_exact,
                              round(secs, 2)])
                fh.flush()
                done += 1
                print(f"[{len(plan) - len(todo) + done}/{len(plan)}] {ds.name:>18} {name:>11} "
                      f"k={k:<5} c{ctx:02d} s{seed} e{eseed} d={ds.d:<3} {n_eval:>5} coal "
                      f"{secs:7.1f}s", flush=True)
    except KeyboardInterrupt:
        print("\nstopped; finished runs are saved, rerun the same command to resume")


if __name__ == "__main__":
    main()
