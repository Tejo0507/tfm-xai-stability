"""Grid runner. One run = (dataset, model, k, context, model seed, explainer seed).

Each run writes results/raw/<dataset>/<model>/k<k>/c<ctx>_s<seed>_e<eseed>.npz and is
skipped if that file exists, so the grid can be stopped and resumed at any point.

    python -m tfmxai.run --config configs/pilot.yaml
    python -m tfmxai.run --config configs/main.yaml --models xgboost logreg   # CPU process
"""

import argparse
import csv
import json
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


def explainer_seeds_for(cfg: dict, ctx: int, exact: bool) -> list[int]:
    # Exact enumeration has no sampling noise; extra explainer seeds only for KernelSHAP.
    seeds = cfg["explainer_seeds"]
    return seeds[:1] if exact or ctx >= cfg["explainer_seed_contexts"] else seeds


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", help="subset of config models")
    ap.add_argument("--datasets", nargs="*", help="subset of config dataset names")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["results_dir"])
    models = args.models or cfg["models"]
    specs = [s for s in cfg["datasets"] if not args.datasets or s["name"] in args.datasets]
    timing_log = root / "timing.csv"
    timing_log.parent.mkdir(parents=True, exist_ok=True)
    new_log = not timing_log.exists()

    with timing_log.open("a", newline="") as fh:
        log = csv.writer(fh)
        if new_log:
            log.writerow(["dataset", "model", "k", "context", "seed", "eseed", "d",
                          "n_coalitions", "exact", "seconds"])
        for spec in specs:
            ds = load_dataset(spec, n_explain=cfg["n_explain"])
            save_oracle(ds, cfg, root)
            exact = ds.d <= cfg["sv"]["exact_if_d_le"]
            for k in cfg["k"]:
                if k > len(ds.y_pool):
                    print(f"[skip] {ds.name}: k={k} > pool {len(ds.y_pool)}")
                    continue
                for ctx in range(cfg["n_contexts"]):
                    idx = sample_context(ds.y_pool, k, cfg["sampling"], context_rng(ds.name, k, ctx))
                    X_ctx, y_ctx = ds.X_pool[idx], ds.y_pool[idx]
                    for name in models:
                        for seed in cfg["model_seeds"]:
                            todo = [e for e in explainer_seeds_for(cfg, ctx, exact)
                                    if not run_path(root, ds.name, name, k, ctx, seed, e).exists()]
                            if not todo:
                                continue
                            model = make_model(name, seed, device=cfg["device"],
                                               n_estimators=cfg["n_estimators"],
                                               cpu_jobs=cfg["cpu_jobs"])
                            value_fn = lambda masks: coalition_values(  # noqa: E731
                                model, X_ctx, y_ctx, ds.X_explain, ds.y_explain, masks,
                                ds.n_classes)
                            for eseed in todo:
                                t0 = time.perf_counter()
                                phi, v_full, is_exact, n_eval = explain(
                                    value_fn, ds.d, exact_if_d_le=cfg["sv"]["exact_if_d_le"],
                                    budget=cfg["sv"]["budget"],
                                    rng=np.random.default_rng([seed, eseed, ctx, k]))
                                secs = time.perf_counter() - t0
                                out = run_path(root, ds.name, name, k, ctx, seed, eseed)
                                out.parent.mkdir(parents=True, exist_ok=True)
                                np.savez_compressed(out, phi=phi, v_full=v_full, ctx_idx=idx,
                                                    exact=is_exact, seconds=secs)
                                log.writerow([ds.name, name, k, ctx, seed, eseed, ds.d, n_eval,
                                              is_exact, round(secs, 2)])
                                fh.flush()
                                print(f"{ds.name:>18} {name:>11} k={k:<5} c{ctx:02d} s{seed} "
                                      f"e{eseed} d={ds.d:<3} {n_eval:>5} coal {secs:7.1f}s")
    (root / "last_config.json").write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
