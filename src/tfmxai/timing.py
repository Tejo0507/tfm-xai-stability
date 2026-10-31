"""Estimate grid cost without running it: time a few coalition refits per
(dataset, model, k) and extrapolate to the full config.

    python -m tfmxai.timing --config configs/main.yaml --probe 12
"""

import argparse
import time
import warnings

import numpy as np

from tfmxai.contexts import context_rng, sample_context
from tfmxai.data import load_dataset
from tfmxai.models import GPU_MODELS, make_model
from tfmxai.run import explainer_seeds_for, load_config
from tfmxai.values import coalition_values

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--probe", type=int, default=12, help="coalitions timed per cell")
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config)
    models = args.models or cfg["models"]
    rng = np.random.default_rng(0)
    hours = {"gpu": 0.0, "cpu": 0.0}

    print(f"{'dataset':>18} {'n_pool':>7} {'d':>3} {'C':>3} {'model':>11} {'k':>5} "
          f"{'s/coal':>7} {'s/run':>7} {'runs':>5}")
    for spec in cfg["datasets"]:
        ds = load_dataset(spec, n_explain=cfg["n_explain"])
        exact = ds.d <= cfg["sv"]["exact_if_d_le"]
        n_coal = 2**ds.d if exact else cfg["sv"]["budget"]
        eseeds = sum(len(explainer_seeds_for(cfg, c, exact)) for c in range(cfg["n_contexts"]))
        runs = eseeds * len(cfg["model_seeds"])
        for name in models:
            model = make_model(name, 0, device=cfg["device"], n_estimators=cfg["n_estimators"],
                               cpu_jobs=cfg["cpu_jobs"])
            for k in cfg["k"]:
                if k > len(ds.y_pool):
                    continue
                idx = sample_context(ds.y_pool, k, cfg["sampling"], context_rng(ds.name, k, 0))
                masks = rng.random((args.probe + 1, ds.d)) < 0.5
                masks[:, 0] = True  # avoid empty coalitions, which cost nothing
                fn = lambda m: coalition_values(model, ds.X_pool[idx], ds.y_pool[idx],  # noqa: E731
                                                ds.X_explain, ds.y_explain, m, ds.n_classes)
                fn(masks[:1])  # warm-up (CUDA init, checkpoint load)
                t0 = time.perf_counter()
                fn(masks[1:])
                per = (time.perf_counter() - t0) / args.probe
                per_run = per * n_coal
                hours["gpu" if name in GPU_MODELS else "cpu"] += per_run * runs / 3600
                print(f"{ds.name:>18} {len(ds.y_pool):>7} {ds.d:>3} {ds.n_classes:>3} "
                      f"{name:>11} {k:>5} {per:7.3f} {per_run:7.1f} {runs:>5}")
    print(f"\nEstimated GPU hours: {hours['gpu']:.1f}   CPU hours: {hours['cpu']:.1f} "
          f"(CPU baselines can run in a parallel process)")


if __name__ == "__main__":
    main()
