"""List grid runs that evaluate a coalition whose features are all constant in the context.

Contexts and KernelSHAP coalitions do not depend on the model, so the list is shared by
every model. Runs written before the constant-coalition rule in `values.py` must be redone.

    python scripts/audit_constant_coalitions.py --config configs/main.yaml
"""

import argparse

import numpy as np

from tfmxai.contexts import context_rng, sample_context
from tfmxai.data import load_dataset
from tfmxai.run import load_config, replicates_for
from tfmxai.shapley import explain
from tfmxai.values import varying_columns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    for spec in cfg["datasets"]:
        ds = load_dataset(spec, n_explain=cfg["n_explain"])
        exact = ds.d <= cfg["sv"]["exact_if_d_le"]
        for k in cfg["k"]:
            if k > len(ds.y_pool):
                continue
            for ctx in range(cfg["n_contexts"]):
                idx = sample_context(ds.y_pool, k, cfg["sampling"], context_rng(ds.name, k, ctx))
                varying = varying_columns(ds.X_pool[idx])
                if varying.all():
                    continue
                for seed, eseed in replicates_for(cfg, ctx, exact):
                    hit = []

                    def record(masks):
                        hit.append(int(((masks & varying).sum(1) == 0)[1:].sum()))
                        return np.zeros((len(masks), 1))

                    # rng matches run.py; masks[0] (or all_masks row 0) is the empty set
                    explain(record, ds.d, exact_if_d_le=cfg["sv"]["exact_if_d_le"],
                            budget=cfg["sv"]["budget"], rng=np.random.default_rng([eseed, ctx, k]))
                    if hit[0]:
                        print(f"{ds.name},{k},{ctx},{seed},{eseed},{hit[0]}")


if __name__ == "__main__":
    main()
