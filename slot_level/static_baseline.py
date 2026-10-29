"""Static-provisioning baseline: keep a FIXED number of hosts awake (no forecast, no reaction).

Why: at long wake-up times the forecast rules on some traces end up keeping a roughly constant share of hosts awake.
If a fixed n does as well, the forecast is not what produces the gain. This script checks it fairly:
  - for every wake-up time and every n = 1..H it evaluates "always keep n hosts awake" on the TEST episodes;
  - the n a practitioner could really choose is picked on the TRAIN part of the trace (separate episodes), then reported on test;
  - the reactive rule is evaluated on the same test episodes so gains are paired.

  python static_baseline.py --trace bb_fast.npy --wakes 120 300 600 --episodes 30 --out static_bbfast
Writes <out>/static_episodes.csv and <out>/static_summary.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from baselines import SLOT_BASELINES, SlotRule, run_episode
from cloud_env import SlotPowerEnv
from wake_sweep import split_trace


def make_env(trace, W, a):
    return SlotPowerEnv(n_tasks=a.n_tasks, forecast_mode="none", load_trace=trace, trace_prescaled=True,
                        forecaster=None, wake_s=float(W), horizon_k=max(1, int(round(W / a.slot_s))), slot_s=a.slot_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--wakes", nargs="+", type=float, default=[120, 300, 600])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--n-tasks", type=int, default=1000)
    ap.add_argument("--slot-s", type=float, default=10.0)
    ap.add_argument("--out", default="static")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train, test = split_trace(a.trace)
    rows = []
    for W in a.wakes:
        envs = {"test": make_env(test, W, a), "train": make_env(train, W, a)}
        H = envs["test"].n_hosts
        for part, base_seed in (("test", 100_000), ("train", 200_000)):
            env = envs[part]
            pols = {f"static_{n}": SlotRule(lambda e, o, n=n: n) for n in range(1, H + 1)}
            pols["reactive"] = SLOT_BASELINES["reactive"][1]()
            for name, pol in pols.items():
                for e in range(a.episodes):
                    info = run_episode(env, pol, seed=base_seed + e)
                    rows.append({"wake_s": W, "part": part, "method": name, "episode": e, **info})
        print(f"[wake {W:g}s] done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "static_episodes.csv"), index=False)

    out = []
    for W in a.wakes:
        d = df[df.wake_s == W]
        tr = d[d.part == "train"].groupby("method").episode_return.mean()
        best = tr.drop("reactive").idxmax()                     # n chosen on the train part only
        te = d[d.part == "test"].pivot(index="episode", columns="method", values="episode_return")
        sla = d[d.part == "test"].groupby("method").sla_violation_rate.mean() * 100
        en = d[d.part == "test"].groupby("method").energy_kwh.mean()
        print(f"\n=== wake {W:g}s: fixed number of awake hosts (test part, {a.episodes} episodes) ===")
        print(f"{'rule':10s} {'return':>9s} {'SLA %':>7s} {'kWh':>7s} {'gain vs reactive [95% CI]':>32s}")
        for m in [c for c in te.columns if c != "reactive"] + ["reactive"]:
            diff = (te[m] - te["reactive"]).to_numpy()
            ci = stats.t.interval(0.95, len(diff) - 1, diff.mean(), stats.sem(diff)) if m != "reactive" and diff.std() > 0 else (0, 0)
            tag = "  <- chosen on TRAIN" if m == best else ""
            print(f"{m:10s} {te[m].mean():9.0f} {sla[m]:7.2f} {en[m]:7.3f} {diff.mean():+9.0f} [{ci[0]:+.0f}, {ci[1]:+.0f}]{tag}")
        diff = (te[best] - te["reactive"]).to_numpy()
        ci = stats.t.interval(0.95, len(diff) - 1, diff.mean(), stats.sem(diff))
        out.append({"wake_s": W, "train_chosen": best, "test_return": te[best].mean(), "gain_vs_reactive": diff.mean(),
                    "ci_lo": ci[0], "ci_hi": ci[1], "sla_pct": sla[best], "energy_kwh": en[best],
                    "episodes_better": float((diff > 0).mean())})
    pd.DataFrame(out).round(3).to_csv(os.path.join(a.out, "static_summary.csv"), index=False)
    print("\n=== static rule chosen on the train part, evaluated on test ===")
    print(pd.DataFrame(out).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
