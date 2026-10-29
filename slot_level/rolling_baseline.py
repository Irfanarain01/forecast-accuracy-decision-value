"""Adaptive-static baseline without any ML: keep enough hosts awake for the q-th percentile of the load seen so far.

  n = hosts_needed( max(current load, percentile_q(load of the last L slots of the episode)) )

This mirrors the forecast rule (max(current load, forecast upper bound)) but replaces the forecast by a rolling
percentile of the OBSERVED load. (q, L) are chosen on TRAIN-part episodes only, then reported on the test episodes.
  python rolling_baseline.py --trace ali_cpu.npy --wakes 120 300 600 --episodes 30 --out rolling_ali
Writes <out>/rolling_episodes.csv and <out>/rolling_summary.csv.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

from baselines import SLOT_BASELINES, SlotRule, run_episode
from cloud_env import SlotPowerEnv
from wake_sweep import split_trace

QS = [0.5, 0.75, 0.9, 0.95, 0.99]
LS = [50, 100, 400]


def make_rule(q, L):
    def fn(env, obs):
        slot = env._slot
        h = env.loads[max(0, slot - L):slot]
        cur = env.current_load()
        p = float(np.quantile(h, q)) if len(h) >= 10 else cur
        return env.hosts_needed(max(cur, p))
    return fn


def work(args):
    trace_path, W, a_n_tasks, slot_s, episodes = args
    train, test = split_trace(trace_path)
    rows = []
    for part, tr, base_seed in (("test", test, 100_000), ("train", train, 200_000)):
        env = SlotPowerEnv(n_tasks=a_n_tasks, forecast_mode="none", load_trace=tr, trace_prescaled=True, forecaster=None,
                           wake_s=float(W), horizon_k=max(1, int(round(W / slot_s))), slot_s=slot_s)
        pols = {f"roll_q{q}_L{L}": SlotRule(make_rule(q, L)) for q in QS for L in LS}
        pols["reactive"] = SLOT_BASELINES["reactive"][1]()
        for name, pol in pols.items():
            for e in range(episodes):
                info = run_episode(env, pol, seed=base_seed + e)
                rows.append({"wake_s": W, "part": part, "method": name, "episode": e, **info})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--wakes", nargs="+", type=float, default=[120, 300, 600])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--n-tasks", type=int, default=1000)
    ap.add_argument("--slot-s", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="rolling")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a.trace, W, a.n_tasks, a.slot_s, a.episodes) for W in a.wakes]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for part in ex.map(work, jobs) for r in part]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "rolling_episodes.csv"), index=False)
    R = "episode_return"
    out = []
    for W in a.wakes:
        d = df[df.wake_s == W]
        trm = d[d.part == "train"].groupby("method")[R].mean()
        ch = trm.drop("reactive").idxmax()
        te = d[d.part == "test"].pivot(index="episode", columns="method", values=R)
        sla = d[d.part == "test"].groupby("method").sla_violation_rate.mean() * 100
        en = d[d.part == "test"].groupby("method").energy_kwh.mean()
        best = te.drop(columns="reactive").mean().idxmax()
        diff = (te[ch] - te["reactive"]).to_numpy()
        ci = stats.t.interval(0.95, len(diff) - 1, diff.mean(), stats.sem(diff))
        out.append({"wake_s": W, "train_chosen": ch, "test_return": te[ch].mean(), "gain_vs_reactive": diff.mean(), "ci_lo": ci[0], "ci_hi": ci[1],
                    "sla_pct": sla[ch], "energy_kwh": en[ch], "hindsight_best": best, "hindsight_return": te[best].mean(),
                    "reactive_return": te["reactive"].mean()})
        print(f"\n=== wake {W:g}s: rolling-percentile grid (test part) ===")
        print(te.mean().sort_values(ascending=False).round(0).head(8).to_string())
    print("\n=== rolling rule chosen on the train part, evaluated on test ===")
    print(pd.DataFrame(out).round(2).to_string(index=False))
    pd.DataFrame(out).round(3).to_csv(os.path.join(a.out, "rolling_summary.csv"), index=False)


if __name__ == "__main__":
    main()
