"""
Train and evaluate PPO schedulers on CloudSchedEnv, with baselines and statistics.

Examples
--------
# quick smoke test (about a minute)
python train_ppo.py --modes none quantile --seeds 0 1 --timesteps 20000 --eval-episodes 5

# full run: four variants, 5 seeds
python train_ppo.py --modes none point quantile oracle --seeds 0 1 2 3 4 \
       --timesteps 300000 --eval-episodes 30 --out results

# with a real load trace (1-D .npy or .csv of CPU utilisation, chronological)
python train_ppo.py --trace google_cpu.npy --modes none quantile ...
"""
import argparse
import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from baselines import BASELINES, POWER_BASELINES, SLOT_BASELINES, run_episode
from cloud_env import CloudSchedEnv, SlotPowerEnv, scale_trace


def load_trace(path):
    if path is None:
        return None, None
    arr = np.load(path) if path.endswith(".npy") else np.loadtxt(path, delimiter=",")
    arr = np.asarray(arr, dtype=float).ravel()
    cut = int(0.8 * len(arr))              # chronological split: train on first 80%, test on last 20%
    tr, te = arr[:cut], arr[cut:]
    if len(te) < 450:
        print(f"WARNING: test part has only {len(te)} points; episodes need ~400. Use a longer trace.")
    return scale_trace(tr, tr), scale_trace(te, tr)   # both scaled with the TRAIN min/max


def make_env(args, mode, trace, drift=False, residual=False):
    fc = None
    if args.forecaster and mode in ("point", "quantile"):   # also used by the forecast-rule baselines
        from forecaster_cnn_lstm import QuantileForecaster
        fc = QuantileForecaster.load(args.forecaster)          # trained CNN-LSTM instead of the placeholder
    kw = dict(n_tasks=args.n_tasks, forecast_mode=mode, load_trace=trace, trace_prescaled=True, drift=drift,
              forecaster=fc, wake_s=args.wake_s, horizon_k=args.horizon_k)
    if args.control == "slot" and not args.no_power:
        return SlotPowerEnv(wake_penalty=args.wake_penalty, residual=residual, res_range=args.res_range, **kw)
    return CloudSchedEnv(power_mgmt=not args.no_power, **kw)


def evaluate_policy(env, policy_fn, episodes, base_seed):
    rows = []
    for e in range(episodes):
        rows.append(run_episode(env, policy_fn, seed=base_seed + e))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["none", "quantile"],
                    choices=["none", "point", "quantile", "oracle"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--timesteps", type=int, default=100_000, help="agent decisions; with --control slot one decision = one 10 s slot")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--n-tasks", type=int, default=1000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--trace", type=str, default=None)
    ap.add_argument("--drift", action="store_true", help="synthetic regime change in the test episodes")
    ap.add_argument("--forecaster", type=str, default=None, help="path to forecaster.pt from forecaster_cnn_lstm.py")
    ap.add_argument("--control", choices=["slot", "task"], default="slot",
                    help="slot: one power decision per time slot (recommended); task: flat per-task action")
    ap.add_argument("--wake-penalty", type=float, default=0.0,
                    help="training-only penalty per wake-up (slot control); reported returns exclude it")
    ap.add_argument("--residual", action="store_true",
                    help="PPO corrects the matching rule (reactive / forecast rule) by -R..+R hosts instead of choosing the count")
    ap.add_argument("--res-range", type=int, default=2, help="R: maximum correction in hosts (residual mode)")
    ap.add_argument("--res-bias", type=float, default=3.0,
                    help="initial logit bonus for 'no correction', so PPO starts at the rule (residual mode)")
    ap.add_argument("--wake-s", type=float, default=60.0, help="host wake-up time in seconds")
    ap.add_argument("--horizon-k", type=int, default=6, help="forecast horizon in 10 s slots (match forecaster --k)")
    ap.add_argument("--no-power", action="store_true", help="disable host sleep/wake (original environment)")
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_trace, test_trace = load_trace(args.trace)
    records = []

    # Baselines on the same held-out episodes (seeds 100000+), one pass.
    baselines = BASELINES if args.no_power else (SLOT_BASELINES if args.control == "slot" else POWER_BASELINES)
    for name, (bmode, ctor) in baselines.items():
        env = make_env(args, bmode, test_trace, args.drift)
        for r in evaluate_policy(env, ctor(), args.eval_episodes, 100_000):
            records.append({"method": name, "seed": -1, **r})
        print(f"baseline {name} done")

    for mode in args.modes:
        for seed in args.seeds:
            set_random_seed(seed)
            venv = make_vec_env(lambda: make_env(args, mode, train_trace, residual=args.residual), n_envs=args.n_envs, seed=seed)
            venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)   # scales returns only
            model = PPO("MlpPolicy", venv, seed=seed, verbose=0, n_steps=512, batch_size=256,
                        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                        ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]))
            if args.residual:                       # start from "no correction" = the rule itself
                import torch
                with torch.no_grad():
                    model.policy.action_net.bias[args.res_range] += args.res_bias
            model.learn(total_timesteps=args.timesteps)
            env = make_env(args, mode, test_trace, args.drift, residual=args.residual)

            def policy(e, obs, m=model):
                a = m.predict(obs, deterministic=True)[0]
                return int(a) if (args.no_power or args.control == "slot") else a

            for r in evaluate_policy(env, policy, args.eval_episodes, 100_000):
                records.append({"method": f"ppo_{mode}", "seed": seed, **r})
            print(f"ppo_{mode} seed {seed} done")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.out, "episodes.csv"), index=False)

    metrics = ["makespan_s", "energy_kwh", "sla_violation_rate", "mean_wait_s", "mean_util", "episode_return"]
    if not args.no_power:
        metrics += ["mean_awake_frac", "wake_events"]
    # per-seed mean first, so seeds (not episodes) are the unit of replication for PPO
    per_seed = df.groupby(["method", "seed"])[metrics].mean().reset_index()
    summary = per_seed.groupby("method")[metrics].agg(["mean", "std"]).round(4)
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    print("\n=== Summary (mean/std over seeds; baselines are deterministic over eval episodes) ===")
    print(summary.to_string())

    # Paired Wilcoxon on episode-level metrics (same eval seeds) between PPO variants.
    try:
        from scipy.stats import wilcoxon
        modes = [f"ppo_{m}" for m in args.modes]
        print("\n=== Wilcoxon (episodes pooled over seeds: NOT valid evidence, optimistic p-values; use the seed-level mean/std above) ===")
        ep = df[df.method.isin(modes)].groupby(["method", df.groupby(["method", "seed"]).cumcount()])[metrics].mean().reset_index()
        ep = ep.rename(columns={"level_1": "episode"})
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                a = ep[ep.method == modes[i]].sort_values("episode")
                b = ep[ep.method == modes[j]].sort_values("episode")
                n = min(len(a), len(b))
                for m in metrics[:3]:
                    try:
                        p = wilcoxon(a[m].values[:n], b[m].values[:n]).pvalue
                    except ValueError:
                        p = float("nan")
                    print(f"{modes[i]} vs {modes[j]:12s} {m:20s} p={p:.4g}")
    except ImportError:
        print("scipy not installed; skipping Wilcoxon tests")


if __name__ == "__main__":
    main()
