"""Robustness of 'decision-selected vs accuracy-selected' forecaster choice.

Rolling-origin evaluation: for each origin fraction f the forecasters are fit/validated on trace[:cut] (cut = f*N, the last
max(450, 15%) steps are validation) and tested on the NEXT block trace[cut:cut+block] (block = 20% of N). Neural models are
trained with several seeds. For every (origin, seed, wake time) the script
  * picks the forecaster by validation MAE (tau = 0.9)            -> 'accuracy-selected'
  * picks (model, tau) by validation episode return               -> 'decision-selected'
  * picks the best static / rolling rule by validation return      -> 'simple rule'
and records their TEST returns on shared test episodes.

  python fdv_robust.py run    --trace bb_fast.npy --out rob_bbfast --wakes 300 600
  python fdv_robust.py report --dir fastStorage=rob_bbfast --dir Rnd=rob_bbrnd [--dir ...] --out rob_report
"""
import argparse
import glob
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from scipy import stats

import fdv_benchmark as fb
from baselines import SLOT_BASELINES, SlotRule, forecast_power, run_episode
from cloud_env import SlotPowerEnv, scale_trace
from forecaster_cnn_lstm import make_windows

NEURAL = ("mlp", "cnn_lstm")


def split(arr, f, block_frac=0.2):
    n = len(arr)
    cut, block = int(f * n), int(block_frac * n)
    nval = max(450, int(0.15 * cut))
    pre = scale_trace(arr[:cut], arr[:cut])
    test = scale_trace(arr[cut:cut + block], arr[:cut])
    return pre[:-nval], pre[-nval:], test


def fit_unit(fit, seed, epochs, full):
    X, Y = make_windows(fit, fb.WINDOW, fb.KMAX)
    zoo = {}
    if full:
        zoo.update({"persistence": fb.Persistence(), "moving_avg": fb.MovingAvg(), "ewma": fb.EWMA(),
                    "ridge_ar": fb.RidgeAR().fit(X, Y)})
    zoo["mlp"] = fb.TorchPoint("mlp").fit(X, Y, epochs, seed=seed)
    zoo["cnn_lstm"] = fb.TorchPoint("cnn_lstm").fit(X, Y, epochs, seed=seed)
    return zoo


def task(args):
    """One (pkl, wake, part) job; rules are evaluated only when full=True."""
    pkl, W, part, episodes, n_tasks, slot_s, full = args
    torch.set_num_threads(1)
    with open(pkl, "rb") as f:
        zoo, rq, val, test = pickle.load(f)
    series, base_seed = (val, 300_000) if part == "val" else (test, 100_000)
    k = max(1, int(round(W / slot_s)))
    envs = {m: SlotPowerEnv(n_tasks=n_tasks, forecast_mode=m, load_trace=series, trace_prescaled=True, forecaster=None,
                            wake_s=float(W), horizon_k=k, slot_s=slot_s) for m in ("none", "point", "quantile", "oracle")}
    cfgs = []
    if full:
        cfgs.append(("reactive", "", "", "none", None, SLOT_BASELINES["reactive"][1]()))
        for n in range(1, 7):
            cfgs.append((f"static_{n}", "", "", "none", None, SlotRule(lambda e, o, n=n: n)))
        for q in fb.QS:
            for L in fb.LS:
                cfgs.append((f"rolling_q{q}_L{L}", "", "", "none", None, SlotRule(fb.make_rolling(q, L))))
    for name, model in zoo.items():
        cfgs.append(("zoo", name, "point", "point", fb.ZooForecaster(model, rq[name], None), SlotRule(forecast_power)))
        for tau in fb.TAUS:
            cfgs.append(("zoo", name, str(tau), "quantile", fb.ZooForecaster(model, rq[name], tau), SlotRule(forecast_power)))
    rows = []
    for method, model, tau, mode, fc, pol in cfgs:
        env = envs[mode]
        if fc is not None:
            env.forecaster = fc
        for e in range(episodes):
            info = run_episode(env, pol, seed=base_seed + e)
            rows.append({"wake_s": W, "part": part, "method": method, "model": model, "tau": tau, "episode": e, **info})
    return rows


def cmd_run(a):
    arr = np.asarray(np.load(a.trace), dtype=float).ravel()
    os.makedirs(a.out, exist_ok=True)
    jobs, meta = [], []
    for f in a.origins:
        for seed in range(a.seeds):
            tag = f"o{int(round(f * 100))}_s{seed}"
            done = os.path.join(a.out, f"decisions_{tag}.csv")
            if os.path.exists(done):
                print("have", tag); continue
            fit, val, test = split(arr, f)
            full = seed == 0
            print(f"[{tag}] fit {len(fit)} val {len(val)} test {len(test)}; training ...", flush=True)
            zoo = fit_unit(fit, seed, a.epochs, full)
            Xv, Yv = make_windows(val, fb.WINDOW, fb.KMAX)
            rq = {n: {t: np.quantile(Yv - m.predict(Xv), t, axis=0) for t in fb.TAUS} for n, m in zoo.items()}
            acc = [{"model": n, "wake_s": W, "val_mae": float(np.abs(Yv[:, :max(1, int(round(W / a.slot_s)))] -
                                                                       m.predict(Xv)[:, :max(1, int(round(W / a.slot_s)))]).mean())}
                   for n, m in zoo.items() for W in a.wakes]
            pd.DataFrame(acc).to_csv(os.path.join(a.out, f"valacc_{tag}.csv"), index=False)
            pkl = os.path.join(a.out, f"zoo_{tag}.pkl")
            with open(pkl, "wb") as fh:
                pickle.dump((zoo, rq, val, test), fh)
            jobs.append((tag, [(pkl, W, part, a.val_episodes if part == "val" else a.test_episodes, a.n_tasks, a.slot_s, full)
                               for W in a.wakes for part in ("val", "test")]))
    for tag, tl in jobs:
        rows = []
        with ProcessPoolExecutor(a.workers) as ex:
            for part in ex.map(task, tl):
                rows += part
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"decisions_{tag}.csv"), index=False)
        print(f"[{tag}] done", flush=True)


# ------------------------------------------------------------------ report
def load_dir(d):
    units = {}
    for p in glob.glob(os.path.join(d, "decisions_*.csv")):
        m = re.search(r"decisions_o(\d+)_s(\d+)\.csv", p)
        o, s = int(m.group(1)), int(m.group(2))
        units[(o, s)] = (pd.read_csv(p, keep_default_na=False), pd.read_csv(os.path.join(d, f"valacc_o{o}_s{s}.csv")))
    return units


def tcrit(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), h


def evaluate(units):
    """-> DataFrame, one row per (origin, seed, wake) with test-return differences."""
    out = []
    for (o, s), (D, A) in sorted(units.items()):
        base = units[(o, 0)]
        for W in sorted(D.wake_s.unique()):
            # pool: seed 0 non-neural + this seed's neural models; rules from seed 0
            def sel(df0, dfs):
                pool = pd.concat([df0[~df0.model.isin(NEURAL)], dfs[dfs.model.isin(NEURAL)]])
                return pool
            d0, ds = base[0][base[0].wake_s == W], D[D.wake_s == W]
            pool = sel(d0, ds)
            zoo = pool[pool.method == "zoo"]
            rules = d0[~d0.method.isin(["zoo", "reactive"])]
            val_z = zoo[zoo.part == "val"].groupby(["model", "tau"]).episode_return.mean()
            test_z = zoo[zoo.part == "test"].groupby(["model", "tau", "episode"]).episode_return.mean()
            val_r = rules[rules.part == "val"].groupby("method").episode_return.mean()
            test_r = rules[rules.part == "test"].groupby(["method", "episode"]).episode_return.mean()
            test_re = d0[(d0.method == "reactive") & (d0.part == "test")].set_index("episode").episode_return
            acc = pd.concat([base[1][~base[1].model.isin(NEURAL)], A[A.model.isin(NEURAL)]])
            acc = acc[acc.wake_s == W].set_index("model").val_mae
            m_acc = acc.idxmin()
            k_acc = (m_acc, "0.9")
            k_dec = val_z.idxmax()
            r_best = val_r.idxmax()
            ra = test_z.xs(k_acc[0], level=0).xs(k_acc[1], level=0)
            rd = test_z.xs(k_dec[0], level=0).xs(k_dec[1], level=0)
            rr = test_r.xs(r_best, level=0)
            out.append({"origin": o, "seed": s, "wake_s": W, "acc_model": m_acc, "dec_cfg": f"{k_dec[0]}|{k_dec[1]}", "rule": r_best,
                        "acc": ra.mean(), "dec": rd.mean(), "rule_ret": rr.mean(), "reactive": test_re.mean(),
                        "dec_minus_acc": (rd - ra).mean(), "dec_minus_rule": (rd - rr).mean(), "acc_minus_rule": (ra - rr).mean()})
    return pd.DataFrame(out)


def cmd_report(a):
    os.makedirs(a.out, exist_ok=True)
    allrows = []
    for spec in a.dir:
        name, d = spec.split("=", 1)
        df = evaluate(load_dir(d)); df.insert(0, "trace", name); allrows.append(df)
    R = pd.concat(allrows)
    R.to_csv(os.path.join(a.out, "robust_units.csv"), index=False)
    lines = ["Rolling-origin / multi-seed robustness (test returns; positive = first minus second is better)\n"]
    for (tr, W), g in R.groupby(["trace", "wake_s"]):
        po = g.groupby("origin")[["dec_minus_acc", "dec_minus_rule", "acc_minus_rule"]].mean()   # average seeds within an origin
        row = [f"{tr:12s} wake {int(W):4d}s  origins={len(po)} seeds/origin={g.groupby('origin').seed.nunique().min()}"]
        for c, lab in (("dec_minus_acc", "decision-sel - accuracy-sel"), ("dec_minus_rule", "decision-sel - simple rule"), ("acc_minus_rule", "accuracy-sel - simple rule")):
            m, h = tcrit(po[c].values)
            row.append(f"   {lab:30s} {m:+8.1f} +/- {h:6.1f} (t-interval over origins); positive in {int((po[c] > 0).sum())}/{len(po)} origins")
        lines += row
        lines.append("   per origin (dec-acc): " + ", ".join(f"{o}%:{v:+.0f}" for o, v in po.dec_minus_acc.items()))
        lines.append("")
    txt = "\n".join(lines)
    open(os.path.join(a.out, "robust_report.txt"), "w").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--trace", required=True); r.add_argument("--out", required=True)
    r.add_argument("--wakes", nargs="+", type=float, default=[300, 600])
    r.add_argument("--origins", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8])
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--val-episodes", type=int, default=20); r.add_argument("--test-episodes", type=int, default=30)
    r.add_argument("--epochs", type=int, default=30); r.add_argument("--n-tasks", type=int, default=1000)
    r.add_argument("--slot-s", type=float, default=10.0); r.add_argument("--workers", type=int, default=2)
    p = sub.add_parser("report")
    p.add_argument("--dir", action="append", required=True); p.add_argument("--out", default="rob_report")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    main()
