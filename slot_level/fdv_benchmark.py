"""Forecast accuracy vs decision value: benchmark.

For one trace:
  1. fit a zoo of forecasters on the TRAIN part (persistence, moving average, EWMA, ridge-AR, MLP, CNN-LSTM),
  2. measure forecast accuracy (MAE, MASE, coverage, error in volatile windows) on a validation segment and on the test part,
  3. calibrate an upper bound from validation residuals at several quantile levels tau,
  4. measure DECISION VALUE: run the power-management simulator with each (forecaster, tau) on validation episodes
     (used for selection) and on test episodes (reported), next to reactive, always-on, oracle, static-n and rolling-percentile rules.
  python fdv_benchmark.py --trace ali_cpu.npy --wakes 120 300 600 --out fdv_ali
Writes <out>/accuracy.csv, <out>/decision_episodes.csv, <out>/drift.json. Analyse with fdv_analyze.py.
"""
import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import Ridge

from baselines import SLOT_BASELINES, SlotRule, forecast_power, run_episode
from cloud_env import SlotPowerEnv, scale_trace
from forecaster_cnn_lstm import CNNLSTM, QUANTILES, pinball_loss, make_windows

WINDOW, KMAX = 24, 60
TAUS = [0.5, 0.8, 0.9, 0.95, 0.99]
QS, LS = [0.5, 0.75, 0.9, 0.95, 0.99], [50, 100]


# ------------------------------------------------------------------ data
def prep(path):
    arr = np.asarray(np.load(path), dtype=float).ravel()
    cut = int(0.8 * len(arr))
    train = scale_trace(arr[:cut], arr[:cut])
    test = scale_trace(arr[cut:], arr[:cut])
    c2 = int(0.85 * len(train))
    return train[:c2], train[c2:], test


def lastwin(h):
    h = np.asarray(h, dtype=np.float32)
    if len(h) == 0:
        h = np.array([0.5], dtype=np.float32)
    if len(h) < WINDOW:
        h = np.concatenate([np.full(WINDOW - len(h), h[0], dtype=np.float32), h])
    return h[-WINDOW:]


# ------------------------------------------------------------------ forecaster zoo: predict(X (n,WINDOW)) -> (n,KMAX)
class Persistence:
    def predict(self, X): return np.repeat(X[:, -1:], KMAX, axis=1)

class MovingAvg:
    def predict(self, X): return np.repeat(X[:, -12:].mean(1, keepdims=True), KMAX, axis=1)

class EWMA:
    def predict(self, X):
        a, level = 0.3, X[:, 0].copy()
        for j in range(1, X.shape[1]):
            level = a * X[:, j] + (1 - a) * level
        return np.repeat(level[:, None], KMAX, axis=1)

class RidgeAR:
    def fit(self, X, Y): self.m = Ridge(alpha=1.0).fit(X, Y); return self
    def predict(self, X): return self.m.predict(X)

class TorchPoint:
    def __init__(self, kind): self.kind = kind
    def fit(self, X, Y, epochs=30, seed=0):
        torch.manual_seed(seed); torch.set_num_threads(1)
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)
        if self.kind == "mlp":
            self.net = nn.Sequential(nn.Linear(WINDOW, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, KMAX))
            loss_fn = lambda p, y: ((p - y) ** 2).mean()
        else:
            self.net = CNNLSTM(KMAX)
            loss_fn = lambda p, y: pinball_loss(p, y)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        for _ in range(epochs):
            self.net.train(); perm = torch.randperm(len(X))
            for i in range(0, len(perm), 128):
                b = perm[i:i + 128]; opt.zero_grad(); loss_fn(self.net(X[b]), Y[b]).backward(); opt.step()
        self.net.eval(); return self
    def predict(self, X):
        torch.set_num_threads(1)
        with torch.no_grad():
            out = self.net(torch.from_numpy(np.asarray(X, dtype=np.float32)))
            if self.kind == "cnn_lstm":
                out = torch.sort(out, dim=2).values[..., 1]          # median quantile
            return out.numpy()


def fit_zoo(fit_series, epochs):
    X, Y = make_windows(fit_series, WINDOW, KMAX)
    return {"persistence": Persistence(), "moving_avg": MovingAvg(), "ewma": EWMA(),
            "ridge_ar": RidgeAR().fit(X, Y), "mlp": TorchPoint("mlp").fit(X, Y, epochs), "cnn_lstm": TorchPoint("cnn_lstm").fit(X, Y, epochs)}


class ZooForecaster:
    """Environment hook: history, k -> (mean, lo, hi). hi = mean + tau-quantile of validation residuals (None: point)."""
    def __init__(self, model, rq, tau):
        self.model, self.rq, self.tau = model, rq, tau
    def __call__(self, history, k):
        m = self.model.predict(lastwin(history)[None])[0][:k]
        if self.tau is None:
            return m, m, m
        return m, m, m + self.rq[self.tau][:k]


# ------------------------------------------------------------------ accuracy
def accuracy_rows(zoo, rq, fit, val, test, wakes, slot_s):
    rows = []
    for split, series in (("val", val), ("test", test)):
        X, Y = make_windows(series, WINDOW, KMAX)
        pers = zoo["persistence"].predict(X)
        for name, m in zoo.items():
            P = m.predict(X)
            for W in wakes:
                k = max(1, int(round(W / slot_s)))
                err = np.abs(Y[:, :k] - P[:, :k]); mae = err.mean()
                mase = mae / max(1e-9, np.abs(Y[:, :k] - pers[:, :k]).mean())
                vol = np.abs(Y[:, k - 1] - X[:, -1]); hi_vol = vol >= np.quantile(vol, 0.8)
                hi90 = P[:, :k] + rq[name][0.9][:k]
                rows.append({"model": name, "split": split, "k": k, "wake_s": W, "mae": mae, "rmse": float(np.sqrt(((Y[:, :k] - P[:, :k]) ** 2).mean())),
                             "mase": mase, "mae_volatile": err[hi_vol].mean(), "cov90": float((Y[:, :k] <= hi90).mean()),
                             "pinball90": float(np.maximum(0.9 * (Y[:, :k] - hi90), -0.1 * (Y[:, :k] - hi90)).mean())})
    return rows


# ------------------------------------------------------------------ decision runs
def make_rolling(q, L):
    def fn(env, obs):
        h = env.loads[max(0, env._slot - L):env._slot]
        cur = env.current_load()
        return env.hosts_needed(max(cur, float(np.quantile(h, q)) if len(h) >= 10 else cur))
    return fn


def decision_task(args):
    pkl, W, part, episodes, n_tasks, slot_s = args
    torch.set_num_threads(1)
    with open(pkl, "rb") as f:
        zoo, rq, val, test = pickle.load(f)
    series, base_seed = (val, 300_000) if part == "val" else (test, 100_000)
    k = max(1, int(round(W / slot_s)))
    envs = {m: SlotPowerEnv(n_tasks=n_tasks, forecast_mode=m, load_trace=series, trace_prescaled=True, forecaster=None,
                            wake_s=float(W), horizon_k=k, slot_s=slot_s) for m in ("none", "point", "quantile", "oracle")}
    cfgs = []   # (method, model, tau, env mode, forecaster, policy)
    cfgs.append(("reactive", "", "", "none", None, SLOT_BASELINES["reactive"][1]()))
    cfgs.append(("always_on", "", "", "none", None, SLOT_BASELINES["always_on"][1]()))
    cfgs.append(("oracle", "", "", "oracle", None, SlotRule(forecast_power)))
    for n in range(1, 7):
        cfgs.append((f"static_{n}", "", "", "none", None, SlotRule(lambda e, o, n=n: n)))
    for q in QS:
        for L in LS:
            cfgs.append((f"rolling_q{q}_L{L}", "", "", "none", None, SlotRule(make_rolling(q, L))))
    for name, model in zoo.items():
        cfgs.append(("zoo", name, "point", "point", ZooForecaster(model, rq[name], None), SlotRule(forecast_power)))
        for tau in TAUS:
            cfgs.append(("zoo", name, str(tau), "quantile", ZooForecaster(model, rq[name], tau), SlotRule(forecast_power)))
    rows = []
    for method, model, tau, mode, fc, pol in cfgs:
        env = envs[mode]
        if fc is not None:
            env.forecaster = fc
        for e in range(episodes):
            info = run_episode(env, pol, seed=base_seed + e)
            rows.append({"wake_s": W, "part": part, "method": method, "model": model, "tau": tau, "episode": e, **info})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--wakes", nargs="+", type=float, default=[120, 300, 600])
    ap.add_argument("--val-episodes", type=int, default=20)
    ap.add_argument("--test-episodes", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-tasks", type=int, default=1000)
    ap.add_argument("--slot-s", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="fdv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fit, val, test = prep(a.trace)
    assert len(val) >= 450, "validation segment too short for 400-slot episodes"
    print(f"fit {len(fit)} / val {len(val)} / test {len(test)} steps; fitting forecaster zoo ...", flush=True)
    zoo = fit_zoo(fit, a.epochs)
    # residual quantiles per horizon from the validation segment
    Xv, Yv = make_windows(val, WINDOW, KMAX)
    rq = {n: {t: np.quantile(Yv - m.predict(Xv), t, axis=0) for t in TAUS} for n, m in zoo.items()}
    acc = pd.DataFrame(accuracy_rows(zoo, rq, fit, val, test, a.wakes, a.slot_s))
    acc.to_csv(os.path.join(a.out, "accuracy.csv"), index=False)
    drift = {"fit_mean": float(fit.mean()), "fit_std": float(fit.std()), "val_mean": float(val.mean()), "test_mean": float(test.mean()), "test_std": float(test.std()),
             "shift_test_std_units": float((test.mean() - fit.mean()) / fit.std()), "ks_fit_test": float(stats.ks_2samp(fit, test).statistic),
             "ks_fit_val": float(stats.ks_2samp(fit, val).statistic), "mean_abs_step_fit": float(np.abs(np.diff(fit)).mean()), "mean_abs_step_test": float(np.abs(np.diff(test)).mean())}
    json.dump(drift, open(os.path.join(a.out, "drift.json"), "w"), indent=1)
    print("drift:", {k: round(v, 3) for k, v in drift.items()}, flush=True)
    pkl = os.path.join(a.out, "zoo.pkl")
    with open(pkl, "wb") as f:
        pickle.dump((zoo, rq, val, test), f)
    jobs = [(pkl, W, part, a.val_episodes if part == "val" else a.test_episodes, a.n_tasks, a.slot_s) for W in a.wakes for part in ("val", "test")]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, part in enumerate(ex.map(decision_task, jobs)):
            rows += part; print(f"  decision task {i + 1}/{len(jobs)} done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "decision_episodes.csv"), index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
