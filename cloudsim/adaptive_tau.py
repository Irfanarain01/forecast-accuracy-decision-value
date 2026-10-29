"""Adaptive, drift-aware upper-bound control for forecast-driven power management.

Streaming controller (one object per episode, warm-started on a pre-roll of recent trace history):
  L1  online conformal tracker: per forecaster m, horizon j and level tau, the residual quantile offset b[m,tau,j]
      follows the quantile gradient  b += eta*(1[e>b] - (1-tau)),  eta = gamma * scale[m,j] * boost.
  L2  drift detector: two-sided CUSUM on the coverage error of a reference bound; an alarm multiplies eta by B for T slots
      and pulls the arm weights toward uniform.
  L3  decision-cost expert aggregation: each arm (forecaster x tau, rolling percentile, reactive) proposes a host count;
      once the realised load of its horizon is known, its hindsight loss r*under + over (in hosts) updates exponential
      weights with fixed share; the controller plays the weighted median of the proposals.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "slot_level"))
import fdv_benchmark as fb
from cloud_env import SlotPowerEnv

WINDOW, KMAX = fb.WINDOW, fb.KMAX
TAUS = fb.TAUS


def load_zoo(pkl):
    import __main__, pickle
    for n in ("Persistence", "MovingAvg", "EWMA", "RidgeAR", "TorchPoint"):
        setattr(__main__, n, getattr(fb, n))
    with open(pkl, "rb") as f:
        return pickle.load(f)           # zoo, rq, val, test


def precompute(zoo, series):
    """P[m][g] = forecast (KMAX values for series[g..g+KMAX-1]) made from history series[:g]; g = 1..L-1."""
    L = len(series)
    padded = np.concatenate([np.full(WINDOW, series[0]), series]).astype(np.float32)
    X = sliding_window_view(padded, WINDOW)[:L].copy()          # row g = window of history length g
    out = {}
    for name, m in zoo.items():
        out[name] = np.vstack([m.predict(X[i:i + 2048]) for i in range(0, L, 2048)]).astype(np.float64)
    return out


class Streams:
    def __init__(self, zoo, rq, series):
        self.names = list(zoo)
        self.series = np.asarray(series, float)
        self.P = np.stack([precompute(zoo, self.series)[n] for n in self.names])        # (M, L, K)
        self.rq = rq
        self.b0 = np.stack([np.stack([rq[n][t][:KMAX] for t in TAUS]) for n in self.names])   # (M, T, K)
        sc = np.stack([(rq[n][0.9] - rq[n][0.5]) / 1.28 for n in self.names])                 # (M, K)
        self.scale = np.maximum(sc, 0.01)


class Hosts:
    """Vectorised copy of SlotPowerEnv.hosts_needed."""
    def __init__(self, env):
        self.cum = np.cumsum(env.host_mips[env.host_rank])
        self.mult = env.target_util * env.total_mips * 1.2
        self.n = env.n_hosts
    def __call__(self, load):
        return np.clip(np.searchsorted(self.cum, np.asarray(load) * self.mult) + 1, 1, self.n)


ROLL = [(q, L) for q in fb.QS for L in fb.LS]


class Controller:
    def __init__(self, S, env, k, ref_model, cfg):
        self.S, self.k, self.cfg = S, k, cfg
        self.H = Hosts(env)
        self.ref = S.names.index(ref_model)
        self.M, self.T = len(S.names), len(TAUS)
        self.b = S.b0.copy()
        self.use_track = cfg.get("track", True)
        self.use_drift = cfg.get("drift", True)
        self.mode = cfg["mode"]            # "fixed" | "hedge"
        self.fixed_tau = cfg.get("tau", 0.9)
        self.gamma, self.B, self.Tb = cfg.get("gamma", 0.05), cfg.get("B", 4.0), cfg.get("Tb", 20)
        self.kappa, self.lam = cfg.get("kappa", None), cfg.get("lam", 0.0)
        self.delta, self.h = cfg.get("delta", 0.05), cfg.get("h", 1.5)
        self.r, self.eta, self.alpha = cfg.get("r", 4.0), cfg.get("eta", 1.0), cfg.get("alpha", 0.02)
        self.cp, self.cm, self.boost_left, self.alarms = 0.0, 0.0, 0, 0
        # arms: forecaster x tau (+ point), rolling, reactive
        self.arms = [("f", m, t) for m in range(self.M) for t in range(self.T)] + [("p", m, -1) for m in range(self.M)]
        if cfg.get("rules", True):
            self.arms += [("r", q, L) for q, L in ROLL] + [("c", 0, 0)]
        if cfg.get("models") is not None:
            keep = set(S.names.index(x) for x in cfg["models"])
            self.arms = [a for a in self.arms if a[0] in ("r", "c") or a[1] in keep]
        if cfg.get("taus") is not None:
            ts = set(TAUS.index(x) for x in cfg["taus"])
            self.arms = [a for a in self.arms if a[0] != "f" or a[2] in ts]
        for nm, tv in cfg.get("extra_arms", []):
            mi = S.names.index(nm)
            self.arms.append(("p", mi, -1) if tv == "point" else ("f", mi, TAUS.index(float(tv))))
        self.A = len(self.arms)
        self.w = np.ones(self.A) / self.A
        self.dec, self.cur = {}, {}

    def _cur(self, g):
        s = self.S.series
        return float(np.mean(s[max(0, g - 3):g])) if g > 0 else 0.5

    def proposals(self, g):
        s, k = self.S.series, self.k
        cur = self._cur(g)
        F = self.S.P[:, g, :k]                                       # (M, k)
        hi = (F[:, None, :] + self.b[:, :, :k]).max(-1)               # (M, T)
        pt = F.max(-1)                                                # (M,)
        out = np.empty(self.A)
        roll = {}
        for i, (kind, a, b) in enumerate(self.arms):
            if kind == "f":
                v = hi[a, b]
            elif kind == "p":
                v = pt[a]
            elif kind == "r":
                h = s[max(0, g - b):g]
                v = np.quantile(h, a) if len(h) >= 10 else cur
            else:
                v = cur
            out[i] = max(cur, v)
        self.cur[g] = cur
        return self.H(out).astype(int)

    def update(self, g):
        s, k = self.S.series, self.k
        if g < 1:
            return
        # L1: residuals of the value revealed at g-1, one per horizon j (forecast made at g-1-j)
        j = np.arange(k)
        gg = g - 1 - j
        ok = gg >= 1
        if ok.any():
            jj, g0 = j[ok], gg[ok]
            e = s[g - 1] - self.S.P[:, g0, jj]                        # (M, n)
            boost = self.B if (self.use_drift and self.boost_left > 0) else 1.0
            if self.use_track:
                tau = np.array(TAUS)[None, :, None]
                ind = (e[:, None, :] > self.b[:, :, jj]).astype(float)
                step = (self.gamma * boost) * self.S.scale[:, None, jj]
                self.b[:, :, jj] += step * (ind - (1 - tau))
                if self.kappa is not None:
                    lo = self.S.b0[:, :, jj] - self.kappa * self.S.scale[:, None, jj]
                    self.b[:, :, jj] = np.maximum(self.b[:, :, jj], lo)
                if self.lam > 0 and self.boost_left == 0:
                    self.b[:, :, jj] += self.lam * (self.S.b0[:, :, jj] - self.b[:, :, jj])
            # L2: coverage-error CUSUM of the reference bound (reference model, tau = 0.9)
            if self.use_drift:
                z = float((e[self.ref] > self.b[self.ref, TAUS.index(0.9), jj]).mean()) - 0.1
                self.cp = max(0.0, self.cp + z - self.delta)
                self.cm = max(0.0, self.cm - z - self.delta)
                if self.boost_left > 0:
                    self.boost_left -= 1
                if self.cp > self.h or self.cm > self.h:
                    self.cp = self.cm = 0.0
                    self.boost_left = self.Tb
                    self.alarms += 1
                    self.w = 0.5 * self.w + 0.5 / self.A
        # L3: hindsight loss of the decision made at t = g-k (needs s[t..g-1])
        t = g - k
        if self.mode == "hedge" and t in self.dec:
            R = int(self.H(max(self.cur[t], float(s[t:g].max()))))
            n = self.dec[t]
            loss = (self.r * np.maximum(0, R - n) + np.maximum(0, n - R)) / (self.r * self.H.n)
            self.w = self.w * np.exp(-self.eta * loss)
            self.w = (1 - self.alpha) * self.w / self.w.sum() + self.alpha / self.A
            for old in [x for x in self.dec if x < t]:
                del self.dec[old]; self.cur.pop(old, None)

    def decide(self, g):
        n = self.proposals(g)
        self.dec[g] = n
        if self.mode == "fixed":
            m = self.arms.index(("p", self.ref, -1) if self.fixed_tau == "point" else ("f", self.ref, TAUS.index(self.fixed_tau)))
            return int(n[m])
        order = np.argsort(n, kind="stable")
        cw = np.cumsum(self.w[order])
        return int(n[order][min(len(cw) - 1, np.searchsorted(cw, self.cfg.get("rho", 0.5) * cw[-1]))])

    def warm(self, g_start, g_end):
        for g in range(g_start, g_end):
            self.update(g); self.decide(g)


class EnvS(SlotPowerEnv):
    """SlotPowerEnv that records the episode start index inside the trace."""
    def _gen_workload(self):
        import copy
        r = copy.deepcopy(self.np_random)
        self.ep_start = int(r.integers(0, max(1, len(self.trace) - 400)))
        super()._gen_workload()


class AdaptivePolicy:
    def __init__(self, S, ref_model, cfg, preroll=300):
        self.S, self.ref, self.cfg, self.pre = S, ref_model, cfg, preroll
        self.ctl = None
    def begin(self, env, k):
        self.ctl = Controller(self.S, env, k, self.ref, self.cfg)
        g0 = max(1, env.ep_start - self.pre)
        self.ctl.warm(g0, env.ep_start)
        self.last = None
    def __call__(self, env, obs):
        g = env.ep_start + env._slot
        self.ctl.update(g)
        return self.ctl.decide(g) - 1


def run_ep(env, policy, seed, k=None):
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "begin"):
        policy.begin(env, env.k)
    done, info = False, {}
    while not done:
        obs, r, done, _, info = env.step(policy(env, obs))
    return info
