"""
Pure-Python Gymnasium environment for online cloud task scheduling.

One step = one scheduling decision: the task at the head of the arrival queue is
assigned to one VM. VMs are FIFO single servers. Hosts are heterogeneous and are
charged idle power only while at least one of their VMs is busy (a host with no
work is assumed to sleep), plus dynamic power proportional to the share of the
host MIPS used by the running task. This gives a real trade-off between
consolidating work (lower energy) and spreading it (lower response time, fewer
SLA violations).

Workload: a time-varying load profile (synthetic, or a real trace you supply)
sets the Poisson arrival rate per time slot. Optional forecast features give
the agent the next-k slot loads:
    forecast_mode = "none"      zeros (reactive agent)
                    "point"     forecaster mean only
                    "quantile"  forecaster mean + lower + upper quantile
                    "oracle"    true future loads (upper bound)

Plug in your own model (for example the CNN-LSTM quantile forecaster) through
`forecaster(history, k) -> (mean, lo, hi)`, each an array of length k.

Reward per step (all terms normalised to O(1) per task):
    r = -(w_makespan * dMakespan + w_energy * dEnergy + w_sla * 1[late] + w_resp * response)
"""
from __future__ import annotations

import bisect
from typing import Callable, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Forecaster = Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]


class NaiveForecaster:
    """Moving-average forecaster with empirical-residual quantile bands.

    A stand-in for a learned model: the mean is the average of the last
    `window` slot loads; the band is mean +/- the 10th/90th percentile of recent
    one-step residuals. Replace it with your CNN-LSTM quantile model.
    """

    def __init__(self, window: int = 6):
        self.window = window

    def __call__(self, history: np.ndarray, k: int):
        if len(history) == 0:
            m = np.full(k, 0.5)
            return m, m - 0.1, m + 0.1
        w = min(self.window, len(history))
        mean = float(np.mean(history[-w:]))
        if len(history) > w + 2:
            roll = np.convolve(history, np.ones(w) / w, mode="valid")[:-1]
            resid = history[w:] - roll
            lo_r, hi_r = np.percentile(resid[-50:], [10, 90])
        else:
            lo_r, hi_r = -0.1, 0.1
        m = np.full(k, mean)
        return m, np.clip(m + lo_r, 0, 1), np.clip(m + hi_r, 0, 1)


def synthetic_loads(rng: np.random.Generator, n: int, drift: bool = False) -> np.ndarray:
    """Daily-like sinusoid + AR(1) noise + bursts; optional regime change halfway."""
    t = np.arange(n)
    period = rng.uniform(20, 30)
    phase = rng.uniform(0, 2 * np.pi)
    base = 0.55 + 0.25 * np.sin(2 * np.pi * t / period + phase)
    if drift:
        half = n // 2
        period2 = period * rng.uniform(0.4, 0.6)
        base[half:] = 0.35 + 0.3 * np.sin(2 * np.pi * (t[half:] - half) / period2 + phase)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 0.05)
    bursts = np.zeros(n)
    for s in rng.integers(0, n, size=max(1, n // 40)):
        bursts[s:s + rng.integers(2, 5)] += rng.uniform(0.2, 0.4)
    return np.clip(base + noise + bursts, 0.05, 1.0)


def scale_trace(tr, ref):
    """Map a utilisation series to ~[0.1, 1.0] using min/max of `ref` (use the TRAINING part for both
    the train and the test parts, so no test information leaks and both parts share one scale).
    Values outside the ref range are clipped to [0.05, 1.2]."""
    tr, ref = np.asarray(tr, dtype=float), np.asarray(ref, dtype=float)
    lo, hi = float(ref.min()), float(ref.max())
    return np.clip(0.1 + 0.9 * (tr - lo) / max(1e-9, hi - lo), 0.05, 1.2)


class CloudSchedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_hosts: int = 6,
        vms_per_host: int = 2,
        n_tasks: int = 1000,
        slot_s: float = 10.0,
        target_util: Optional[float] = None,
        mean_len_mi: float = 60_000.0,
        forecast_mode: str = "none",
        horizon_k: int = 6,
        forecaster: Optional[Forecaster] = None,
        load_trace: Optional[np.ndarray] = None,
        trace_prescaled: bool = False,
        drift: bool = False,
        weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        power_mgmt: bool = True,
        wake_s: float = 60.0,
        sleep_frac: float = 0.05,
    ):
        super().__init__()
        assert forecast_mode in ("none", "point", "quantile", "oracle")
        self.n_hosts, self.vms_per_host = n_hosts, vms_per_host
        self.n_vms = n_hosts * vms_per_host
        self.n_tasks, self.slot_s = n_tasks, slot_s
        self.mean_len, self.k = mean_len_mi, horizon_k
        self.mode, self.drift = forecast_mode, drift
        self.w_m, self.w_e, self.w_s, self.w_r = weights
        self.power_mgmt, self.wake_s, self.sleep_frac = power_mgmt, wake_s, sleep_frac
        if target_util is None:
            target_util = 1.0 if power_mgmt else 1.5
        self.target_util = target_util
        self.forecaster = forecaster if forecaster is not None else NaiveForecaster()
        self.trace = None
        if load_trace is not None:
            tr = np.asarray(load_trace, dtype=float)
            if trace_prescaled:      # already scaled with scale_trace() (train-set min/max), keep as is
                self.trace = tr
            else:
                self.trace = scale_trace(tr, tr)

        # Fixed heterogeneous hosts (three types: small / medium / large).
        types = [(10_000.0, 60.0, 150.0), (20_000.0, 90.0, 250.0), (40_000.0, 150.0, 450.0)]
        self.host_mips = np.array([types[h % 3][0] for h in range(n_hosts)])
        self.p_idle = np.array([types[h % 3][1] for h in range(n_hosts)])
        self.p_max = np.array([types[h % 3][2] for h in range(n_hosts)])
        self.vm_host = np.repeat(np.arange(n_hosts), vms_per_host)
        self.vm_mips = self.host_mips[self.vm_host] / vms_per_host
        self.total_mips = float(self.vm_mips.sum())

        self.rate_max = target_util * self.total_mips / self.mean_len  # tasks/s at load 1
        mean_rate = self.rate_max * 0.55
        self.t_ref = n_tasks / mean_rate                      # expected episode length (s)
        mean_exec = self.mean_len / float(self.vm_mips.mean())
        self.mean_exec = mean_exec
        self.backlog_scale = 10.0 * mean_exec
        self.m_scale = self.t_ref / n_tasks                   # per-task makespan scale
        self.e_scale = (self.p_idle.mean() + 0.5 * (self.p_max - self.p_idle).mean()) * self.m_scale

        # hosts ranked by energy efficiency (MIPS per watt at full load): wake the best first
        self.host_rank = np.argsort(-(self.host_mips / self.p_max), kind="stable")
        if power_mgmt:
            # e_scale: cost of keeping ALL hosts awake for one mean inter-arrival time = 1.0
            self.e_scale = float(self.p_idle.sum()) * self.m_scale

        self.obs_dim = 4 * self.n_vms + 3 + 3 * self.k + (2 * self.n_hosts + 1 if power_mgmt else 0)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_dim,), dtype=np.float32)
        if power_mgmt:
            # (VM for the head task, number of hosts to keep awake - 1)
            self.action_space = spaces.MultiDiscrete([self.n_vms, self.n_hosts])
        else:
            self.action_space = spaces.Discrete(self.n_vms)

    # ------------------------------------------------------------------ helpers
    def _gen_workload(self):
        rng = self.np_random
        if self.trace is not None:
            need = 400
            start = int(self._start) if self._start is not None else int(rng.integers(0, max(1, len(self.trace) - need)))
            loads = self.trace[start:start + need]
            if len(loads) < need:
                loads = np.pad(loads, (0, need - len(loads)), mode="edge")
        else:
            loads = synthetic_loads(rng, 400, self.drift)
        arrivals, slot = [], 0
        while len(arrivals) < self.n_tasks:
            if slot >= len(loads):
                loads = np.concatenate([loads, loads[-50:]])
            lam = self.rate_max * loads[slot] * self.slot_s
            cnt = rng.poisson(lam)
            arrivals.extend(slot * self.slot_s + np.sort(rng.uniform(0, self.slot_s, cnt)))
            slot += 1
        self.loads = loads[:max(slot, 1)]
        self.arrivals = np.array(arrivals[: self.n_tasks])
        self.lengths = rng.lognormal(np.log(self.mean_len) - 0.18, 0.6, self.n_tasks)
        self.deadlines = self.arrivals + rng.uniform(3.0, 8.0, self.n_tasks) * self.lengths / float(self.vm_mips.mean())

    # ----------------------------------------------------------- power states
    def hosts_needed(self, load: float, margin: float = 1.2) -> int:
        """Smallest number of top-ranked hosts whose MIPS cover `load` (rule-based helper)."""
        demand = load * self.target_util * self.total_mips * margin
        cum = np.cumsum(self.host_mips[self.host_rank])
        return int(np.clip(np.searchsorted(cum, demand) + 1, 1, self.n_hosts))

    def vm_start_times(self) -> np.ndarray:
        """Earliest time each VM could start the head task (includes wake-up delay of sleeping hosts)."""
        now = self.arrivals[self._i]
        t = np.maximum(self._free, now)
        if self.power_mgmt:
            ready = np.where(self._awake, self._ready, now + self.wake_s)
            t = np.maximum(t, ready[self.vm_host])
        return t

    def _advance(self, now: float) -> float:
        """Accrue idle/sleep energy (J) over [self._t, now] with the current host states."""
        t0, e = self._t, 0.0
        for h in range(self.n_hosts):
            if self._awake[h]:
                end = now
                if self._drain[h] and self._host_end[h] <= now:
                    end = max(t0, self._host_end[h])
                    self._awake[h], self._drain[h] = False, False
                awake_dt = max(0.0, end - t0)
                e += self.p_idle[h] * awake_dt + self.sleep_frac * self.p_idle[h] * max(0.0, now - end)
                self._awake_time += awake_dt
            else:
                e += self.sleep_frac * self.p_idle[h] * max(0.0, now - t0)
        self._t = now
        self._idle_j += e
        return e

    def _wake(self, h: int, now: float):
        self._awake[h], self._drain[h] = True, False
        self._ready[h] = now + self.wake_s
        self._wakes += 1

    def _apply_target(self, n: int, now: float):
        wanted = set(self.host_rank[:n].tolist())
        for h in range(self.n_hosts):
            if h in wanted:
                if not self._awake[h]:
                    self._wake(h, now)
                else:
                    self._drain[h] = False
            elif self._awake[h]:
                if self._host_end[h] <= now:
                    self._awake[h] = False
                else:
                    self._drain[h] = True
        self._target = n

    def _finalize(self) -> float:
        """End of episode: busy hosts finish their work, then everything sleeps until the makespan."""
        e, T = 0.0, max(self._makespan, self._t)
        for h in range(self.n_hosts):
            if self._awake[h]:
                end = max(self._t, self._host_end[h])
                self._awake_time += end - self._t
                e += self.p_idle[h] * (end - self._t) + self.sleep_frac * self.p_idle[h] * (T - end)
            else:
                e += self.sleep_frac * self.p_idle[h] * (T - self._t)
            self._awake[h] = False
        self._idle_j += e
        return e

    def _add_active(self, h: int, s: float, f: float) -> float:
        """Insert [s, f] into host h's merged busy intervals; return newly covered time."""
        iv = self._busy[h]
        added = f - s
        i = bisect.bisect_left(iv, [s, s])
        if i > 0 and iv[i - 1][1] >= s:
            i -= 1
        ns, nf = s, f
        j = i
        while j < len(iv) and iv[j][0] <= nf:
            os_, of = iv[j]
            overlap = max(0.0, min(f, of) - max(s, os_))
            added -= overlap
            ns, nf = min(ns, os_), max(nf, of)
            j += 1
        iv[i:j] = [[ns, nf]]
        return max(0.0, added)

    def _forecast(self, slot: int):
        k = self.k
        if self.mode == "none":
            z = np.zeros(k)
            return z, z, z
        if self.mode == "oracle":
            idx = np.minimum(np.arange(slot, slot + k), len(self.loads) - 1)
            f = self.loads[idx]
            return f, f, f
        hist = self.loads[:slot]
        mean, lo, hi = self.forecaster(hist, k)
        mean, lo, hi = np.asarray(mean), np.asarray(lo), np.asarray(hi)
        if self.mode == "point":
            return mean, mean, mean
        return mean, lo, hi

    def current_load(self, slot: Optional[int] = None) -> float:
        """Mean realised load of the last 3 slots (what a reactive controller can see)."""
        if slot is None:
            slot = int(self.arrivals[min(self._i, self.n_tasks - 1)] // self.slot_s)
        recent = self.loads[max(0, slot - 3):slot]
        return float(np.mean(recent)) if len(recent) else 0.5

    def _obs(self) -> np.ndarray:
        i = self._i
        now = self.arrivals[i]
        wait = np.maximum(self._free - now, 0.0)
        backlog = wait / (wait + self.backlog_scale)                 # squashed, never saturates
        est = np.maximum(self.vm_start_times() - now, 0.0) + self.lengths[i] / self.vm_mips   # est. response time per VM
        est = est / (est + self.backlog_scale)
        speed = self.vm_mips / self.vm_mips.max()
        host_busy = np.array([1.0 if self._host_end[h] > now else 0.0 for h in range(self.n_hosts)])[self.vm_host]
        slot = int(now // self.slot_s)
        cur = self.current_load(slot)
        if slot != self._fc_slot:
            self._fc = self._forecast(slot)
            self._fc_slot = slot
        task = np.array([
            min(1.0, self.lengths[i] / (3 * self.mean_len)),
            np.clip((self.deadlines[i] - now) / (10 * self.mean_exec), 0, 1),
            cur,
        ])
        fc = np.concatenate([np.clip(a, 0, 1) for a in self._fc])
        parts = [backlog, speed, host_busy, est, task, fc]
        if self.power_mgmt:
            wake_rem = np.clip(np.maximum(self._ready - now, 0.0) / self.wake_s, 0, 1)
            parts += [self._awake.astype(float), wake_rem, np.array([self._target / self.n_hosts])]
        return np.concatenate(parts).astype(np.float32)

    # --------------------------------------------------------------- gym API
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._start = (options or {}).get("start")
        self._gen_workload()
        self._i = 0
        self._free = np.zeros(self.n_vms)
        self._host_end = np.zeros(self.n_hosts)
        self._busy = [[] for _ in range(self.n_hosts)]
        self._makespan = 0.0
        self._dyn_j = 0.0
        self._active_s = np.zeros(self.n_hosts)
        self._late = 0
        self._wait = 0.0
        self._exec_sum = 0.0
        self._ret = 0.0
        self._t, self._idle_j, self._awake_time, self._wakes = 0.0, 0.0, 0.0, 0
        self._awake = np.ones(self.n_hosts, dtype=bool)
        self._drain = np.zeros(self.n_hosts, dtype=bool)
        self._ready = np.zeros(self.n_hosts)
        self._target = self.n_hosts
        self._fc_slot, self._fc = -1, None
        return self._obs(), {}

    def step(self, action):
        reward, terminated = self._step_core(action)
        if terminated:
            return np.zeros(self.obs_dim, dtype=np.float32), reward, True, False, self.episode_metrics()
        return self._obs(), reward, False, False, {}

    def _step_core(self, action):
        i = self._i
        now = self.arrivals[i]
        idle_e = 0.0
        if self.power_mgmt:
            j, n_target = int(action[0]), int(action[1]) + 1
            idle_e = self._advance(now)
            self._apply_target(n_target, now)
            h = int(self.vm_host[j])
            if not self._awake[h]:
                self._wake(h, now)
            start = max(now, self._free[j], self._ready[h])
        else:
            j = int(action)
            h = int(self.vm_host[j])
            start = max(now, self._free[j])
        exec_t = self.lengths[i] / self.vm_mips[j]
        fin = start + exec_t
        self._free[j] = fin
        self._host_end[h] = max(self._host_end[h], fin)

        if self.power_mgmt:
            d_energy = idle_e
        else:
            added = self._add_active(h, start, fin)
            self._active_s[h] += added
            d_energy = self.p_idle[h] * added
        dyn = (self.p_max[h] - self.p_idle[h]) * (self.vm_mips[j] / self.host_mips[h]) * exec_t
        self._dyn_j += dyn
        d_energy += dyn                                                # joules

        d_make = max(0.0, fin - self._makespan)
        self._makespan = max(self._makespan, fin)
        late = float(fin > self.deadlines[i])
        self._late += late
        self._wait += start - now
        self._exec_sum += exec_t

        self._i += 1
        terminated = self._i >= self.n_tasks
        if terminated and self.power_mgmt:
            d_energy += self._finalize()
        resp = (fin - now) / (5.0 * self.mean_exec)                    # normalised response time
        reward = -(self.w_m * d_make / self.m_scale + self.w_e * d_energy / self.e_scale
                   + self.w_s * late + self.w_r * resp)
        self._ret += reward
        return float(reward), terminated

    def episode_metrics(self) -> dict:
        if self.power_mgmt:
            energy_j = float(self._idle_j + self._dyn_j)
        else:
            energy_j = float((self.p_idle * self._active_s).sum() + self._dyn_j)
        extra = {}
        if self.power_mgmt:
            extra = {"wake_events": float(self._wakes),
                     "mean_awake_frac": self._awake_time / (self.n_hosts * max(1e-9, self._makespan))}
        return {**extra,
            "makespan_s": float(self._makespan),
            "energy_kwh": energy_j / 3.6e6,
            "sla_violation_rate": self._late / self.n_tasks,
            "mean_wait_s": self._wait / self.n_tasks,
            "mean_util": self._exec_sum / (self.n_vms * max(1e-9, self._makespan)),
            "episode_return": float(self._ret),   # same objective for every method
        }


class SlotPowerEnv(CloudSchedEnv):
    """Hierarchical version: ONE agent decision per time slot = how many hosts to keep awake.

    Tasks arriving in the slot are placed by the Earliest-Finish rule. This removes the per-task
    power thrashing seen in the flat environment and makes the forecast directly useful: the
    agent decides, once per slot, how much capacity to have ready for the load that is coming.
    `wake_penalty` (default 0) is an optional training-only shaping term per wake-up; the reported
    `episode_return` never includes it, so every method is compared on the same objective.
    """

    def __init__(self, wake_penalty: float = 0.0, residual: bool = False, res_range: int = 2, **kw):
        kw["power_mgmt"] = True
        super().__init__(**kw)
        self.wake_penalty = wake_penalty
        # residual=True: the agent corrects the matching rule instead of choosing the host count itself.
        #   rule = reactive (forecast_mode none) or max(recent load, forecast upper bound) (point/quantile/oracle)
        #   action a in {0..2R} -> hosts awake = clip(rule + a - R, 1, n_hosts); a = R reproduces the rule exactly.
        self.residual, self.res_range = residual, int(res_range)
        self._base_n = 1
        self.action_space = spaces.Discrete(2 * self.res_range + 1 if residual else self.n_hosts)
        self.obs_dim = 3 * self.n_hosts + 1 + 3 * self.k + 2 + (1 if residual else 0)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_dim,), dtype=np.float32)

    def current_load(self, slot: Optional[int] = None) -> float:
        return super().current_load(self._slot if slot is None else slot)

    def _slot_obs(self) -> np.ndarray:
        ts = self._slot * self.slot_s
        host_wait = np.maximum(self._host_end - ts, 0.0)
        host_bl = host_wait / (host_wait + self.backlog_scale)
        wake_rem = np.clip(np.maximum(self._ready - ts, 0.0) / self.wake_s, 0, 1)
        vm_wait = np.maximum(self._free - ts, 0.0)
        queue = float(np.mean(vm_wait / (vm_wait + self.backlog_scale)))
        if self._slot != self._fc_slot:
            self._fc = self._forecast(self._slot)
            self._fc_slot = self._slot
        fc = np.concatenate([np.clip(a, 0, 1) for a in self._fc])
        cur = self.current_load(self._slot)
        if self.mode == "none":
            self._base_n = self.hosts_needed(cur)
        else:
            self._base_n = self.hosts_needed(max(cur, float(np.max(self._fc[2]))))
        parts = [self._awake.astype(float), wake_rem, host_bl,
                 np.array([cur]), fc,
                 np.array([self._target / self.n_hosts, queue])]
        if self.residual:
            parts.append(np.array([self._base_n / self.n_hosts]))
        return np.concatenate(parts).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        self._slot = int(self.arrivals[0] // self.slot_s)
        self._fc_slot, self._fc = -1, None
        return self._slot_obs(), {}

    def step(self, action):
        if self.residual:
            n = int(np.clip(self._base_n + int(action) - self.res_range, 1, self.n_hosts))
        else:
            n = int(action) + 1
        ts, end = self._slot * self.slot_s, (self._slot + 1) * self.slot_s
        wakes0 = self._wakes
        extra_e = self._advance(ts)
        self._apply_target(n, ts)
        reward, terminated = 0.0, False
        while self._i < self.n_tasks and self.arrivals[self._i] < end:
            j = int(np.argmin(self.vm_start_times() + self.lengths[self._i] / self.vm_mips))
            r, terminated = self._step_core(np.array([j, n - 1]))
            reward += r
            if terminated:
                break
        if not terminated:
            extra_e += self._advance(end)
        e_rew = -(self.w_e * extra_e / self.e_scale)
        reward += e_rew
        self._ret += e_rew                       # pure objective, no shaping
        train_reward = reward - self.wake_penalty * (self._wakes - wakes0)
        self._slot += 1
        if terminated:
            return np.zeros(self.obs_dim, dtype=np.float32), float(train_reward), True, False, self.episode_metrics()
        return self._slot_obs(), float(train_reward), False, False, {}
