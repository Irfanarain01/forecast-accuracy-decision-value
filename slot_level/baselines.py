"""Heuristic schedulers used as baselines.

A policy is a callable (env, obs) -> action. Without power management the action is a VM index.
With power management (the default environment) it is np.array([vm, n_hosts_awake - 1]); the
`Combo` helper joins a VM-selection rule with a power rule.
"""
import numpy as np


# ------------------------------------------------------------------ VM selection
def first_fit(env, obs):
    """Lowest-index VM whose backlog is below a threshold, else the least backlogged."""
    backlog = obs[: env.n_vms]
    ok = np.where(backlog < 0.5)[0]
    return int(ok[0]) if len(ok) else int(np.argmin(backlog))


class RoundRobin:
    def __init__(self):
        self.i = -1

    def __call__(self, env, obs):
        self.i = (self.i + 1) % env.n_vms
        return self.i


def random_policy(env, obs):
    return int(env.np_random.integers(env.n_vms))


def earliest_finish(env, obs):
    """Greedy: pick the VM that finishes the head task soonest (includes wake-up delay)."""
    fin = env.vm_start_times() + env.lengths[env._i] / env.vm_mips
    return int(np.argmin(fin))


# ------------------------------------------------------------------ power rules
def _cur_load(env, obs):
    return env.current_load()


def always_on(env, obs):
    return env.n_hosts


def reactive_power(env, obs):
    """Keep enough hosts awake for the recently observed load (no forecast)."""
    return env.hosts_needed(_cur_load(env, obs))


def forecast_power(env, obs):
    """Keep enough hosts awake for max(recent load, forecast upper bound over the horizon).
    The environment's forecast mode decides what the forecast is: point, quantile or oracle."""
    fc_hi = float(np.max(env._fc[2]))
    return env.hosts_needed(max(_cur_load(env, obs), fc_hi))


class Combo:
    def __init__(self, vm_policy, power_policy):
        self.vm, self.pw = vm_policy, power_policy

    def __call__(self, env, obs):
        return np.array([self.vm(env, obs), self.pw(env, obs) - 1])


class SlotRule:
    """Per-slot power rule for SlotPowerEnv (VMs are placed by Earliest-Finish inside the env)."""
    def __init__(self, power_fn):
        self.pw = power_fn

    def __call__(self, env, obs):
        return self.pw(env, obs) - 1


# (environment forecast mode, constructor)
BASELINES = {   # used when power management is OFF
    "first_fit": ("none", lambda: first_fit),
    "round_robin": ("none", lambda: RoundRobin()),
    "random": ("none", lambda: random_policy),
    "earliest_finish": ("none", lambda: earliest_finish),
}
POWER_BASELINES = {   # used when power management is ON (default)
    "first_fit_always_on": ("none", lambda: Combo(first_fit, always_on)),
    "round_robin_always_on": ("none", lambda: Combo(RoundRobin(), always_on)),
    "earliest_finish_always_on": ("none", lambda: Combo(earliest_finish, always_on)),
    "earliest_finish_reactive": ("none", lambda: Combo(earliest_finish, reactive_power)),
    "earliest_finish_fc_point": ("point", lambda: Combo(earliest_finish, forecast_power)),
    "earliest_finish_fc_quantile": ("quantile", lambda: Combo(earliest_finish, forecast_power)),
    "earliest_finish_fc_oracle": ("oracle", lambda: Combo(earliest_finish, forecast_power)),
}


SLOT_BASELINES = {   # used with the slot-level environment (default)
    "always_on": ("none", lambda: SlotRule(always_on)),
    "reactive": ("none", lambda: SlotRule(reactive_power)),
    "fc_point": ("point", lambda: SlotRule(forecast_power)),
    "fc_quantile": ("quantile", lambda: SlotRule(forecast_power)),
    "fc_oracle": ("oracle", lambda: SlotRule(forecast_power)),
}


def run_episode(env, policy, seed, options=None):
    obs, _ = env.reset(seed=seed, options=options)
    done, info = False, {}
    while not done:
        obs, r, done, _, info = env.step(policy(env, obs))
    return info
