"""Analyse fdv_benchmark.py output: accuracy vs decision value, and accuracy-selected vs decision-selected forecasters.

  python fdv_analyze.py --dir Alibaba=fdv_ali --dir Azure=fdv_azf --dir fastStorage=fdv_bbfast --dir Rnd=fdv_bbrnd --out fdv_report
Prints the tables and writes fdv_report/strategies.csv, fdv_report/accuracy_vs_value.csv.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

R = "episode_return"
MODELS = ["persistence", "moving_avg", "ewma", "ridge_ar", "mlp", "cnn_lstm"]


def key(r):
    return r.method if r.method != "zoo" else f"zoo|{r.model}|{r.tau}"


def ci(d):
    d = np.asarray(d, dtype=float)
    if len(d) < 2 or d.std() == 0:
        return float(d.mean()), float(d.mean()), float(d.mean())
    lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
    return float(d.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, help="label=folder")
    ap.add_argument("--out", default="fdv_report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    strat_rows, av_rows = [], []
    for item in a.dir:
        label, path = item.split("=", 1)
        D = pd.read_csv(os.path.join(path, "decision_episodes.csv"), keep_default_na=False)
        D["tau"] = D["tau"].astype(str)
        D["cfg"] = [f"zoo|{m}|{t}" if me == "zoo" else me for me, m, t in zip(D.method, D.model, D.tau)]
        A = pd.read_csv(os.path.join(path, "accuracy.csv"))
        drift = json.load(open(os.path.join(path, "drift.json")))
        print(f"\n################ {label}  (test mean shift {drift['shift_test_std_units']:+.2f} train-std, KS fit/test {drift['ks_fit_test']:.2f})")
        for W in sorted(D.wake_s.unique()):
            d = D[D.wake_s == W]
            val = d[d.part == "val"].groupby("cfg")[R].mean()
            te = d[d.part == "test"].pivot(index="episode", columns="cfg", values=R)
            sla = d[d.part == "test"].groupby("cfg").sla_violation_rate.mean() * 100
            en = d[d.part == "test"].groupby("cfg").energy_kwh.mean()
            k = int(round(W / 10))
            acc = A[(A.wake_s == W)]
            zoo_cfgs = [c for c in te.columns if c.startswith("zoo|")]
            stat_cfgs = [c for c in te.columns if c.startswith("static_")]
            roll_cfgs = [c for c in te.columns if c.startswith("rolling_")]
            best = lambda cfgs: max(cfgs, key=lambda c: val[c])
            val_mae = acc[acc.split == "val"].set_index("model").mae
            acc_model = min(MODELS, key=lambda m: val_mae[m])
            strategies = {
                "reactive": "reactive", "always_on": "always_on",
                "static (n chosen on val)": best(stat_cfgs), "rolling pct (q,L chosen on val)": best(roll_cfgs),
                "forecast: ACCURACY-selected model, tau=0.9": f"zoo|{acc_model}|0.9",
                "forecast: ACCURACY-selected model, point": f"zoo|{acc_model}|point",
                "forecast: DECISION-selected model, tau=0.9": best([c for c in zoo_cfgs if c.endswith("|0.9")]),
                "forecast: DECISION-selected (model,tau)": best(zoo_cfgs),
                "oracle (upper bound)": "oracle",
                "forecast: best in hindsight on test (optimistic)": max(zoo_cfgs, key=lambda c: te[c].mean()),
            }
            simple = best(stat_cfgs + roll_cfgs)       # best no-ML rule by validation
            print(f"\n--- wake {int(W)} s (k={k}); best simple rule on val: {simple}; accuracy-selected model (val MAE): {acc_model}")
            print(f"{'strategy':52s} {'config':22s} {'return':>8s} {'SLA%':>6s} {'kWh':>6s}  {'vs reactive [95% CI]':>24s}  {'vs simple rule [95% CI]':>24s}")
            for name, c in strategies.items():
                dr = ci(te[c] - te["reactive"]); ds = ci(te[c] - te[simple])
                print(f"{name:52s} {c.replace('zoo|',''):22s} {te[c].mean():8.0f} {sla[c]:6.2f} {en[c]:6.3f}  {dr[0]:+7.0f} [{dr[1]:+.0f},{dr[2]:+.0f}]".ljust(112) + f"  {ds[0]:+7.0f} [{ds[1]:+.0f},{ds[2]:+.0f}]")
                strat_rows.append({"trace": label, "wake_s": W, "strategy": name, "config": c, "test_return": te[c].mean(), "sla_pct": sla[c], "energy_kwh": en[c],
                                   "diff_vs_reactive": dr[0], "ci_lo_r": dr[1], "ci_hi_r": dr[2], "diff_vs_simple": ds[0], "ci_lo_s": ds[1], "ci_hi_s": ds[2], "simple_rule": simple})
            # accuracy vs value (tau=0.9 and point), test split
            at = acc[acc.split == "test"].set_index("model")
            for form in ("0.9", "point"):
                for m in MODELS:
                    c = f"zoo|{m}|{form}"
                    av_rows.append({"trace": label, "wake_s": W, "model": m, "form": form, "test_mae": at.loc[m, "mae"], "test_mase": at.loc[m, "mase"],
                                    "mae_volatile": at.loc[m, "mae_volatile"], "cov90": at.loc[m, "cov90"], "val_mae": val_mae[m],
                                    "test_return": te[c].mean(), "gain_pct_vs_reactive": 100 * (te[c].mean() - te["reactive"].mean()) / abs(te["reactive"].mean()),
                                    "val_return": val[c], "sla_pct": sla[c]})
    S = pd.DataFrame(strat_rows); S.to_csv(os.path.join(a.out, "strategies.csv"), index=False)
    V = pd.DataFrame(av_rows); V.to_csv(os.path.join(a.out, "accuracy_vs_value.csv"), index=False)
    print("\n################ ACCURACY vs DECISION VALUE (test split)")
    for form in ("0.9", "point"):
        v = V[V.form == form]
        print(f"\n[decision rule: {form}]  Spearman rank correlation, forecast error (lower = better) vs gain over reactive:")
        rows = []
        for (tr, w), g in v.groupby(["trace", "wake_s"]):
            rho = stats.spearmanr(g.test_mase, g.gain_pct_vs_reactive).correlation
            rows.append((tr, int(w), rho))
        for tr, w, rho in rows:
            print(f"   {tr:14s} {w:4d}s  rho(MASE, gain) = {rho:+.2f}   (negative = accuracy helps; positive = more accurate models do WORSE)")
        r = stats.spearmanr(v.test_mase, v.gain_pct_vs_reactive)
        print(f"   pooled over {len(v)} (trace,wake,model) points: rho = {r.correlation:+.2f}, p = {r.pvalue:.3f}")
        r2 = stats.spearmanr(v.mae_volatile, v.gain_pct_vs_reactive)
        print(f"   using MAE in volatile windows: rho = {r2.correlation:+.2f}, p = {r2.pvalue:.3f}")
    print("\n################ SUMMARY: ACCURACY-selected vs DECISION-selected forecaster (difference in test return, 95% CI over episodes)")
    for (tr, w), g in S.groupby(["trace", "wake_s"]):
        g = g.set_index("strategy")
        print(f"   {tr:14s} {int(w):4d}s  decision-selected(model,tau) - accuracy-selected(tau=0.9): {g.loc['forecast: DECISION-selected (model,tau)','test_return'] - g.loc['forecast: ACCURACY-selected model, tau=0.9','test_return']:+7.0f}"
              f"   | decision-selected - best simple rule: {g.loc['forecast: DECISION-selected (model,tau)','diff_vs_simple']:+6.0f} [{g.loc['forecast: DECISION-selected (model,tau)','ci_lo_s']:+.0f},{g.loc['forecast: DECISION-selected (model,tau)','ci_hi_s']:+.0f}]")


if __name__ == "__main__":
    main()
