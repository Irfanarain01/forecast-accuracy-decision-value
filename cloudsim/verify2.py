import numpy as np, pandas as pd
from scipy import stats
exec(open("verify1.py").read().split("MODELS =")[0])
MODELS = ["persistence", "moving_avg", "ewma", "ridge_ar", "mlp", "cnn_lstm"]
def ci(d):
    d = np.asarray(d); mu = d.mean(); h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)); return mu - h, mu + h
WS = {"paper": (1, 1, 1, 1), "SLA10": (1, 1, 10, 1), "SLA50": (1, 1, 50, 1), "en3": (1, 3, 1, 1), "en.33": (1, .33, 1, 1), "nomk": (0, 1, 1, 1), "resp5": (1, 1, 1, 5)}
base_simple = {}
for wn, (wm, we, ws, wr) in WS.items():
    ap = dp = dn = dacc = chg = 0
    for tr in ("Alibaba", "Azure", "fastStorage", "Rnd"):
        for W in (120, 300, 600):
            R = pd.read_csv(f"res_fin_{tr}_{W}.csv")
            R["J"] = -(wm * R.makespan / m + we * (R.idle_j + R.dyn_j) / e + ws * R.late + wr * R.resp / (5 * x)) / R.N * 1000
            V = R[(R.part == "val") & (R.ep < 20)].pivot(index="ep", columns="config", values="J"); T = R[R.part == "test"].pivot(index="ep", columns="config", values="J")
            zoo = [c for c in V if c.startswith("zoo|")]; sr = [c for c in V if c.startswith(("static_", "rolling_"))]
            dec = V[zoo].mean().idxmax(); simple = V[sr].mean().idxmax(); sm = pd.read_csv(f"selmae_fin_{tr}_{W}.csv").set_index("model").sel_mae; acc = "zoo|" + sm.idxmin() + "|0.9"
            lo, hi = ci(T[dec] - T[simple]); dp += lo > 0; dn += hi < 0
            lo, hi = ci(T[acc] - T[simple]); ap += lo > 0
            lo, hi = ci(T[dec] - T[acc]); dacc += lo > 0
            if wn == "paper": base_simple[(tr, W)] = simple
            chg += simple != base_simple[(tr, W)]
    print(wn, "acc+", ap, "dec+", dp, "dec-", dn, "dec>acc", dacc, "rule changes", chg)
