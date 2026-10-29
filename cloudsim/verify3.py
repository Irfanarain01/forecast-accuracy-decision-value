import numpy as np, pandas as pd
from scipy import stats
def setup(nh, rho):
    hm = np.array([[10000, 20000, 40000][h % 3] for h in range(nh)]); pi = np.array([[60, 90, 150][h % 3] for h in range(nh)]); C = hm.sum(); Lb = 6e4
    m = 1 / (0.55 * rho * C / Lb); x = Lb / (C / (nh * 2)); e = m * pi.sum(); return m, e, x
def ci(d):
    d = np.asarray(d); mu = d.mean(); h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)); return mu, mu - h, mu + h
MODELS = ["persistence", "moving_avg", "ewma", "ridge_ar", "mlp", "cnn_lstm"]
def run(tag, nh, rho, cells, nv=20):
    m, e, x = setup(nh, rho); out = []
    for tr, W in cells:
        R = pd.read_csv(f"res_{tag}_{tr}_{W}.csv"); R["J"] = -(R.makespan / m + (R.idle_j + R.dyn_j) / e + R.late + R.resp / (5 * x)) / R.N * 1000
        V = R[(R.part == "val") & (R.ep < nv)].pivot(index="ep", columns="config", values="J"); T = R[R.part == "test"].pivot(index="ep", columns="config", values="J")
        zoo = [c for c in V if c.startswith("zoo|")]; sr = [c for c in V if c.startswith(("static_", "rolling_"))]
        dec = V[zoo].mean().idxmax(); simple = V[sr].mean().idxmax(); sm = pd.read_csv(f"selmae_{tag}_{tr}_{W}.csv").set_index("model").sel_mae; acc = "zoo|" + sm.idxmin() + "|0.9"
        out.append((tr, W, simple, ci(T[dec] - T[simple]), ci(T[acc] - T[simple]), ci(T["oracle"] - T["reactive"])[0], round(100 * (R[(R.part == 'test') & (R.config == 'reactive')].late / R[(R.part == 'test') & (R.config == 'reactive')].N).mean(), 1)))
    return out
allc = [(t, W) for t in ("Alibaba", "Azure", "fastStorage", "Rnd") for W in (120, 300, 600)]
sw = run("finsw", 18, 1.0, allc); print("swapped: dec wins", sum(o[3][1] > 0 for o in sw), "losses", sum(o[3][2] < 0 for o in sw), "acc wins", sum(o[4][1] > 0 for o in sw), "losses", sum(o[4][2] < 0 for o in sw))
v60 = run("fin", 18, 1.0, allc, 60); print("V60: dec wins", sum(o[3][1] > 0 for o in v60), "losses", sum(o[3][2] < 0 for o in v60), "acc wins", sum(o[4][1] > 0 for o in v60), "losses", sum(o[4][2] < 0 for o in v60))
for tag, nh, rho in (("fin", 18, 1.0), ("finlo", 18, 0.6), ("finhi", 18, 1.3), ("finbig", 54, 1.0)):
    for o in run(tag, nh, rho, [("Alibaba", 300), ("fastStorage", 300)]): print(tag, o)
