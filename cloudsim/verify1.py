"""Independent recomputation (no p1_analyze/p1_metrics) of the headline numbers from raw result CSVs."""
import numpy as np, pandas as pd, json
from scipy import stats
NH, VPH, RHO = 18, 2, 1.0
hm = np.array([[10000, 20000, 40000][h % 3] for h in range(NH)]); pi = np.array([[60, 90, 150][h % 3] for h in range(NH)])
C = hm.sum(); Lb = 6e4; lam = RHO * C / Lb; m = 1 / (0.55 * lam); cvm = C / (NH * VPH); x = Lb / cvm; e = m * pi.sum()
def J1000(R): return -(R.makespan / m + (R.idle_j + R.dyn_j) / e + R.late + R.resp / (5 * x)) / R.N * 1000
MODELS = ["persistence", "moving_avg", "ewma", "ridge_ar", "mlp", "cnn_lstm"]
def ci(d):
    d = np.asarray(d); mu = d.mean(); h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)); return mu, mu - h, mu + h
out = {}; rows = []
for tr in ("Alibaba", "Azure", "fastStorage", "Rnd"):
    for W in (120, 300, 600):
        R = pd.read_csv(f"res_fin_{tr}_{W}.csv"); R["J"] = J1000(R)
        assert R.groupby("config").size().eq(90).all()
        V = R[(R.part == "val") & (R.ep < 20)].pivot(index="ep", columns="config", values="J"); T = R[R.part == "test"].pivot(index="ep", columns="config", values="J")
        zoo = [c for c in V if c.startswith("zoo|")]; sr = [c for c in V if c.startswith(("static_", "rolling_"))]
        dec = V[zoo].mean().idxmax(); simple = V[sr].mean().idxmax()
        sm = pd.read_csv(f"selmae_fin_{tr}_{W}.csv").set_index("model").sel_mae; acc = "zoo|" + sm.idxmin() + "|0.9"
        r = dict(trace=tr, W=W, dec=dec, simple=simple, acc=acc)
        for nm, (a, b) in {"dec_sim": (dec, simple), "acc_sim": (acc, simple), "dec_acc": (dec, acc), "ora_rea": ("oracle", "reactive"), "sim_rea": (simple, "reactive")}.items():
            r[nm] = ci(T[a] - T[b])
        rows.append(r)
cnt = lambda nm, s: sum((r[nm][1] > 0) if s > 0 else (r[nm][2] < 0) for r in rows)
def pool(nm):
    y = np.array([r[nm][0] for r in rows]); v = np.array([((r[nm][2] - r[nm][1]) / (2 * 2.045)) ** 2 for r in rows]) + 1e-9
    w = 1 / v; mu = (w * y).sum() / w.sum(); Q = (w * (y - mu) ** 2).sum(); t2 = max(0, (Q - 11) / (w.sum() - (w ** 2).sum() / w.sum())); wr = 1 / (v + t2); mr = (wr * y).sum() / wr.sum(); s = np.sqrt(1 / wr.sum()); return mr, mr - 1.96 * s, mr + 1.96 * s
res = {f"cnt_pos_{n}": cnt(n, 1) for n in ("dec_sim", "acc_sim", "dec_acc", "ora_rea", "sim_rea")}; res.update({f"cnt_neg_{n}": cnt(n, -1) for n in ("dec_sim", "acc_sim", "dec_acc", "ora_rea", "sim_rea")})
res.update({f"pool_{n}": tuple(round(v) for v in pool(n)) for n in ("dec_sim", "acc_sim", "dec_acc", "ora_rea")})
res["static_sel"] = sum(r["simple"].startswith("static") for r in rows)
print({k: (v if not hasattr(v, "item") else v.item()) for k, v in res.items()}); pd.DataFrame(rows).to_pickle("verify1_rows.pkl")
d = json.load(open("p1_numbers.json"))
chk = {"cnt@pos@dec@sim": res["cnt_pos_dec_sim"], "cnt@neg@dec@sim": res["cnt_neg_dec_sim"], "cnt@pos@acc@sim": res["cnt_pos_acc_sim"], "cnt@neg@acc@sim": res["cnt_neg_acc_sim"], "cnt@pos@dec@acc": res["cnt_pos_dec_acc"], "cnt@neg@dec@acc": res["cnt_neg_dec_acc"], "cnt@pos@ora@rea": res["cnt_pos_ora_rea"], "cnt@pos@sim@rea": res["cnt_pos_sim_rea"], "n@static@sel": res["static_sel"]}
bad = [(k, d[k], v) for k, v in chk.items() if str(d[k]) != str(v)]; print("MISMATCH:", bad)
for n in ("dec_sim", "acc_sim", "dec_acc", "ora_rea"): print(n, res[f"pool_{n}"], d["pool@" + n.replace("_", "@")])
# worked example
R = pd.read_csv("res_fin_Alibaba_300.csv"); z = R[(R.config == "oracle") & (R.part == "test") & (R.ep == 0)].iloc[0]
print("worked:", z.N, z.makespan, round(z.makespan / m, 1), round((z.idle_j + z.dyn_j) / 1e6, 3), round((z.idle_j + z.dyn_j) / e, 1), z.late, round(z.resp, 1), round(z.resp / (5 * x), 1), "J", round(-(z.makespan / m + (z.idle_j + z.dyn_j) / e + z.late + z.resp / (5 * x)), 1), "J1000", round(float(J1000(pd.DataFrame([z])).iloc[0]), 1))
print("m,e,x", round(m, 4), round(e, 1), round(x, 3))
