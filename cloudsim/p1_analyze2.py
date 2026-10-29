"""Analysis of CloudSim Plus results for paper 1. python p1_analyze.py <tag> <hosts> <vph> <rho>   (reads res_<tag>_<trace>_<W>.csv)"""
import sys, glob, numpy as np, pandas as pd
from scipy import stats
from evalad import DIRS
from p1_metrics import add_components
MODELS = ["persistence", "moving_avg", "ewma", "ridge_ar", "mlp", "cnn_lstm"]
WS = {"paper weights (1,1,1,1)": (1, 1, 1, 1), "SLA x10": (1, 1, 10, 1), "SLA x50": (1, 1, 50, 1), "energy x3": (1, 3, 1, 1), "energy x0.33": (1, .33, 1, 1), "no makespan": (0, 1, 1, 1), "response x5": (1, 1, 1, 5)}
def ci(d):
    d = np.asarray(d, float); m = d.mean()
    if len(d) < 2 or d.std() == 0: return m, m, m, 0.0
    lo, hi = stats.t.interval(.95, len(d) - 1, m, stats.sem(d)); return m, lo, hi, m / d.std(ddof=1)
def load(tag, nh, vph, rho):
    out = {}
    for f in sorted(glob.glob(f"res_{tag}_*_*.csv")):
        _, _, tr, W = f[:-4].split("_"); R0 = pd.read_csv(f)
        if R0.config.nunique() < 55 or (R0.groupby("config").size().min() < 50): continue
        R = add_components(R0, nh, vph, rho); out[(tr, int(W))] = R
    return out
NV_USE = 20
def cell_stats(R, tr, W, w=(1, 1, 1, 1)):
    wm, we, ws, wr = w; R = R[(R.part == "test") | (R.ep < NV_USE)].copy(); R["Jw"] = -(wm * R.c_mk + we * R.c_en + ws * R.c_late + wr * R.c_resp); R["Jw"] = R.Jw / R.N * 1000
    val = R[R.part == "val"].groupby("config").Jw.mean(); te = R[R.part == "test"].pivot(index="ep", columns="config", values="Jw")
    vm = pd.read_csv(f"selmae_{TAG}_{tr}_{W}.csv").set_index("model").sel_mae; acc = f"zoo|{min(MODELS, key=lambda m: vm[m])}|0.9"
    zoo = [c for c in te.columns if c.startswith("zoo|")]; sr = [c for c in te.columns if c.startswith(("static_", "rolling_"))]
    best = lambda cf: max(cf, key=lambda c: val[c]); simple = best(sr); dec = best(zoo)
    return R, val, te, acc, simple, dec, zoo, sr
def main(tag, nh, vph, rho):
    global TAG, NV_USE; TAG = tag; NV_USE = int(__import__("os").environ.get("NV_USE", "20"))
    cells = load(tag, nh, vph, rho); rows = []; comp = []
    for (tr, W), R in sorted(cells.items()):
        for wn, w in WS.items():
            R2, val, te, acc, simple, dec, zoo, sr = cell_stats(R, tr, W, w)
            r = dict(trace=tr, W=W, weights=wn, simple=simple, dec=dec, acc=acc)
            for nm, (x, y) in {"oracle-react": ("oracle", "reactive"), "dec-simple": (dec, simple), "acc-simple": (acc, simple), "dec-acc": (dec, acc), "simple-react": (simple, "reactive")}.items():
                m, lo, hi, dz = ci(te[x] - te[y]); r.update({nm: m, nm + "_lo": lo, nm + "_hi": hi, nm + "_dz": dz, nm + "_pos": lo > 0, nm + "_neg": hi < 0})
            rows.append(r)
            if wn.startswith("paper"):
                T = R[R.part == "test"].groupby("config").agg(sla=("sla", "mean"), kwh1000=("kwh", lambda s: 0), N=("N", "mean"))
                for nm, c in {"reactive": "reactive", "always_on": "always_on", "oracle": "oracle", "simple": simple, "accuracy-selected": acc, "return-selected": dec}.items():
                    x = R[(R.part == "test") & (R.config == c)]
                    comp.append(dict(trace=tr, W=W, strategy=nm, config=c, J1000=(x.J1000).mean(), sla=x.sla.mean(), kwh_per1000=(x.kwh / x.N * 1000).mean(), resp_per_task=(x.resp / x.N).mean(), makespan=x.makespan.mean(), wakes=x.wakes.mean()))
    S = pd.DataFrame(rows); S.to_csv(f"p1_{tag}{NV_USE}_strategies.csv", index=False); pd.DataFrame(comp).to_csv(f"p1_{tag}{NV_USE}_components.csv", index=False)
    pd.set_option("display.width", 220)
    g = S.groupby("weights", sort=False).agg(cells=("trace", "count"), acc_beats_simple=("acc-simple_pos", "sum"), dec_beats_simple=("dec-simple_pos", "sum"), dec_loses_to_simple=("dec-simple_neg", "sum"),
        dec_beats_acc=("dec-acc_pos", "sum"), acc_beats_dec=("dec-acc_neg", "sum"), mean_dec_minus_simple=("dec-simple", "mean"), mean_oracle_minus_react=("oracle-react", "mean"), mean_simple_minus_react=("simple-react", "mean"))
    base = S[S.weights.str.startswith("paper")].set_index(["trace", "W"])
    g["simple_rule_changes"] = [(S[S.weights == w].set_index(["trace", "W"]).simple != base.simple).sum() for w in g.index]
    print(f"=== {tag}: {len(cells)} cells, fleet {nh} hosts x {vph} VMs, rho {rho}"); print(g.round(1).to_string())
    # random-effects pooling of cell-level differences (paper weights)
    for nm in ("dec-simple", "acc-simple", "dec-acc", "oracle-react"):
        b = S[S.weights.str.startswith("paper")]; se = (b[nm + "_hi"] - b[nm + "_lo"]) / (2 * 2.045); y = b[nm].values; v = (se ** 2).values + 1e-9
        wfx = 1 / v; mu = (wfx * y).sum() / wfx.sum(); Q = (wfx * (y - mu) ** 2).sum(); k = len(y); tau2 = max(0, (Q - (k - 1)) / (wfx.sum() - (wfx ** 2).sum() / wfx.sum()))
        wr = 1 / (v + tau2); mur = (wr * y).sum() / wr.sum(); ser = np.sqrt(1 / wr.sum()); print(f"  pooled {nm:13s}: {mur:+8.1f} [{mur-1.96*ser:+.1f},{mur+1.96*ser:+.1f}]  between-cell sd {np.sqrt(tau2):.1f}  cells>0 {int((y>0).sum())}/{k}")
    return S
if __name__ == "__main__": main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]))
