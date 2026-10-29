"""Builds tables, numbers and figures for the revised paper 1 from CloudSim Plus results."""
import sys, glob, json, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from evalad import DIRS
from p1_metrics import add_components
import p1_analyze2 as A
from p1_analyze2 import cell_stats, ci, MODELS, WS
A.TAG = "fin"; A.NV_USE = 20
import os; P = os.environ.get("OUT", "out").rstrip("/") + "/"; [os.makedirs(P + d, exist_ok=True) for d in ("tables", "figures")]; TR = ["Alibaba", "Azure", "fastStorage", "Rnd"]; WK = [120, 300, 600]
LAB = {"Alibaba": "Alibaba", "Azure": "Azure Functions", "fastStorage": "Bitbrains fastStorage", "Rnd": "Bitbrains Rnd"}
ML = {"persistence": "Persistence", "moving_avg": "Moving average", "ewma": "EWMA", "ridge_ar": "Ridge AR", "mlp": "MLP", "cnn_lstm": "CNN-LSTM"}
TAUS = ["0.5", "0.8", "0.9", "0.95", "0.99", "point"]
NUM = {}
def put(k, v): NUM[k.replace("_", "@")] = v
def m(x): return ("\\textminus{}" if x < 0 else "+") + f"{abs(x):.0f}"
def fm(x, d=0): return ("\\textminus{}" if x < 0 else "") + f"{abs(x):.{d}f}"
def iv(a, lo, hi): return f"{m(a)} [{m(lo)}, {m(hi)}]"
def cfg(c):
    if c.startswith("zoo|"): _, mo, t = c.split("|"); return f"{ML[mo]}, {'point' if t == 'point' else t}"
    if c.startswith("static_"): return f"static pool {c[7:]}/6"
    if c.startswith("rolling"): _, q, L = c.split("_"); return f"rolling $q{{=}}{q[1:]}$, $L{{=}}{L[1:]}$"
    return c
def load(tag, nh, rho):
    out = {}
    for f in sorted(glob.glob(f"res_{tag}_*_*.csv")):
        _, _, tr, W = f[:-4].split("_"); R0 = pd.read_csv(f)
        if R0.config.nunique() < 55 or R0.groupby("config").size().min() < 50: continue
        out[(tr, int(W))] = add_components(R0, nh, 2, rho)
    return out
cells = load("fin", 18, 1.0); assert len(cells) == 12
rows = []; K = {}; comp = []; opt = []
for (tr, W), R in sorted(cells.items()):
    R2, val, te, acc, simple, dec, zoo, sr = cell_stats(R, tr, W)
    vv = R[(R.part == "val") & (R.ep < 20)].copy(); vv["Jw"] = vv.J / vv.N * 1000; vpiv = vv.pivot(index="ep", columns="config", values="Jw")
    ts = lambda c: te[c].mean()
    d = {}
    for nm, (x, y) in {"ora_rea": ("oracle", "reactive"), "sim_rea": (simple, "reactive"), "dec_sim": (dec, simple), "acc_sim": (acc, simple), "dec_acc": (dec, acc), "dec_rea": (dec, "reactive"), "acc_rea": (acc, "reactive")}.items():
        d[nm] = ci(te[x] - te[y])
    hz = max(zoo, key=ts); hs = max(sr, key=ts)
    K[(tr, W)] = dict(simple=simple, dec=dec, acc=acc, d=d, te=te, hz=hz, hs=hs, zoo=zoo, sr=sr)
    S = lambda c: R[(R.part == "test") & (R.config == c)]
    for nm, c in {"Reactive": "reactive", "Always on": "always_on", "Oracle forecast": "oracle", "Best simple rule (validation)": simple, "Accuracy-selected forecaster": acc, "Decision-selected forecaster": dec, "Best forecaster in hindsight": hz, "Best simple rule in hindsight": hs}.items():
        x = S(c); comp.append(dict(trace=tr, W=W, strategy=nm, J=x.J1000.mean(), sla=x.sla.mean(), kwh=(x.kwh / x.N * 1000).mean(), resp=(x.resp / x.N).mean(), mk=x.makespan.mean(),
                                   en_sh=x.c_en.sum() / (x.c_mk + x.c_en + x.c_late + x.c_resp).sum(), mk_sh=x.c_mk.sum() / (x.c_mk + x.c_en + x.c_late + x.c_resp).sum(), late_sh=x.c_late.sum() / (x.c_mk + x.c_en + x.c_late + x.c_resp).sum(), resp_sh=x.c_resp.sum() / (x.c_mk + x.c_en + x.c_late + x.c_resp).sum()))
    # selection optimism: winner vs rest on validation
    vz = vpiv[zoo]; w = vz.mean().idxmax(); gap = [(vz[w] - vz[c]).mean() / ((vz[w] - vz[c]).std(ddof=1) / np.sqrt(len(vz))) for c in zoo if c != w]
    near = sum(g < 2 for g in gap); opt.append(dict(trace=tr, W=W, near=near, runner_gap=min((vz[w] - vz[c]).mean() for c in zoo if c != w), optimism=te[max(zoo, key=lambda c: te[c].mean())].mean() - te[w].mean()))
C = pd.DataFrame(comp); O = pd.DataFrame(opt)
# ---------------- numbers
for tr in TR:
    for W in WK:
        k = K[(tr, W)]; d = k["d"]
        for nm in d: put(f"c_{nm}_{tr}_{W}", fm(d[nm][0]))
        x = C[(C.trace == tr) & (C.W == W)].set_index("strategy"); put(f"c_sla_react_{tr}_{W}", f"{x.loc['Reactive','sla']:.1f}"); put(f"c_sla_on_{tr}_{W}", f"{x.loc['Always on','sla']:.2f}")
cnt = lambda nm, sign: sum((K[k]["d"][nm][1] > 0) if sign > 0 else (K[k]["d"][nm][2] < 0) for k in K)
for nm in ("dec_sim", "acc_sim", "dec_acc"):
    put(f"cnt_pos_{nm}", cnt(nm, 1)); put(f"cnt_neg_{nm}", cnt(nm, -1))
    y = np.array([K[k]["d"][nm][0] for k in K]); se = np.array([(K[k]["d"][nm][2] - K[k]["d"][nm][1]) / (2 * 2.045) for k in K]) ** 2 + 1e-9
    wf = 1 / se; mu = (wf * y).sum() / wf.sum(); Q = (wf * (y - mu) ** 2).sum(); t2 = max(0, (Q - 11) / (wf.sum() - (wf ** 2).sum() / wf.sum())); wr = 1 / (se + t2); mr = (wr * y).sum() / wr.sum(); sr_ = np.sqrt(1 / wr.sum())
    put(f"pool_{nm}", f"{mr:+.0f} [{mr-1.96*sr_:+.0f}, {mr+1.96*sr_:+.0f}]".replace("-", "\\textminus{}")); put(f"cells_pos_{nm}", int((y > 0).sum()))
    put(f"tau_{nm}", f"{np.sqrt(t2):.0f}")
y = np.array([K[k]["d"]["ora_rea"][0] for k in K]); se = np.array([(K[k]["d"]["ora_rea"][2] - K[k]["d"]["ora_rea"][1]) / (2 * 2.045) for k in K]) ** 2 + 1e-9
wf = 1 / se; mu = (wf * y).sum() / wf.sum(); Q = (wf * (y - mu) ** 2).sum(); t2 = max(0, (Q - 11) / (wf.sum() - (wf ** 2).sum() / wf.sum())); wr = 1 / (se + t2); mr = (wr * y).sum() / wr.sum(); sr_ = np.sqrt(1 / wr.sum())
put("pool_ora_rea", f"+{mr:.0f} [+{mr-1.96*sr_:.0f}, +{mr+1.96*sr_:.0f}]"); put("cnt_pos_ora_rea", cnt("ora_rea", 1)); put("cnt_neg_ora_rea", cnt("ora_rea", -1))
put("cnt_pos_sim_rea", cnt("sim_rea", 1)); put("cnt_neg_sim_rea", cnt("sim_rea", -1))
put("opt_near_med", f"{O.near.median():.1f}"); put("opt_gap_min", f"{O.runner_gap.min():.1f}"); put("opt_gap_max", f"{O.runner_gap.max():.1f}"); put("opt_mean", f"{O.optimism.mean():.0f}"); put("opt_max", f"{O.optimism.max():.0f}"); put("opt_min", f"{O.optimism.min():.0f}")
# strategy means over cells
G = C.groupby("strategy")[["J", "sla", "kwh", "resp", "mk", "en_sh", "mk_sh", "late_sh", "resp_sh"]].mean()
for s in G.index:
    sk = s.replace(" ", "").replace("(", "").replace(")", "")
    for col in G.columns: put(f"g_{col}_{sk}", f"{G.loc[s,col]:.2f}" if col.endswith("_sh") or col in ("sla", "kwh") else f"{G.loc[s,col]:.0f}")
for col in ("en_sh", "mk_sh", "late_sh", "resp_sh"):
    x = C[C.strategy == "Decision-selected forecaster"][col]; put(f"share_{col}_lo", f"{100*x.min():.0f}"); put(f"share_{col}_hi", f"{100*x.max():.0f}")
# ---------------- strategy table W=300
lines = []
order = ["Reactive", "Always on", "Best simple rule (validation)", "Accuracy-selected forecaster", "Decision-selected forecaster", "Oracle forecast", "Best forecaster in hindsight", "Best simple rule in hindsight"]
for s in order:
    cs = []
    for tr in TR:
        x = C[(C.trace == tr) & (C.W == 300) & (C.strategy == s)].iloc[0]; cs.append(f"{fm(x.J)} ({x.sla:.1f})")
    lines.append(s.replace("hindsight", "hindsight$^\\dagger$") + " & " + " & ".join(cs) + " \\\\")
open(P + "tables/tab_strat.tex", "w").write("""\\begin{table}[t]
\\centering
\\caption{Mean test return per 1000 tasks at the 300 s wake-up time over 30 shared episodes (CloudSim Plus, 18 hosts), with the SLA violation rate in percent in brackets. Higher return is better. The simple rule (static pool or rolling percentile) and both forecaster selections are chosen on 20 selection episodes. Rows marked $\\dagger$ pick by test return, so they are not deployable and serve as ceilings.}
\\label{tab:strat}
\\small
\\begin{tabular}{@{}lrrrr@{}}
\\toprule
Strategy & Alibaba & Azure Functions & BB fastStorage & BB Rnd \\\\
\\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
# ---------------- all-cell table
L = []
for tr in TR:
    for W in WK:
        k = K[(tr, W)]; d = k["d"]
        L.append(f"{LAB[tr] if W == 120 else ''} & {W} & {cfg(k['acc'])} & {cfg(k['dec'])} & {cfg(k['simple'])} & {iv(*d['dec_acc'][:3])} & {iv(*d['dec_sim'][:3])} & {iv(*d['acc_sim'][:3])} \\\\")
    L.append("\\addlinespace[2pt]")
open(P + "tables/tab_sel.tex", "w").write("""\\begin{table*}[htbp]
\\centering
\\caption{Selection comparison in all twelve cells (CloudSim Plus, 18 hosts, 30 paired test episodes; residual quantiles from the calibration half, selection on the selection half of the validation part). Differences are mean test return per 1000 tasks with 95\\% $t$-intervals. The simple rule is the best static pool or rolling percentile on validation.}
\\label{tab:sel}
\\scriptsize
\\setlength{\\tabcolsep}{3pt}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{@{}lrlllrrr@{}}
\\toprule
Trace & $W$ (s) & Accuracy-sel. & Decision-sel. & Simple rule & Dec.\\ $-$ acc. & Dec.\\ $-$ simple & Acc.\\ $-$ simple \\\\
\\midrule
""" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")
# ---------------- weight sensitivity (recompute with selection)
WR = []
for wn, w in WS.items():
    cp = cd = cl = ca = ch = chn = 0; chg = 0; md = []
    for (tr, W), R in sorted(cells.items()):
        R2, val, te, acc, simple, dec, zoo, sr = cell_stats(R, tr, W, w)
        a = ci(te[acc] - te[simple]); b = ci(te[dec] - te[simple]); e = ci(te[dec] - te[acc]); md.append(b[0])
        ca += a[1] > 0; cp += b[1] > 0; cl += b[2] < 0; ch += e[1] > 0; chn += e[2] < 0; chg += simple != K[(tr, W)]["simple"]
    WR.append((wn, ca, cp, cl, ch, chn, chg, np.mean(md)))
    put("w_" + wn.split()[0] + "_" + str(len(WR)), f"{ca}/{cp}/{cl}")
put("wmaxchg", int(max(r[6] for r in WR)))
put("wminchg", int(min(r[6] for r in WR[1:])))
def wlab(s): return s.replace("(1,1,1,1)", "").replace("x", "$\\times$").strip()
open(P + "tables/tab_wsens.tex", "w").write("""\\begin{table}[t]
\\centering
\\caption{Sensitivity of the main conclusions to the return weights (twelve cells, CloudSim Plus). Each row multiplies one term of Eq.~\\eqref{eq:return} and re-runs every selection on the selection half. Columns count cells where a forecaster selection beats (+) or loses to ($-$) the simple rule chosen on the selection half with a 95\\% interval excluding zero, and cells where the selected simple rule differs from the one chosen under the paper weights.}
\\label{tab:wsens}
\\small
\\begin{tabular}{@{}lrrrrr@{}}
\\toprule
Weights & Acc.\\ $+$ & Dec.\\ $+$ & Dec.\\ $-$ & Dec.\\ $>$ acc. & Rule changes \\\\
\\midrule
""" + "\n".join(f"{wlab(wn)} & {ca} & {cp} & {cl} & {ch} & {chg} \\\\" for wn, ca, cp, cl, ch, chn, chg, _ in WR) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
json.dump([list(map(lambda v: v if isinstance(v, str) else float(v), r)) for r in WR], open("wsens.json", "w"))
# ---------------- component table
G2 = G.loc[order[:6]]
open(P + "tables/tab_comp.tex", "w").write("""\\begin{table}[t]
\\centering
\\caption{Components of the outcome averaged over the twelve cells (CloudSim Plus, 18 hosts). Return is per 1000 tasks. Energy is in kWh per 1000 tasks, response time in seconds per task, and makespan in seconds per episode.}
\\label{tab:comp}
\\small
\\begin{tabular}{@{}lrrrrr@{}}
\\toprule
Strategy & Return & SLA viol.\\ (\\%) & Energy & Response & Makespan \\\\
\\midrule
""" + "\n".join(f"{s} & {fm(G2.loc[s,'J'])} & {G2.loc[s,'sla']:.1f} & {G2.loc[s,'kwh']:.3f} & {G2.loc[s,'resp']:.1f} & {G2.loc[s,'mk']:.0f} \\\\" for s in G2.index) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
# ---------------- accuracy vs value, tau, model spread
EXACT = {}; rho_rows = []; tau_sp = {}; mod_sp = {}; mid_sp = {}; best_tau = {}; allpts = {"0.9": [], "point": []}
for (tr, W), k in K.items():
    a = pd.read_csv(DIRS[tr] + "/accuracy.csv"); a = a[(a.wake_s == W) & (a.split == "test")].set_index("model").mase
    te = k["te"]; Jr = te["reactive"].mean()
    for form in ("0.9", "point"):
        g = {mo: 100 * (te[f"zoo|{mo}|{form}"].mean() - Jr) / abs(Jr) for mo in MODELS}
        r = stats.spearmanr([a[mo] for mo in MODELS], [g[mo] for mo in MODELS])[0]; EXACT.setdefault(form, []).append(r); put(f"rho_{form}_{tr}_{W}", f"{r:.2f}".replace("-", "\\textminus{}"))
        allpts[form] += [(a[mo], g[mo]) for mo in MODELS]
    M = pd.DataFrame({t: [te[f"zoo|{mo}|{t}"].mean() for mo in MODELS] for t in TAUS}, index=MODELS)
    tau_sp[(tr, W)] = M.mean(0).max() - M.mean(0).min(); mod_sp[(tr, W)] = M["0.9"].max() - M["0.9"].min(); mid_sp[(tr, W)] = M[TAUS[:4]].mean(0).max() - M[TAUS[:4]].mean(0).min(); best_tau[(tr, W)] = M.mean(0).idxmax()
    k["M"] = M
for form in ("0.9", "point"):
    x = np.array(allpts[form]); r, p = stats.spearmanr(x[:, 0], x[:, 1]); put(f"rho_{form}_pooled", f"{r:.2f}".replace("-", "\\textminus{}")); put(f"rho_{form}_p", f"{p:.2f}")
    cell_r = EXACT[form]; put(f"rho_{form}_mean", f"{np.mean(cell_r):.2f}".replace("-", "\\textminus{}")); put(f"rho_{form}_neg", sum(v < 0 for v in cell_r)); put(f"rho_{form}_pos", sum(v > 0 for v in cell_r))
for tr in TR:
    put(f"tauspread_{tr}", f"{tau_sp[(tr,300)]:.0f}"); put(f"modspread_{tr}", f"{mod_sp[(tr,300)]:.0f}"); put(f"midspread_{tr}", f"{mid_sp[(tr,300)]:.0f}"); put(f"besttau_{tr}", best_tau[(tr, 300)].replace("point", "the point form"))
put("tauspread_mean", f"{np.mean(list(tau_sp.values())):.0f}"); put("modspread_mean", f"{np.mean(list(mod_sp.values())):.0f}"); put("midspread_mean", f"{np.mean(list(mid_sp.values())):.0f}")
put("tau_gt_mod_cells", sum(tau_sp[k] > mod_sp[k] for k in K)); put("tau_gt_mid_cells", sum(mid_sp[k] > mod_sp[k] for k in K))
put("besttau_distinct", len(set(best_tau.values())))
# ---------------- overload frequency
over = {}
for tr in TR:
    import pickle
    zoo_, rq, val_, test_ = __import__("adaptive_tau").load_zoo(DIRS[tr] + "/zoo.pkl"); s = np.asarray(test_, float)
    for rho in (0.6, 1.0, 1.3):
        over[(tr, rho)] = 100 * np.mean(1.2 * rho * s > 1.0); put(f"over_{tr}_{rho}", f"{over[(tr, rho)]:.1f}"); put(f"load_max_{tr}", f"{s.max():.2f}")
# ---------------- scale sensitivity table
SC = []
for tag, nh, rho, lab in (("fin", 18, 1.0, "18 hosts, load 1.0"), ("finlo", 18, 0.6, "18 hosts, load 0.6"), ("finhi", 18, 1.3, "18 hosts, load 1.3"), ("finbig", 54, 1.0, "54 hosts, load 1.0")):
    cl = load(tag, nh, rho); A.TAG = tag
    for tr in ("Alibaba", "fastStorage"):
        R = cl[(tr, 300)]; R2, val, te, acc, simple, dec, zoo, sr = cell_stats(R, tr, 300); x = R[R.part == "test"]
        SC.append((lab, tr, ci(te["oracle"] - te["reactive"]), ci(te[simple] - te["reactive"]), ci(te[dec] - te[simple]), ci(te[acc] - te[simple]), x[x.config == "reactive"].sla.mean(), x[x.config == "always_on"].sla.mean()))
open(P + "tables/tab_scale.tex", "w").write("""\\begin{table}[t]
\\centering
\\caption{Sensitivity to fleet size and load at the 300 s wake-up time (CloudSim Plus). Entries are mean test return per 1000 tasks over 30 paired episodes; brackets give 95\\% intervals for the selection differences. Load is the arrival intensity relative to the capacity of the whole fleet. SLA columns give the violation rate in percent.}
\\label{tab:scale}
\\scriptsize
\\begin{tabular}{@{}llrrrrrr@{}}
\\toprule
Setting & Trace & Oracle $-$ reactive & Simple $-$ reactive & Dec.\\ $-$ simple & Acc.\\ $-$ simple & SLA reactive & SLA always on \\\\
\\midrule
""" + "\n".join(f"{l} & {LAB[t]} & {m(o[0])} & {m(s_[0])} & {iv(*d_[:3])} & {iv(*a_[:3])} & {r_:.1f} & {a2:.2f} \\\\" for l, t, o, s_, d_, a_, r_, a2 in SC) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
for l, t, o, s_, d_, a_, r_, a2 in SC: put("sc_dec_sim_" + l.split(",")[1].strip().replace(" ", "") + l.split()[0] + "_" + t, f"{d_[0]:.0f}")

# ---------------- extra numbers
hind = [C[(C.trace == t) & (C.W == W) & (C.strategy == "Best forecaster in hindsight")].J.iloc[0] - C[(C.trace == t) & (C.W == W) & (C.strategy == "Best simple rule in hindsight")].J.iloc[0] for t in TR for W in WK]
put("hind_fc_gt_rule", sum(h > 0 for h in hind)); put("hind_mean", f"{np.mean(hind):+.0f}".replace("-", "\\textminus{}"))
for mo, ab in (("persistence", "pers"), ("mlp", "mlp"), ("ridge_ar", "ridge")):
    x = cells[("fastStorage", 300)]; x = x[(x.part == "test") & (x.config == f"zoo|{mo}|point")]; put(f"fs300_{ab}_pt_sla", f"{x.sla.mean():.1f}"); put(f"fs300_{ab}_pt_J", fm(x.J1000.mean())); put(f"fs300_{ab}_pt_kwh", f"{(x.kwh/x.N*1000).mean():.3f}")
nst = sum(K[k]["simple"].startswith("static") for k in K); put("n_static_sel", nst)
# rho table
RL = []
for tr in TR:
    RL.append(f"{LAB[tr]} & " + " & ".join(f"{NUM[f'rho@0.9@{tr}@{W}']} / {NUM[f'rho@point@{tr}@{W}']}" for W in WK) + " \\\\")
open(P + "tables/tab_rho.tex", "w").write("""\\begin{table}[t]
\\centering
\\caption{Spearman correlation between test MASE and gain over reactive control across the six models in each cell (CloudSim Plus). Each entry gives the value for the $\\tau=0.9$ bound and for the point forecast. A negative value means that more accurate models gain more.}
\\label{tab:rho}
\\small
\\begin{tabular}{@{}lrrr@{}}
\\toprule
Trace & $W=120$ s & $W=300$ s & $W=600$ s \\\\
\\midrule
""" + "\n".join(RL) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
# ---------------- figures
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})
Cc = {"blue": "#0072B2", "orange": "#D55E00", "sky": "#56B4E9", "grey": "#666666"}
fig, ax = plt.subplots(2, 2, figsize=(6.6, 4.6))
for a_, tr in zip(ax.ravel(), TR):
    for key, ls, mk, lab, col, off in (("ora_rea", "-", "o", "Oracle forecast", Cc["blue"], -4), ("sim_rea", "--", "s", "Best simple rule", Cc["orange"], 4)):
        v = np.array([K[(tr, W)]["d"][key] for W in WK]); a_.errorbar(np.array(WK) + off, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=col, ls=ls, marker=mk, ms=4.5, lw=1.1, capsize=2.5, mfc=col if mk == "o" else "white", label=lab)
    a_.axhline(0, color="0.5", lw=0.6); a_.set_xticks(WK); a_.set_title(LAB[tr], fontsize=9); a_.set_xlabel("Wake-up time (s)"); a_.set_ylabel("Return gain over reactive")
ax[0, 0].legend(frameon=False, fontsize=8, loc="upper left"); fig.tight_layout(); fig.savefig(P + "figures/fig_wake.pdf"); fig.savefig(P + "figures/fig_wake.png", dpi=300); plt.close(fig)
MC = {"persistence": "#666666", "moving_avg": "#E69F00", "ewma": "#56B4E9", "ridge_ar": "#009E73", "mlp": "#CC79A7", "cnn_lstm": "#D55E00"}; MK = {"persistence": "o", "moving_avg": "s", "ewma": "^", "ridge_ar": "D", "mlp": "v", "cnn_lstm": "P"}
fig, ax = plt.subplots(2, 2, figsize=(6.6, 4.8))
for a_, tr in zip(ax.ravel(), TR):
    k = K[(tr, 300)]; te = k["te"]; Jr = te["reactive"].mean(); acc_ = pd.read_csv(DIRS[tr] + "/accuracy.csv"); acc_ = acc_[(acc_.wake_s == 300) & (acc_.split == "test")].set_index("model").mase
    for mo in MODELS:
        for form, fill in (("0.9", True), ("point", False)):
            g = 100 * (te[f"zoo|{mo}|{form}"].mean() - Jr) / abs(Jr); a_.scatter(acc_[mo], g, marker=MK[mo], s=28, c=MC[mo] if fill else "white", edgecolors=MC[mo], linewidths=1)
    a_.set_title(f"{LAB[tr]} ($\\rho_{{0.9}}$={NUM[f'rho@0.9@{tr}@300'].replace(chr(92)+'textminus{}','-')}, $\\rho_{{pt}}$={NUM[f'rho@point@{tr}@300'].replace(chr(92)+'textminus{}','-')})", fontsize=8.5); a_.set_xlabel("Test MASE"); a_.set_ylabel("Gain over reactive (%)")
h = [plt.Line2D([], [], marker=MK[mo], color="w", markerfacecolor=MC[mo], markeredgecolor=MC[mo], label=ML[mo], ms=5) for mo in MODELS]
fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.04)); fig.tight_layout(); fig.savefig(P + "figures/fig_scatter.pdf"); fig.savefig(P + "figures/fig_scatter.png", dpi=300); plt.close(fig)
fig, ax = plt.subplots(2, 2, figsize=(6.6, 4.6))
for a_, tr in zip(ax.ravel(), TR):
    M = K[(tr, 300)]["M"]; best = M.mean(0).max(); rel = M - best; x = np.arange(6)
    a_.fill_between(x, rel.min(0), rel.max(0), color=Cc["sky"], alpha=0.35, lw=0, label="Range over six models"); a_.plot(x, rel.mean(0), "-o", color=Cc["blue"], ms=4, lw=1.2, label="Mean over six models"); a_.plot(x, rel.loc["cnn_lstm"], "--s", color=Cc["orange"], mfc="white", ms=4, lw=1.1, label="CNN-LSTM")
    a_.set_xticks(x); a_.set_xticklabels(TAUS); a_.set_title(LAB[tr], fontsize=9); a_.set_xlabel("Bound level $\\tau$"); a_.set_ylabel("Return relative to best mean")
ax[0, 0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(P + "figures/fig_tau.pdf"); fig.savefig(P + "figures/fig_tau.png", dpi=300); plt.close(fig)
# ---------------- write numbers file
with open(P + "tables/numbers_cs.tex", "w") as f:
    f.write("% generated by p1_paper.py\n\\makeatletter\n")
    for k_, v in NUM.items(): f.write("\\expandafter\\def\\csname num@" + k_.replace("_", "@") + "\\endcsname{" + str(v) + "}\n")
    f.write("\\makeatother\n")
json.dump({k: str(v) for k, v in NUM.items()}, open("p1_numbers.json", "w"), indent=0)
print(len(NUM)); print(O.to_string()); print(G.round(3).to_string())
for k in ["pool_dec_sim", "pool_acc_sim", "pool_dec_acc", "pool_ora_rea", "rho@0.9@pooled", "rho@point@pooled", "rho@0.9@p", "rho@point@p", "tauspread@mean", "modspread@mean", "midspread@mean", "tau@gt@mod@cells", "tau@gt@mid@cells", "besttau@distinct", "rho@0.9@neg", "rho@point@neg"]: print(k, NUM.get(k.replace("_", "@")))
for tr in TR: print(tr, [NUM[f"{n}@{tr}"] for n in ("tauspread", "modspread", "midspread", "besttau")], [NUM[f"over@{tr}@{r}"] for r in (0.6, 1.0, 1.3)], NUM[f"load@max@{tr}"])
C.to_csv("p1_paper_components.csv", index=False)
