"""New figures for paper 1: per-cell selection comparison and design robustness, from the final-run CSVs."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from p1_extra import pool
import os; P = os.environ.get("OUT", "out").rstrip("/") + "/"; [os.makedirs(P + d, exist_ok=True) for d in ("tables", "figures")]
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})
LAB = {"Alibaba": "Alibaba", "Azure": "Azure Fn", "fastStorage": "fastStorage", "Rnd": "Rnd"}
def prim(tag, nv):
    S = pd.read_csv(f"p1_{tag}{nv}_strategies.csv"); return S[S.weights.str.startswith("paper")].reset_index(drop=True)
S = prim("fin", 20)
TR = ["Alibaba", "Azure", "fastStorage", "Rnd"]; WK = [120, 300, 600]
order = [(t, w) for t in TR for w in WK]
y = np.arange(len(order))[::-1] + np.repeat([0, 0.6, 1.2, 1.8][::-1], 3)[::-1][::-1] * 0
ypos = []; yy = 0
for i, (t, w) in enumerate(order):
    ypos.append(yy); yy -= 1
    if w == 600: yy -= 0.6
ypos = np.array(ypos)
fig, ax = plt.subplots(1, 3, figsize=(6.8, 4.1), sharey=True)
panels = [("dec-simple", "(a) Decision-sel.\nminus simple rule"), ("acc-simple", "(b) Accuracy-sel.\nminus simple rule"), ("dec-acc", "(c) Decision-sel.\nminus accuracy-sel.")]
for a, (nm, ttl) in zip(ax, panels):
    for (t, w), yv in zip(order, ypos):
        r = S[(S.trace == t) & (S.W == w)].iloc[0]; v, lo, hi = r[nm], r[nm + "_lo"], r[nm + "_hi"]
        sig = lo > 0 or hi < 0
        a.plot([lo, hi], [yv, yv], color="0.15", lw=1.0)
        a.plot(v, yv, marker="o" if sig else "s", ms=4.5, color="0.1", mfc="0.1" if sig else "white", mew=0.9)
    a.axvline(0, color="0.5", lw=0.7, ls="--"); a.set_title(ttl, fontsize=8)
    a.set_yticks(ypos); a.set_yticklabels([f"{LAB[t]}, {w}" for t, w in order], fontsize=7.4)
    a.spines["left"].set_visible(a is ax[0]); 
    if a is not ax[0]: a.tick_params(left=False)
    for gy in [ypos[2] - 0.8, ypos[5] - 0.8, ypos[8] - 0.8]: a.axhline(gy, color="0.85", lw=0.5)
ax[0].set_ylabel("Trace, wake-up time $W$ (s)")
fig.supxlabel("Return difference per 1000 tasks (95% interval)", fontsize=8.5, y=0.115)
h = [plt.Line2D([], [], marker="o", color="0.1", ls="", ms=4.5, label="95% interval excludes zero"), plt.Line2D([], [], marker="s", color="0.1", mfc="white", ls="", ms=4.5, label="95% interval includes zero")]
fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02)); fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(P + "figures/fig_forest.pdf"); fig.savefig(P + "figures/fig_forest.png", dpi=300); plt.close(fig)
# design robustness: pooled random-effects estimate with 95% interval, three comparisons x three designs
V = [("Primary ($V{=}20$)", "fin", 20, "o", "0.1"), ("More episodes ($V{=}60$)", "fin", 60, "s", "0.45"), ("Halves swapped", "finsw", 20, "^", "0.7")]
cmp_ = [("dec-simple", "Decision-selected\nminus simple rule"), ("acc-simple", "Accuracy-selected\nminus simple rule"), ("dec-acc", "Decision-selected\nminus accuracy-selected")]
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for j, (nm, lab) in enumerate(cmp_):
    for i, (vl, tag, nv, mk, col) in enumerate(V):
        T = prim(tag, nv); mr, lo, hi = pool(T, nm); yv = -j * 1.6 - i * 0.42
        ax.plot([lo, hi], [yv, yv], color=col, lw=1.4); ax.plot(mr, yv, marker=mk, ms=5.5, color=col, mec="0.1", mew=0.7, ls="")
        ax.plot(T[nm].median(), yv, marker="|", ms=7, color="0.1", mew=1.0, ls="")
ax.axvline(0, color="0.5", lw=0.7, ls="--"); ax.set_yticks([-j * 1.6 - 0.42 for j in range(3)]); ax.set_yticklabels([l for _, l in cmp_], fontsize=7.8)
ax.set_xlabel("Pooled return difference per 1000 tasks (twelve cells)")
h = [plt.Line2D([], [], marker=mk, color=col, mec="0.1", ls="-", ms=5.5, lw=1.4, label=vl) for vl, _, _, mk, col in V] + [plt.Line2D([], [], marker="|", color="0.1", ls="", ms=7, mew=1.0, label="Median of the twelve cells")]
ax.legend(handles=h, frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); fig.tight_layout(); fig.savefig(P + "figures/fig_design.pdf"); fig.savefig(P + "figures/fig_design.png", dpi=300); plt.close(fig)
print("ok")
