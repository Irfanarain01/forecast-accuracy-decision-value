"""Issue 4a: bootstrap of the 20 stored validation episodes -> selection stability and selection uncertainty."""
import sys, numpy as np, pandas as pd
import sys
from p1_analyze2 import load, MODELS
TAGB = sys.argv[1]; NVB = int(sys.argv[2])
from evalad import DIRS
rng = np.random.default_rng(7); B = 1000
cells = load(TAGB, 18, 2, 1.0); rows = []
for (tr, W), R in sorted(cells.items()):
    R = R.copy(); R["Jw"] = R.J1000
    V = R[(R.part == "val") & (R.ep < NVB)].pivot(index="ep", columns="config", values="Jw"); T = R[R.part == "test"].pivot(index="ep", columns="config", values="Jw")
    zoo = [c for c in V.columns if c.startswith("zoo|")]; sr = [c for c in V.columns if c.startswith(("static_", "rolling_"))]
    d0 = V[zoo].mean().idxmax(); s0 = V[sr].mean().idxmax(); n = len(V); diffs = []; chd = chs = 0
    for b in range(B):
        ix = rng.integers(0, n, n); m = V.values[ix].mean(0); mv = pd.Series(m, index=V.columns)
        d = mv[zoo].idxmax(); s = mv[sr].idxmax(); chd += d != d0; chs += s != s0
        diffs.append((T[d] - T[s]).mean())
    diffs = np.array(diffs); te = T[d0] - T[s0]; se = te.std(ddof=1) / np.sqrt(len(te))
    rows.append(dict(trace=tr, W=W, dec=d0, simple=s0, p_dec_changes=chd / B, p_simple_changes=chs / B, base=te.mean(), test_ci_halfwidth=1.96 * se,
        boot_lo=np.percentile(diffs, 2.5), boot_hi=np.percentile(diffs, 97.5), boot_mean=diffs.mean(), p_dec_beats_simple=(diffs > 0).mean()))
D = pd.DataFrame(rows); D.to_csv(f"p1_boot_{TAGB}{NVB}.csv", index=False); pd.set_option("display.width", 250); print(D.round(2).to_string())
print("median p_dec_changes", D.p_dec_changes.median(), "median p_simple_changes", D.p_simple_changes.median())
