"""Build CloudSim Plus inputs for paper 1: per-config host-count sequences and per-episode task streams.
Usage: python p1_inputs2.py <trace> <wake_s> <hosts> <vph> <rho> <outdir>        (run from any folder; imports the slot-level code from ../slot_level)"""
import sys, os, zlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "slot_level"))
from adaptive_tau import Streams, load_zoo, TAUS
from evalad import DIRS
TYPES = [(10_000.0, 60.0, 150.0), (20_000.0, 90.0, 250.0), (40_000.0, 150.0, 450.0)]
SLOT, EP_SLOTS, MEAN_LEN, NT = 10.0, 100, 6e4, 30
NV = int(os.environ.get("NV", "60")); SWAP = os.environ.get("SWAP", "0") == "1"
QS, LS = [0.5, 0.75, 0.9, 0.95, 0.99], [50, 100]

def fleet(n_hosts, vph):
    hm = np.array([TYPES[h % 3][0] for h in range(n_hosts)]); pi = np.array([TYPES[h % 3][1] for h in range(n_hosts)]); pm = np.array([TYPES[h % 3][2] for h in range(n_hosts)])
    rank = np.argsort(-(hm / pm), kind="stable"); return hm, pi, pm, rank

def build(trace, W, n_hosts, vph, rho, out):
    os.makedirs(out, exist_ok=True)
    zoo, rq, val, test = load_zoo(DIRS[trace] + "/zoo.pkl"); k = int(round(W / SLOT))
    # calibration/selection split of the validation segment: residual quantiles from one half, selection episodes from the other
    import fdv_benchmark as fb
    from forecaster_cnn_lstm import make_windows
    v = np.asarray(val, float); h = len(v) // 2
    cal, sel0 = (v[h:], 0) if SWAP else (v[:h], 0); sel_lo, sel_hi = (0, h) if SWAP else (h, len(v))
    Xc, Yc = make_windows(cal, fb.WINDOW, fb.KMAX)
    rq = {n: {t: np.quantile(Yc - m.predict(Xc), t, axis=0) for t in fb.TAUS} for n, m in zoo.items()}
    Xs, Ys = make_windows(v[sel_lo:sel_hi], fb.WINDOW, fb.KMAX)
    pd.DataFrame([dict(model=n, wake_s=W, sel_mae=float(np.abs(Ys[:, :k] - m.predict(Xs)[:, :k]).mean())) for n, m in zoo.items()]).to_csv(f"{out}/sel_mae.csv", index=False)
    hm, pi, pm, rank = fleet(n_hosts, vph); cum = np.cumsum(hm[rank]); total = hm.sum(); mean_vm = total / (n_hosts * vph)
    H = lambda x: np.clip(np.searchsorted(cum, np.asarray(x) * rho * total * 1.2) + 1, 1, n_hosts)   # host count; demand scales with the arrival intensity rho
    rate_max = rho * total / MEAN_LEN
    seqs, names = {}, []
    for part, series in (("val", val), ("test", test)):
        series = np.asarray(series, float); L = len(series); S = Streams(zoo, rq, series)
        cur = np.array([series[max(0, g - 3):g].mean() if g > 0 else 0.5 for g in range(L)])
        def add(name, x): seqs[(name, part)] = H(np.maximum(cur, x)).astype(np.int16)
        add("reactive", np.zeros(L)); seqs[("always_on", part)] = np.full(L, n_hosts, np.int16)
        add("oracle", np.array([series[g:g + k].max() if g < L else 0 for g in range(L)]))
        for j in range(1, 7):
            n = max(1, int(round(n_hosts * j / 6))); seqs[(f"static_{j}", part)] = np.full(L, n, np.int16)
        for q in QS:
            for Lw in LS:
                add(f"rolling_q{q}_L{Lw}", np.array([np.quantile(series[max(0, g - Lw):g], q) if g >= 10 else 0 for g in range(L)]))
        for mi, m in enumerate(S.names):
            F = S.P[mi, :, :k]
            add(f"zoo|{m}|point", F.max(1))
            for ti, t in enumerate(TAUS): add(f"zoo|{m}|{t}", (F + S.b0[mi, ti, :k]).max(1))
    cfgs = sorted({n for n, _ in seqs}); rows = [(n, p, " ".join(map(str, seqs[(n, p)]))) for n in cfgs for p in ("val", "test") if (n, p) in seqs]
    with open(f"{out}/kseq.csv", "w") as f:
        f.write("config,part,k\n"); [f.write(f"{n},{p},{s}\n") for n, p, s in rows]
    tasks = []; ep_rows = []
    for part, series, ne in (("val", val, NV), ("test", test, NT)):
        series = np.asarray(series, float); L = len(series)
        for e in range(ne):
            rng = np.random.default_rng(hash((trace, part, e)) % (2 ** 32) if False else (zlib.crc32(trace.encode()) % 1000) * 100000 + (0 if part == "val" else 50000) + e)
            lo, hi = (sel_lo, sel_hi) if part == "val" else (0, L)
            start = lo + int(rng.integers(0, max(1, hi - lo - EP_SLOTS - k - 1))); ep_rows.append((part, e, start))
            for i in range(EP_SLOTS):
                cnt = rng.poisson(rate_max * series[start + i] * SLOT)
                for ta in i * SLOT + np.sort(rng.uniform(0, SLOT, cnt)):
                    ln = rng.lognormal(np.log(MEAN_LEN) - 0.18, 0.6); dl = ta + rng.uniform(3.0, 8.0) * ln / mean_vm
                    tasks.append((part, e, i, round(float(ta), 4), round(float(ln), 1), round(float(dl), 4)))
    pd.DataFrame(tasks, columns=["part", "ep", "slot", "arrival", "length", "deadline"]).to_csv(f"{out}/tasks.csv", index=False)
    pd.DataFrame(ep_rows, columns=["part", "ep", "start"]).to_csv(f"{out}/episodes.csv", index=False)
    return len(cfgs), len(tasks)

if __name__ == "__main__":
    tr, W, nh, vph, rho, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
    print(build(tr, W, nh, vph, rho, out))
