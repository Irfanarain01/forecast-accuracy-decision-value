"""Convert a real cluster-utilisation file into the 1-D .npy that train_ppo.py / forecaster_cnn_lstm.py expect.

Output: one value per fixed time bin = mean CPU utilisation over all samples (all machines) that fall in the bin.
Rows may be in ANY order and machines may report at irregular times (as in Alibaba machine_usage, which is sorted by
machine, not by time): every row is put into bin floor(time / --time-bin), so no sorting is needed.

Examples
  # Alibaba cluster-trace-v2018 machine_usage.csv (no header; col 1 = time_stamp [s], col 2 = cpu_util_percent)
  python prepare_trace.py --in machine_usage.csv --no-header --time-col 1 --value-col 2 --time-bin 60 --out ali_cpu.npy

  # any CSV with a header
  python prepare_trace.py --in vm_cpu.csv --time-col timestamp --value-col avg_cpu --time-bin 300 --out azure_cpu.npy

  # Google 2011 task_usage (gz is read directly; no header; times in microseconds; tasks -> SUM per 5-min window)
  # (column numbers: check against the v2.1 schema; I believe 0 = start time, 5 = mean CPU rate)
  python prepare_trace.py --in part-00000-of-00500.csv.gz --no-header --time-col 0 --value-col 5 --agg sum --time-bin 300000000 --out google_cpu.npy

One output step = one simulator slot, so --time-bin sets how much real time a slot stands for.
"""
import argparse
import glob
import numpy as np
import pandas as pd


def col(x, no_header):
    return int(x) if no_header else (int(x) if str(x).isdigit() else x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sep", default=",")
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--time-col", required=True)
    ap.add_argument("--value-col", required=True)
    ap.add_argument("--time-bin", type=float, default=1.0, help="bin width in the units of the time column")
    ap.add_argument("--agg", choices=["mean", "sum"], default="mean",
                    help="mean of the samples in a bin (utilisation %%) or sum (per-task usage, e.g. Google task_usage)")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    a = ap.parse_args()

    tc, vc = col(a.time_col, a.no_header), col(a.value_col, a.no_header)
    S = np.zeros(0)
    C = np.zeros(0)
    nrows = 0
    paths = []
    for p in a.inp:                                   # wildcards are expanded here (Windows shells do not)
        paths += sorted(glob.glob(p)) or [p]
    print(f"{len(paths)} input file(s)")
    for path in paths:
        rd = pd.read_csv(path, sep=a.sep, header=None if a.no_header else 0, usecols=[tc, vc],
                         chunksize=a.chunk, skipinitialspace=True,
                         engine="python" if len(a.sep) > 1 else "c")
        for ch in rd:
            if isinstance(tc, int) and isinstance(vc, int):   # positions: usecols keeps FILE order
                order = sorted([tc, vc])
                ch = ch.iloc[:, [order.index(tc), order.index(vc)]]
            else:                                             # names
                ch = ch[[tc, vc]]
            ch.columns = ["t", "v"]
            ch = ch.apply(pd.to_numeric, errors="coerce").dropna()
            if ch.empty:
                continue
            idx = np.floor(ch["t"].to_numpy() / a.time_bin).astype(np.int64)
            if idx.min() < 0 or idx.max() > 50_000_000:
                raise SystemExit("time bin index out of range - choose a larger --time-bin for this time unit")
            if idx.max() >= len(S):
                n = idx.max() + 1
                S = np.pad(S, (0, n - len(S)))
                C = np.pad(C, (0, n - len(C)))
            S += np.bincount(idx, weights=ch["v"].to_numpy(), minlength=len(S))
            C += np.bincount(idx, minlength=len(C))
            nrows += len(ch)
    nz = np.nonzero(C)[0]
    first, last = nz[0], nz[-1]
    S, C = S[first:last + 1], C[first:last + 1]
    series = S / np.maximum(C, 1) if a.agg == "mean" else S.copy()
    empty = C == 0
    if empty.any():                                   # fill empty bins by linear interpolation
        x = np.arange(len(series))
        series[empty] = np.interp(x[empty], x[~empty], series[~empty])
    print(f"rows used: {nrows}   time steps: {len(series)}   empty bins interpolated: {int(empty.sum())}")
    print(f"samples per bin: min {int(C.min())}  median {int(np.median(C))}")
    print(f"value range: min {series.min():.3f}  mean {series.mean():.3f}  max {series.max():.3f}  std {series.std():.3f}")
    d1 = np.abs(np.diff(series))
    print(f"mean |step-to-step change|: {d1.mean():.3f}  ({100 * d1.mean() / max(1e-9, series.mean()):.2f}% of the mean)")
    if len(series) < 2500:
        print("WARNING: need >= ~2500 steps (training part + a >=450-step test part). Use a smaller --time-bin.")
    if series.std() < 0.02 * max(1e-9, abs(series.mean())):
        print("WARNING: series is almost flat - forecasting will show no benefit on it.")
    if empty.mean() > 0.05:
        print("WARNING: more than 5% of bins are empty - use a larger --time-bin.")
    np.save(a.out, series.astype(np.float32))
    print("saved", a.out)


if __name__ == "__main__":
    main()
