"""Azure Functions 2019 invocation counts -> 1-D load series (total invocations per time step).

Input : invocations_per_function_md.anon.d01.csv ... d14.csv  (columns: HashOwner, HashApp, HashFunction, Trigger, 1..1440 = minutes of the day)
Output: .npy with one value per minute (14 days -> 20160 values); --bin N sums/averages N consecutive minutes.

  python prepare_azure_functions.py --dir azure_functions --bin 1 --out azf_load.npy
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder with invocations_per_function_md.anon.dXX.csv")
    ap.add_argument("--bin", type=int, default=1, help="average N consecutive minutes")
    ap.add_argument("--trigger", default=None, help="optional: keep only one trigger group, e.g. http")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "invocations_per_function_md.anon.d*.csv")))
    if not files:
        raise SystemExit("no invocations_per_function_md.anon.d*.csv files found in " + a.dir)
    days = []
    for f in files:
        df = pd.read_csv(f)
        if a.trigger:
            df = df[df["Trigger"] == a.trigger]
        m = df.iloc[:, 4:].to_numpy(dtype=np.float64)
        if m.shape[1] != 1440:
            raise SystemExit(f"{f}: expected 1440 minute columns, found {m.shape[1]}")
        days.append(m.sum(axis=0))                       # total invocations in each minute, all functions
        print(f"{os.path.basename(f)}: {len(df)} functions, total {days[-1].sum():.3g} invocations")
    s = np.concatenate(days)
    if a.bin > 1:
        n = len(s) // a.bin * a.bin
        s = s[:n].reshape(-1, a.bin).mean(axis=1)
    d1 = np.abs(np.diff(s))
    print(f"steps: {len(s)}  min {s.min():.4g}  mean {s.mean():.4g}  max {s.max():.4g}  std {s.std():.4g}")
    print(f"mean |step-to-step change|: {100 * d1.mean() / s.mean():.2f}% of the mean;  max/mean {s.max() / s.mean():.2f}")
    np.save(a.out, s.astype(np.float32))
    print("saved", a.out)


if __name__ == "__main__":
    main()
