"""
CNN-LSTM quantile forecaster for the scheduling environment.

Train it (synthetic loads, or your own trace):
    python forecaster_cnn_lstm.py --out forecaster.pt
    python forecaster_cnn_lstm.py --trace google_cpu.npy --out forecaster.pt

Use it in the environment:
    from forecaster_cnn_lstm import QuantileForecaster
    fc = QuantileForecaster.load("forecaster.pt")
    env = CloudSchedEnv(forecast_mode="quantile", forecaster=fc)
or from the PPO script:  python train_ppo.py --forecaster forecaster.pt ...

The model sees the last `window` slot loads and predicts the next `k` slots at
quantiles 0.1, 0.5 and 0.9 (pinball loss). Train it only on training data.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

QUANTILES = (0.1, 0.5, 0.9)


class CNNLSTM(nn.Module):
    def __init__(self, k: int, channels: int = 32, hidden: int = 64):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(nn.Conv1d(1, channels, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, k * len(QUANTILES))

    def forward(self, x):                      # x: (batch, window)
        z = self.conv(x.unsqueeze(1))          # (batch, channels, window)
        out, _ = self.lstm(z.transpose(1, 2))  # (batch, window, hidden)
        y = self.head(out[:, -1])              # (batch, k*3)
        return y.view(-1, self.k, len(QUANTILES))


def pinball_loss(pred, target):                # pred (B,k,Q), target (B,k)
    q = torch.tensor(QUANTILES, device=pred.device).view(1, 1, -1)
    err = target.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1) * err).mean()


def make_windows(series, window, k):
    X, Y = [], []
    for i in range(window, len(series) - k + 1):
        X.append(series[i - window:i])
        Y.append(series[i:i + k])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


class QuantileForecaster:
    """Callable with the signature the environment expects: (history, k) -> (mean, lo, hi)."""

    def __init__(self, model: CNNLSTM, window: int):
        torch.set_num_threads(1)
        self.model, self.window, self.k = model.eval(), window, model.k

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu")
        m = CNNLSTM(ck["k"])
        m.load_state_dict(ck["state"])
        return cls(m, ck["window"])

    def __call__(self, history, k):
        h = np.asarray(history, dtype=np.float32)
        if len(h) == 0:
            h = np.array([0.5], dtype=np.float32)
        if len(h) < self.window:
            h = np.concatenate([np.full(self.window - len(h), h[0], dtype=np.float32), h])
        x = torch.from_numpy(h[-self.window:]).unsqueeze(0)
        with torch.no_grad():
            q = self.model(x)[0].numpy()               # (k_model, 3)
        q = np.sort(q, axis=1)                          # enforce lo <= mid <= hi
        q = np.clip(q, 0.0, 1.0)
        idx = np.minimum(np.arange(k), self.k - 1)      # repeat last step if env asks for more
        return q[idx, 1], q[idx, 0], q[idx, 2]


def main():
    from cloud_env import synthetic_loads

    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=str, default=None, help="1-D .npy/.csv of utilisation (chronological)")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--k", type=int, default=6, help="forecast horizon in slots; keep equal to the environment horizon_k")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="forecaster.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    if a.trace:
        arr = np.load(a.trace) if a.trace.endswith(".npy") else np.loadtxt(a.trace, delimiter=",")
        arr = np.asarray(arr, dtype=float).ravel()
        arr = arr[: int(0.8 * len(arr))]             # training part only; the last 20% is the test set
        from cloud_env import scale_trace
        arr = scale_trace(arr, arr)                  # same scaling as train_ppo.py (train-set min/max)
        cut = int(0.85 * len(arr))
        train_series, val_series = arr[:cut], arr[cut:]
    else:
        # synthetic: many independent series, seeds disjoint from the environment's evaluation seeds
        train_series = np.concatenate([synthetic_loads(rng, 400, drift=bool(i % 2)) for i in range(40)])
        val_series = np.concatenate([synthetic_loads(rng, 400, drift=bool(i % 2)) for i in range(8)])

    Xtr, Ytr = make_windows(train_series, a.window, a.k)
    Xva, Yva = make_windows(val_series, a.window, a.k)
    Xtr, Ytr, Xva, Yva = map(torch.from_numpy, (Xtr, Ytr, Xva, Yva))
    model = CNNLSTM(a.k)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best, best_state = 1e9, None
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            opt.zero_grad()
            loss = pinball_loss(model(Xtr[b]), Ytr[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = pinball_loss(model(Xva), Yva).item()
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep:3d} val pinball {val:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sort(model(Xva), dim=2).values
        mae = (p[..., 1] - Yva).abs().mean().item()
        cover = ((Yva >= p[..., 0]) & (Yva <= p[..., 2])).float().mean().item()
    print(f"validation: median-forecast MAE {mae:.4f}, 80% interval coverage {cover:.2%} (target 80%)")
    torch.save({"state": model.state_dict(), "k": a.k, "window": a.window}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
