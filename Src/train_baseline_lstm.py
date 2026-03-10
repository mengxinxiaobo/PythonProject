import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from Src.models.lstm_predictor import LSTMPredictor


class SlidingWindowDataset(Dataset):
    def __init__(self, values_path: str, starts_path: str, T: int, horizon: int):
        self.x = np.load(values_path, mmap_mode="r")   # [L, N]
        self.starts = np.load(starts_path)             # [B]
        self.T = T
        self.h = horizon

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        st = int(self.starts[idx])
        x_win = self.x[st: st + self.T]                # [T, N]
        y = self.x[st + self.T + self.h - 1]           # [N]
        return torch.from_numpy(x_win).float(), torch.from_numpy(y).float()


def f1_pr_recall(y_true, y_pred):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1


@torch.no_grad()
def compute_residuals(model, loader, device):
    model.eval()
    res = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        yhat = model(x)
        mse = torch.mean((yhat - y) ** 2, dim=1)  # [B]
        res.append(mse.detach().cpu().numpy())
    return np.concatenate(res, axis=0)


def main():
    # ---- 绝对路径定位：不受PyCharm工作目录影响 ----
    project_root = Path(__file__).resolve().parents[1]  # Src/.. -> 项目根
    cache_dir = project_root / "Src" / "runs" / "prep" / "swat_T100_S10_H1"
    cache_dir = cache_dir.resolve()

    scaler_path = cache_dir / "scaler.json"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing {scaler_path}. "
            f"Please run: python -m Src.data.make_cache --T 100 --stride 10 --horizon 1"
        )

    with open(scaler_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    T = int(meta["T"])
    horizon = int(meta["horizon"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("cache_dir:", str(cache_dir))

    train_ds = SlidingWindowDataset(
        values_path=str(cache_dir / "normal_values.npy"),
        starts_path=str(cache_dir / "train_starts.npy"),
        T=T, horizon=horizon
    )
    val_ds = SlidingWindowDataset(
        values_path=str(cache_dir / "normal_values.npy"),
        starts_path=str(cache_dir / "val_starts.npy"),
        T=T, horizon=horizon
    )
    test_ds = SlidingWindowDataset(
        values_path=str(cache_dir / "merged_values.npy"),
        starts_path=str(cache_dir / "test_starts.npy"),
        T=T, horizon=horizon
    )
    test_win_labels = np.load(str(cache_dir / "test_win_labels.npy"))

    sample_x, _ = train_ds[0]
    N = sample_x.shape[-1]
    print("N:", N, "T:", T)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    model = LSTMPredictor(num_features=N, hidden=128, num_layers=2, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    out_dir = project_root / "Src" / "runs" / "baseline_lstm"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"

    best_val = float("inf")
    for epoch in range(1, 11):
        model.train()
        total = 0.0
        cnt = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.size(0)
            cnt += x.size(0)

        train_loss = total / max(cnt, 1)
        val_res = compute_residuals(model, val_loader, device)
        val_loss = float(val_res.mean())
        print(f"epoch {epoch:02d} train_loss={train_loss:.6f} val_res_mean={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), str(best_path))

    # 阈值：验证集残差 99.5% 分位
    model.load_state_dict(torch.load(str(best_path), map_location=device))
    val_res = compute_residuals(model, val_loader, device)
    thresh = float(np.quantile(val_res, 0.995))
    print("threshold:", thresh)

    test_res = compute_residuals(model, test_loader, device)
    y_pred = (test_res > thresh).astype(np.int8)

    p, r, f1 = f1_pr_recall(test_win_labels[:len(y_pred)], y_pred)
    print(f"Test Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")

    np.save(str(out_dir / "test_residuals.npy"), test_res)
    np.save(str(out_dir / "test_pred.npy"), y_pred)
    print("saved:", str(out_dir))


if __name__ == "__main__":
    main()