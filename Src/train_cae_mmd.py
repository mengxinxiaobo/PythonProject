import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from Src.models.cae_mmd import CAE1D, rbf_mmd


class WindowDataset(Dataset):
    def __init__(self, values_path: str, starts_path: str, T: int):
        self.x = np.load(values_path, mmap_mode="r")  # [L, N]
        self.starts = np.load(starts_path)            # [B]
        self.T = T

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        st = int(self.starts[idx])
        win = self.x[st: st + self.T].copy()          # [T, N]  copy消除只读warning
        return torch.from_numpy(win).float()


def main():
    project_root = Path(__file__).resolve().parents[1]
    cache_dir = (project_root / "Src" / "runs" / "prep" / "swat_T100_S10_H1").resolve()

    with open(cache_dir / "scaler.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    T = int(meta["T"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "T:", T)

    train_ds = WindowDataset(str(cache_dir / "normal_values.npy"), str(cache_dir / "train_starts.npy"), T)
    val_ds = WindowDataset(str(cache_dir / "normal_values.npy"), str(cache_dir / "val_starts.npy"), T)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = CAE1D(T=T, z_dim=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    lam_mmd = 0.1
    sigma = 1.0

    out_dir = (project_root / "Src" / "runs" / "cae_mmd").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"

    best_val = float("inf")
    mse = torch.nn.MSELoss()

    for epoch in range(1, 21):
        model.train()
        total = 0.0
        cnt = 0

        for win in train_loader:
            # win: [B, T, N]
            win = win.to(device, non_blocking=True)
            B, T_, N = win.shape

            # 展平为 (B*N) 个单变量序列
            x = win.permute(0, 2, 1).contiguous().view(B * N, 1, T_)  # [B*N,1,T]

            opt.zero_grad(set_to_none=True)
            xhat, z = model(x)

            rec = mse(xhat, x)

            # MMD：z 与标准高斯
            z_prior = torch.randn_like(z)
            mmd = rbf_mmd(z, z_prior, sigma=sigma)

            loss = rec + lam_mmd * mmd
            loss.backward()
            opt.step()

            total += float(loss.item()) * x.size(0)
            cnt += x.size(0)

        train_loss = total / max(cnt, 1)

        # val：只看重构误差（也可加mmd）
        model.eval()
        with torch.no_grad():
            vals = []
            for win in val_loader:
                win = win.to(device, non_blocking=True)
                B, T_, N = win.shape
                x = win.permute(0, 2, 1).contiguous().view(B * N, 1, T_)
                xhat, z = model(x)
                vals.append(torch.mean((xhat - x) ** 2).item())
            val_loss = float(np.mean(vals))

        print(f"epoch {epoch:02d} train_loss={train_loss:.6f} val_rec_mse={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), str(best_path))

    print("saved:", str(best_path))


if __name__ == "__main__":
    main()