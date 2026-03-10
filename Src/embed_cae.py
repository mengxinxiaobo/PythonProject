# Src/embed_cae.py
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from Src.models.cae_mmd import CAE1D


class WindowDataset(Dataset):
    def __init__(self, values_path: str, starts_path: str, T: int):
        self.x = np.load(values_path, mmap_mode="r")  # [L, N]
        self.starts = np.load(starts_path)            # [B]
        self.T = T

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        st = int(self.starts[idx])
        win = self.x[st: st + self.T].copy()          # [T, N]
        return torch.from_numpy(win).float()


@torch.no_grad()
def embed_split(model, loader, device, z_dim: int):
    model.eval()
    outs = []
    for win in loader:
        win = win.to(device, non_blocking=True)               # [B, T, N]
        B, T, N = win.shape
        x = win.permute(0, 2, 1).contiguous().view(B * N, 1, T)  # [B*N,1,T]
        _, z = model(x)                                       # [B*N, z_dim]
        z = z.view(B, N, z_dim).detach().cpu().numpy()
        outs.append(z)
    return np.concatenate(outs, axis=0)


def main():
    project_root = Path(__file__).resolve().parents[1]
    cache_dir = (project_root / "Src" / "runs" / "prep" / "swat_T100_S10_H1").resolve()

    with open(cache_dir / "scaler.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    T = int(meta["T"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "T:", T)

    z_dim = 32
    model = CAE1D(T=T, z_dim=z_dim).to(device)
    ckpt = project_root / "Src" / "runs" / "cae_mmd" / "best.pt"
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    print("loaded:", str(ckpt))

    train_ds = WindowDataset(str(cache_dir / "normal_values.npy"), str(cache_dir / "train_starts.npy"), T)
    val_ds   = WindowDataset(str(cache_dir / "normal_values.npy"), str(cache_dir / "val_starts.npy"), T)
    test_ds  = WindowDataset(str(cache_dir / "merged_values.npy"), str(cache_dir / "test_starts.npy"), T)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    out_dir = (project_root / "Src" / "runs" / "embeddings").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_train = embed_split(model, train_loader, device, z_dim)
    emb_val   = embed_split(model, val_loader, device, z_dim)
    emb_test  = embed_split(model, test_loader, device, z_dim)

    np.save(str(out_dir / "emb_train.npy"), emb_train.astype(np.float32))
    np.save(str(out_dir / "emb_val.npy"), emb_val.astype(np.float32))
    np.save(str(out_dir / "emb_test.npy"), emb_test.astype(np.float32))

    print("saved:", str(out_dir))
    print("emb_train:", emb_train.shape, "emb_val:", emb_val.shape, "emb_test:", emb_test.shape)


if __name__ == "__main__":
    main()