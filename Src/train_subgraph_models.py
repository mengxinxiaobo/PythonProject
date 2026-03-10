# Src/train_subgraph_models.py
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from Src.models.gcn_bilstm import normalize_adjacency, GCNBiLSTMPredictor


# ======= small utils =======
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


def make_loader(ds, batch_size, shuffle):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )


# ======= Dataset =======
class SubgraphWindowDataset(Dataset):
    """
    values: [L,N] (normal_values.npy or merged_values.npy)
    starts: [num_windows]
    nodes: list[int] 子图节点索引
    """
    def __init__(self, values_path: str, starts_path: str, T: int, horizon: int, nodes: list[int]):
        self.x = np.load(values_path, mmap_mode="r")   # [L,N]
        self.starts = np.load(starts_path)             # [B]
        self.T = T
        self.h = horizon
        self.nodes = np.array(nodes, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        st = int(self.starts[idx])
        x_win = self.x[st: st + self.T, self.nodes].copy()      # [T,V]
        y = self.x[st + self.T + self.h - 1, self.nodes].copy() # [V]
        return torch.from_numpy(x_win).float(), torch.from_numpy(y).float()


# ======= Residual computation =======
@torch.no_grad()
def residuals_nodewise(model, loader, device, max_batches=None):
    """
    return residuals: [num_windows, V]  (per-node squared error)
    If max_batches is not None: only compute that many batches (speed up val).
    """
    model.eval()
    outs = []
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            yhat = model(x)                 # [B,V]
        res = (yhat - y) ** 2               # [B,V]
        outs.append(res.detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 0), dtype=np.float32)


def main():
    project_root = Path(__file__).resolve().parents[1]

    cache_dir = (project_root / "Src" / "runs" / "prep" / "swat_T100_S10_H1").resolve()
    graph_dir = (project_root / "Src" / "runs" / "graph").resolve()
    sub_dir = (project_root / "Src" / "runs" / "subgraphs").resolve()
    out_dir = (project_root / "Src" / "runs" / "subgraph_models").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cache_dir / "scaler.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    T = int(meta["T"])
    horizon = int(meta["horizon"])

    # load subgraphs
    with open(sub_dir / "subgraphs.json", "r", encoding="utf-8") as f:
        subgraphs = json.load(f)

    # adjacency for subgraphs
    A = np.load(str(graph_dir / "A_topk.npy")).astype(np.float32)  # [51,51]
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.clip(A, 0.0, None)
    np.fill_diagonal(A, 1.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "T:", T, "horizon:", horizon)

    # cache paths
    normal_values = cache_dir / "normal_values.npy"
    merged_values = cache_dir / "merged_values.npy"
    train_starts = cache_dir / "train_starts.npy"
    val_starts = cache_dir / "val_starts.npy"
    test_starts = cache_dir / "test_starts.npy"
    test_win_labels = np.load(str(cache_dir / "test_win_labels.npy"))

    # For thresholding: use VAL aggregated score (normal only)
    num_val = len(np.load(str(val_starts)))
    num_test = len(np.load(str(test_starts)))
    N_total = A.shape[0]

    # numerator/denominator for weighted aggregation
    val_numer = np.zeros((num_val, N_total), dtype=np.float64)
    test_numer = np.zeros((num_test, N_total), dtype=np.float64)
    denom = np.zeros((N_total,), dtype=np.float64)

    conf_weights = {}   # per subgraph scalar
    topo_weights = {}   # per subgraph vector

    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    # Train each subgraph model
    for gname, ginfo in subgraphs.items():
        nodes = ginfo["nodes"]
        V = len(nodes)
        if V < 2:
            continue

        print(f"\n=== Train {gname} V={V} ===")
        t0 = time.time()

        # ----- adaptive hyperparams by V -----
        if V <= 8:
            gcn_h, lstm_h = 32, 32
            lr = 3e-4
            max_epochs = 12
            patience = 2
            batch_train = 512
            batch_eval = 512
            val_max_batches = None
        elif V <= 20:
            gcn_h, lstm_h = 48, 48
            lr = 5e-4
            max_epochs = 10
            patience = 2
            batch_train = 512
            batch_eval = 512
            val_max_batches = 20
        else:
            gcn_h, lstm_h = 64, 64
            lr = 5e-4
            max_epochs = 10
            patience = 2
            batch_train = 256
            batch_eval = 512
            val_max_batches = 20

        # subgraph adjacency
        A_sub = A[np.ix_(nodes, nodes)]
        np.fill_diagonal(A_sub, 1.0)
        A_norm = normalize_adjacency(torch.from_numpy(A_sub).to(device))

        # topo weight: weighted degree -> normalize mean=1
        deg = A_sub.sum(axis=1)
        deg = deg / (deg.mean() + 1e-12)
        topo_weights[gname] = deg.astype(np.float32)

        # datasets/loaders
        train_ds = SubgraphWindowDataset(str(normal_values), str(train_starts), T, horizon, nodes)
        val_ds   = SubgraphWindowDataset(str(normal_values), str(val_starts), T, horizon, nodes)
        test_ds  = SubgraphWindowDataset(str(merged_values), str(test_starts), T, horizon, nodes)

        train_loader = make_loader(train_ds, batch_train, shuffle=True)
        val_loader   = make_loader(val_ds, batch_eval, shuffle=False)
        test_loader  = make_loader(test_ds, batch_eval, shuffle=False)

        model = GCNBiLSTMPredictor(
            A_norm=A_norm,
            gcn_hidden=gcn_h,
            lstm_hidden=lstm_h,
            num_layers=1,
            dropout=0.2,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()

        best_val = float("inf")
        bad = 0
        best_path = out_dir / f"{gname}.pt"

        for epoch in range(1, max_epochs + 1):
            model.train()
            total = 0.0
            cnt = 0

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)    # [B,T,V]
                y = y.to(device, non_blocking=True)    # [B,V]
                opt.zero_grad(set_to_none=True)

                if device == "cuda":
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        yhat = model(x)
                        loss = loss_fn(yhat, y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    yhat = model(x)
                    loss = loss_fn(yhat, y)
                    loss.backward()
                    opt.step()

                total += float(loss.item()) * x.size(0)
                cnt += x.size(0)

            train_mse = total / max(cnt, 1)

            # val (sampled for large graphs to reduce epoch time)
            val_res = residuals_nodewise(model, val_loader, device, max_batches=val_max_batches)
            val_mse = float(val_res.mean()) if val_res.size else float("inf")
            print(f"epoch {epoch:02d} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

            if val_mse < best_val - 1e-6:
                best_val = val_mse
                bad = 0
                torch.save(model.state_dict(), str(best_path))
            else:
                bad += 1
                if bad >= patience:
                    print(f"early stop at epoch {epoch} (best_val={best_val:.6f})")
                    break

        # load best
        model.load_state_dict(torch.load(str(best_path), map_location=device))

        # full val residuals for conf + aggregation (must be full)
        val_res_full = residuals_nodewise(model, val_loader, device, max_batches=None)  # [num_val,V]
        if val_res_full.shape[0] != num_val:
            print(f"WARNING: val_res_full rows {val_res_full.shape[0]} != num_val {num_val}")

        # conf weight from val variance of window residual
        val_win = val_res_full.mean(axis=1)  # [num_val]
        conf = 1.0 / (val_win.var() + 1e-6)
        conf = float(np.clip(conf, 0.05, 20.0))
        conf_weights[gname] = conf  # ★关键：写入字典
        print(f"{gname} conf_weight={conf:.4f} (from val variance)")

        # test residuals
        test_res = residuals_nodewise(model, test_loader, device, max_batches=None)  # [num_test,V]

        # accumulate aggregation numer/denom
        topo = topo_weights[gname].astype(np.float64)  # [V]
        w = topo * conf_weights[gname]                 # [V]

        val_numer[:, nodes] += val_res_full * w[None, :]
        test_numer[:, nodes] += test_res * w[None, :]
        denom[nodes] += w

        print(f"{gname} done in {time.time() - t0:.1f}s")

    # ---- Aggregate global residuals ----
    denom = np.maximum(denom, 1e-12)
    global_node_res_val = (val_numer / denom[None, :]).astype(np.float32)    # [num_val, N]
    global_node_res_test = (test_numer / denom[None, :]).astype(np.float32)  # [num_test, N]

    # ===== Node-wise z-score normalization using VAL(normal) =====
    mu = global_node_res_val.mean(axis=0)
    sigma = global_node_res_val.std(axis=0) + 1e-6
    z_val = (global_node_res_val - mu) / sigma
    z_test = (global_node_res_test - mu) / sigma

    # score: 95th percentile across nodes (more robust than max)
    score_val = np.percentile(z_val, 99, axis=1)
    score_test = np.percentile(z_test, 99, axis=1)

    # threshold from val(normal) quantile
    thresh = float(np.quantile(score_val, 0.997))
    y_pred = (score_test > thresh).astype(np.int8)

    p, r, f1 = f1_pr_recall(test_win_labels[:len(y_pred)], y_pred)
    print("\n=== Aggregated Results (z-score + p95, threshold from VAL normal quantile) ===")
    print("threshold:", thresh)
    print(f"Test Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")

    # save artifacts
    np.save(str(out_dir / "global_node_res_val.npy"), global_node_res_val)
    np.save(str(out_dir / "global_node_res_test.npy"), global_node_res_test)
    np.save(str(out_dir / "score_val.npy"), score_val.astype(np.float32))
    np.save(str(out_dir / "score_test.npy"), score_test.astype(np.float32))
    np.save(str(out_dir / "pred_test.npy"), y_pred)

    with open(out_dir / "weights.json", "w", encoding="utf-8") as f:
        json.dump({"conf_weights": conf_weights}, f, ensure_ascii=False, indent=2)

    print("saved:", str(out_dir))


if __name__ == "__main__":
    main()