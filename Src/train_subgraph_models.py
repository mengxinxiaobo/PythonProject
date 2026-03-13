# Src/train_subgraph_models.py
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from Src.models.gcn_bilstm import normalize_adjacency, GCNBiLSTMPredictor


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


class SubgraphWindowDataset(Dataset):
    def __init__(self, values_path: str, starts_path: str, T: int, horizon: int, nodes: list[int]):
        self.x = np.load(values_path, mmap_mode="r")
        self.starts = np.load(starts_path)
        self.T = T
        self.h = horizon
        self.nodes = np.array(nodes, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        st = int(self.starts[idx])
        x_win = self.x[st: st + self.T, self.nodes].copy()
        y = self.x[st + self.T + self.h - 1, self.nodes].copy()
        return torch.from_numpy(x_win).float(), torch.from_numpy(y).float()


@torch.no_grad()
def residuals_nodewise(model, loader, device, max_batches=None):
    model.eval()
    outs = []
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            yhat = model(x)
        res = (yhat - y) ** 2
        outs.append(res.detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 0), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_dir", default="Src/runs/prep/swat_T100_S10_H1")
    ap.add_argument("--graph_dir", default="Src/runs/graph")
    ap.add_argument("--subgraph_file", default="Src/runs/subgraphs/subgraphs_stage.json")
    ap.add_argument("--out_dir", default="Src/runs/subgraph_models_stage")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    cache_dir = (project_root / args.prep_dir).resolve()
    graph_dir = (project_root / args.graph_dir).resolve()
    subgraph_path = (project_root / args.subgraph_file).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(cache_dir / "scaler.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    T = int(meta["T"])
    horizon = int(meta["horizon"])

    with open(subgraph_path, "r", encoding="utf-8") as f:
        subgraphs = json.load(f)

    print("using subgraphs:", str(subgraph_path))

    A = np.load(str(graph_dir / "A_topk.npy")).astype(np.float32)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.clip(A, 0.0, None)
    np.fill_diagonal(A, 1.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "T:", T, "horizon:", horizon)

    normal_values = cache_dir / "normal_values.npy"
    merged_values = cache_dir / "merged_values.npy"
    train_starts = cache_dir / "train_starts.npy"
    val_starts = cache_dir / "val_starts.npy"
    test_starts = cache_dir / "test_starts.npy"
    test_win_labels = np.load(str(cache_dir / "test_win_labels.npy"))

    num_val = len(np.load(str(val_starts)))
    num_test = len(np.load(str(test_starts)))
    N_total = A.shape[0]

    subgraph_names = list(subgraphs.keys())
    S = len(subgraph_names)

    # 保存每个子图的窗口级分数（后续做 stage fusion / scheduler）
    sub_val_mean = np.zeros((num_val, S), dtype=np.float32)
    sub_test_mean = np.zeros((num_test, S), dtype=np.float32)
    sub_val_max = np.zeros((num_val, S), dtype=np.float32)
    sub_test_max = np.zeros((num_test, S), dtype=np.float32)

    # 兼容旧逻辑：保留节点级聚合残差
    val_numer = np.zeros((num_val, N_total), dtype=np.float64)
    test_numer = np.zeros((num_test, N_total), dtype=np.float64)
    denom = np.zeros((N_total,), dtype=np.float64)

    conf_weights = {}
    topo_weights = {}

    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    for s_idx, gname in enumerate(subgraph_names):
        ginfo = subgraphs[gname]
        nodes = ginfo["nodes"]
        V = len(nodes)
        if V < 2:
            continue

        print(f"\n=== Train {gname} V={V} ===")
        t0 = time.time()

        # 自适应超参
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

        A_sub = A[np.ix_(nodes, nodes)]
        np.fill_diagonal(A_sub, 1.0)
        A_norm = normalize_adjacency(torch.from_numpy(A_sub).to(device))

        deg = A_sub.sum(axis=1)
        deg = deg / (deg.mean() + 1e-12)
        topo_weights[gname] = deg.astype(np.float32)

        train_ds = SubgraphWindowDataset(str(normal_values), str(train_starts), T, horizon, nodes)
        val_ds = SubgraphWindowDataset(str(normal_values), str(val_starts), T, horizon, nodes)
        test_ds = SubgraphWindowDataset(str(merged_values), str(test_starts), T, horizon, nodes)

        train_loader = make_loader(train_ds, batch_train, shuffle=True)
        val_loader = make_loader(val_ds, batch_eval, shuffle=False)
        test_loader = make_loader(test_ds, batch_eval, shuffle=False)

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
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
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

        model.load_state_dict(torch.load(str(best_path), map_location=device))

        val_res_full = residuals_nodewise(model, val_loader, device, max_batches=None)
        test_res = residuals_nodewise(model, test_loader, device, max_batches=None)

        # 子图窗口级分数：保存 mean / max 两种
        sub_val_mean[:, s_idx] = val_res_full.mean(axis=1).astype(np.float32)
        sub_test_mean[:, s_idx] = test_res.mean(axis=1).astype(np.float32)
        sub_val_max[:, s_idx] = val_res_full.max(axis=1).astype(np.float32)
        sub_test_max[:, s_idx] = test_res.max(axis=1).astype(np.float32)

        # 置信度
        val_win = val_res_full.mean(axis=1)
        conf = 1.0 / (val_win.var() + 1e-6)
        conf = float(np.clip(conf, 0.05, 20.0))
        conf_weights[gname] = conf
        print(f"{gname} conf_weight={conf:.4f}")

        # 节点级聚合（兼容旧评估）
        topo = topo_weights[gname].astype(np.float64)
        w = topo * conf

        val_numer[:, nodes] += val_res_full * w[None, :]
        test_numer[:, nodes] += test_res * w[None, :]
        denom[nodes] += w

        print(f"{gname} done in {time.time() - t0:.1f}s")

    # 节点级聚合结果（保留）
    denom = np.maximum(denom, 1e-12)
    global_node_res_val = (val_numer / denom[None, :]).astype(np.float32)
    global_node_res_test = (test_numer / denom[None, :]).astype(np.float32)

    np.save(str(out_dir / "global_node_res_val.npy"), global_node_res_val)
    np.save(str(out_dir / "global_node_res_test.npy"), global_node_res_test)

    # 子图级分数（新增）
    np.save(str(out_dir / "subgraph_score_val_mean.npy"), sub_val_mean)
    np.save(str(out_dir / "subgraph_score_test_mean.npy"), sub_test_mean)
    np.save(str(out_dir / "subgraph_score_val_max.npy"), sub_val_max)
    np.save(str(out_dir / "subgraph_score_test_max.npy"), sub_test_max)

    with open(out_dir / "subgraph_names.json", "w", encoding="utf-8") as f:
        json.dump(subgraph_names, f, ensure_ascii=False, indent=2)

    with open(out_dir / "weights.json", "w", encoding="utf-8") as f:
        json.dump({"conf_weights": conf_weights}, f, ensure_ascii=False, indent=2)

    # 给一个默认 stage-fusion quick check：zscore + max + q=0.999
    mu = sub_val_mean.mean(axis=0)
    sigma = sub_val_mean.std(axis=0) + 1e-6
    z_val = (sub_val_mean - mu) / sigma
    z_test = (sub_test_mean - mu) / sigma
    score_val = z_val.max(axis=1)
    score_test = z_test.max(axis=1)
    thr = float(np.quantile(score_val, 0.999))
    pred = (score_test > thr).astype(np.int8)

    p, r, f1 = f1_pr_recall(test_win_labels[:len(pred)], pred)
    print("\n=== Quick Stage-Fusion Baseline ===")
    print(f"P={p:.4f} R={r:.4f} F1={f1:.4f} | threshold={thr:.6f}")

    np.save(str(out_dir / "score_val_default.npy"), score_val.astype(np.float32))
    np.save(str(out_dir / "score_test_default.npy"), score_test.astype(np.float32))
    np.save(str(out_dir / "pred_test_default.npy"), pred.astype(np.int8))

    print("saved:", str(out_dir))


if __name__ == "__main__":
    main()