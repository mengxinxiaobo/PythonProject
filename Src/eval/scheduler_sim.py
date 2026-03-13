# Src/eval/scheduler_sim.py
import argparse
import json
from pathlib import Path
import numpy as np


def apply_hysteresis_per_col(binary_mat, H):
    """
    对每个子图的触发序列单独做迟滞
    binary_mat: [T, S]
    """
    if H <= 1:
        return binary_mat.astype(np.int32)

    out = np.zeros_like(binary_mat, dtype=np.int32)
    T, S = binary_mat.shape
    for s in range(S):
        i = 0
        while i < T:
            if binary_mat[i, s] == 1:
                st = i
                while i < T and binary_mat[i, s] == 1:
                    i += 1
                ed = i
                if (ed - st) >= H:
                    out[st:ed, s] = 1
            else:
                i += 1
    return out


def pick_sentinels(A, nodes, k):
    """
    用子图内加权度选 Top-k 哨兵节点
    """
    subA = A[np.ix_(nodes, nodes)]
    deg = subA.sum(axis=1)
    order = np.argsort(-deg)
    take = min(k, len(nodes))
    return [nodes[i] for i in order[:take]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_dir", default="Src/runs/prep/swat_T100_S10_H1")
    ap.add_argument("--graph_dir", default="Src/runs/graph")
    ap.add_argument("--subgraph_file", default="Src/runs/subgraphs/subgraphs_stage.json")
    ap.add_argument("--score_dir", default="Src/runs/subgraph_models_stage")
    ap.add_argument("--source", default="mean", choices=["mean", "max"])
    ap.add_argument("--use_zscore", action="store_true")
    ap.add_argument("--quantile", type=float, default=0.999)
    ap.add_argument("--hysteresis", type=int, default=3)
    ap.add_argument("--topk", type=int, default=2, help="number of sentinel nodes per subgraph")
    ap.add_argument("--channels", type=int, default=1)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    prep_dir = (root / args.prep_dir).resolve()
    graph_dir = (root / args.graph_dir).resolve()
    subgraph_path = (root / args.subgraph_file).resolve()
    score_dir = (root / args.score_dir).resolve()

    with open(subgraph_path, "r", encoding="utf-8") as f:
        subgraphs = json.load(f)
    with open(prep_dir / "features.json", "r", encoding="utf-8") as f:
        features = json.load(f)["features"]
    with open(prep_dir / "scaler.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    T_window = int(meta["T"])
    A = np.load(str(graph_dir / "A_topk.npy")).astype(np.float64)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.clip(A, 0.0, None)
    np.fill_diagonal(A, 1.0)

    if args.source == "mean":
        val = np.load(str(score_dir / "subgraph_score_val_mean.npy")).astype(np.float64)
        test = np.load(str(score_dir / "subgraph_score_test_mean.npy")).astype(np.float64)
    else:
        val = np.load(str(score_dir / "subgraph_score_val_max.npy")).astype(np.float64)
        test = np.load(str(score_dir / "subgraph_score_test_max.npy")).astype(np.float64)

    if args.use_zscore:
        mu = val.mean(axis=0)
        sigma = val.std(axis=0) + 1e-6
        val = (val - mu) / sigma
        test = (test - mu) / sigma

    subgraph_names = list(subgraphs.keys())
    S = len(subgraph_names)
    N_total = len(features)
    num_test = test.shape[0]

    # 每个子图单独阈值
    thresholds = np.quantile(val, args.quantile, axis=0)  # [S]
    triggered = (test > thresholds[None, :]).astype(np.int32)
    modes = apply_hysteresis_per_col(triggered, args.hysteresis)  # [num_test,S]

    # 选哨兵节点
    sentinel_info = {}
    normal_nodes_per_stage = []
    full_nodes_per_stage = []
    for s_idx, gname in enumerate(subgraph_names):
        nodes = subgraphs[gname]["nodes"]
        sentinels = pick_sentinels(A, nodes, args.topk)
        sentinel_info[gname] = {
            "size": len(nodes),
            "sentinel_idx": sentinels,
            "sentinel_names": [features[i] for i in sentinels],
        }
        normal_nodes_per_stage.append(len(sentinels))
        full_nodes_per_stage.append(len(nodes))

    normal_nodes_per_stage = np.array(normal_nodes_per_stage, dtype=np.int32)
    full_nodes_per_stage = np.array(full_nodes_per_stage, dtype=np.int32)

    # 每个窗口的上传节点数
    # 常态：每个 stage 上传 top-k sentinel
    # 异常：该 stage 上传 full nodes
    nodes_uploaded = np.zeros((num_test,), dtype=np.float64)
    for t in range(num_test):
        per_stage = np.where(modes[t] == 1, full_nodes_per_stage, normal_nodes_per_stage)
        nodes_uploaded[t] = per_stage.sum()

    # 通信量（按 "节点数 * 窗长 * 通道数" 近似）
    actual_load = nodes_uploaded * T_window * args.channels
    full_load = np.ones_like(actual_load) * (N_total * T_window * args.channels)

    avg_ratio = float(actual_load.mean() / full_load.mean())
    saved_ratio = float(1.0 - avg_ratio)

    trigger_ratio_global = float((modes.sum(axis=1) > 0).mean())
    trigger_ratio_per_stage = {
        subgraph_names[s]: float(modes[:, s].mean()) for s in range(S)
    }

    # 一个简单的全局 score 供参考：max over stage scores
    global_score = test.max(axis=1)
    global_thr = float(np.quantile(val.max(axis=1), args.quantile))
    global_pred = (global_score > global_thr).astype(np.int8)

    out_dir = score_dir / "scheduler_sim"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(out_dir / "modes.npy"), modes.astype(np.int8))
    np.save(str(out_dir / "nodes_uploaded.npy"), nodes_uploaded.astype(np.float32))
    np.save(str(out_dir / "global_pred.npy"), global_pred.astype(np.int8))

    report = {
        "source": args.source,
        "use_zscore": bool(args.use_zscore),
        "quantile": args.quantile,
        "hysteresis": args.hysteresis,
        "topk": args.topk,
        "window_len": T_window,
        "channels": args.channels,
        "num_nodes_total": N_total,
        "avg_upload_ratio_vs_full": avg_ratio,
        "avg_saved_ratio_vs_full": saved_ratio,
        "global_trigger_ratio": trigger_ratio_global,
        "trigger_ratio_per_stage": trigger_ratio_per_stage,
        "sentinel_info": sentinel_info,
    }

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== Scheduler Simulation ===")
    print(f"source={args.source}, use_zscore={args.use_zscore}, q={args.quantile}, H={args.hysteresis}, topk={args.topk}")
    print(f"avg_upload_ratio_vs_full = {avg_ratio:.4f}")
    print(f"avg_saved_ratio_vs_full  = {saved_ratio:.4f}")
    print(f"global_trigger_ratio     = {trigger_ratio_global:.4f}")
    print("saved:", str(out_dir))


if __name__ == "__main__":
    main()