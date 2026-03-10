# Src/graph/make_subgraphs.py
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering


def main():
    project_root = Path(__file__).resolve().parents[2]
    graph_dir = project_root / "Src" / "runs" / "graph"
    prep_dir = project_root / "Src" / "runs" / "prep" / "swat_T100_S10_H1"
    out_dir = project_root / "Src" / "runs" / "subgraphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    A_fused_path = graph_dir / "A_fused.npy"
    if not A_fused_path.exists():
        raise FileNotFoundError(f"Missing {A_fused_path}. Run: python -m Src.graph.build_graph")

    with open(prep_dir / "features.json", "r", encoding="utf-8") as f:
        feat = json.load(f)["features"]  # 长度 N
    N = len(feat)

    A = np.load(str(A_fused_path))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0)

    # 超参：先跑通
    C = 8            # 社区数
    beta = 0.85      # 重叠强度：越大越保守（1.0=不重叠）
    max_extra = 1    # 每个节点最多额外加入几个社区（防爆）

    sc = SpectralClustering(
        n_clusters=C,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    labels = sc.fit_predict(A)

    # 初始硬社区
    clusters = {c: [] for c in range(C)}
    for i in range(N):
        clusters[int(labels[i])].append(i)

    # 计算节点对社区的平均相似度 sim(i,c)
    sim_ic = np.zeros((N, C), dtype=np.float32)
    for c in range(C):
        idx = clusters[c]
        if len(idx) == 0:
            continue
        sim_ic[:, c] = A[:, idx].mean(axis=1)

    # 重叠扩展：基于 beta * 主社区相似度
    membership = {i: [int(labels[i])] for i in range(N)}
    for i in range(N):
        main_c = int(labels[i])
        base = float(sim_ic[i, main_c])
        # 候选社区（除主社区）
        cand = []
        for c in range(C):
            if c == main_c:
                continue
            if sim_ic[i, c] >= beta * base:
                cand.append((float(sim_ic[i, c]), c))
        cand.sort(reverse=True)
        for _, c in cand[:max_extra]:
            membership[i].append(int(c))

    # 反推每个社区的节点列表（重叠后）
    clusters_overlap = {c: [] for c in range(C)}
    for i, cs in membership.items():
        for c in cs:
            clusters_overlap[c].append(i)

    # 保存：用变量名增强可读性
    subgraphs = {}
    for c in range(C):
        subgraphs[f"G{c}"] = {
            "nodes": clusters_overlap[c],
            "node_names": [feat[i] for i in clusters_overlap[c]],
            "size": len(clusters_overlap[c]),
        }

    node_membership = {feat[i]: [f"G{c}" for c in membership[i]] for i in range(N)}

    with open(out_dir / "subgraphs.json", "w", encoding="utf-8") as f:
        json.dump(subgraphs, f, ensure_ascii=False, indent=2)
    with open(out_dir / "node_membership.json", "w", encoding="utf-8") as f:
        json.dump(node_membership, f, ensure_ascii=False, indent=2)

    # 打印一个摘要
    sizes = [subgraphs[f"G{c}"]["size"] for c in range(C)]
    avg_membership = np.mean([len(v) for v in membership.values()])
    print("saved:", str(out_dir))
    print("C:", C, "sizes:", sizes, "avg_membership:", float(avg_membership))


if __name__ == "__main__":
    main()