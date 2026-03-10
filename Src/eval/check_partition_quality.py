import json
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[2]
A = np.load(root/"Src/runs/graph/A_fused.npy").astype(np.float64)
with open(root/"Src/runs/subgraphs/subgraphs.json","r",encoding="utf-8") as f:
    subs = json.load(f)

A = 0.5*(A + A.T)
np.fill_diagonal(A, 0.0)

# 统计每个子图的 intra 平均
for g, info in subs.items():
    nodes = info["nodes"]
    if len(nodes) < 2:
        continue
    subA = A[np.ix_(nodes, nodes)]
    intra = subA[subA > 0].mean() if (subA > 0).any() else 0.0
    print(f"{g} size={len(nodes)} intra_mean={intra:.4f}")

# 统计跨子图平均
all_nodes = list(range(A.shape[0]))
# 简单：把所有“不同社区的节点对”拿出来均值（近似）
# 先给每个节点一个主社区（选它所属子图中size最大的那个）
node_main = {}
for i in all_nodes:
    cand = [(g, len(subs[g]["nodes"])) for g in subs if i in subs[g]["nodes"]]
    if not cand:
        node_main[i] = None
    else:
        node_main[i] = max(cand, key=lambda x: x[1])[0]

mask = np.zeros_like(A, dtype=bool)
for i in all_nodes:
    for j in all_nodes:
        if i>=j:
            continue
        if node_main[i] is not None and node_main[j] is not None and node_main[i] != node_main[j]:
            mask[i,j] = True
inter_vals = A[mask]
print("inter_mean:", float(inter_vals.mean()) if inter_vals.size else None)