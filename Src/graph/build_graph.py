# Src/graph/build_graph.py
import json
from pathlib import Path

import numpy as np
import torch


def sanitize_affinity(A: np.ndarray) -> np.ndarray:
    """
    将相似度矩阵转成谱聚类可用的 affinity：
    - 去 NaN/Inf
    - 映射到 [0, 1]
    - 对称化、对角置 1
    """
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    # 余弦相似度范围[-1,1] -> affinity[0,1]
    A = (A + 1.0) / 2.0
    A = np.clip(A, 0.0, 1.0)

    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0)
    return A


def topk_sparsify(A: np.ndarray, k: int) -> np.ndarray:
    """对每行保留 top-k（不含对角），返回稀疏化后的 dense affinity（非负）"""
    N = A.shape[0]
    out = np.zeros_like(A, dtype=np.float32)
    for i in range(N):
        row = A[i].copy()
        row[i] = -np.inf
        idx = np.argpartition(row, -k)[-k:]
        out[i, idx] = A[i, idx]
    out = np.maximum(out, out.T)
    np.fill_diagonal(out, 1.0)
    return out


def diffusion_similarity(A: np.ndarray, hops: int = 4, alpha: float = 0.3) -> np.ndarray:
    """
    扩散核：S = Σ_{h=1..hops} alpha^h P^h
    注意：这里要求 A 非负，否则 row-normalize 会出问题。
    """
    A = np.clip(A, 0.0, None)
    D = A.sum(axis=1, keepdims=True)
    P = A / (D + 1e-12)

    S = np.zeros_like(A, dtype=np.float32)
    Ph = P.copy()
    for h in range(1, hops + 1):
        S += (alpha ** h) * Ph
        Ph = Ph @ P

    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S


@torch.no_grad()
def build_average_cosine(emb_path: str, device: str = "cuda", batch_windows: int = 512) -> np.ndarray:
    """
    emb_train: [K, N, D]
    A_cos[i,j] = mean_k cos(z_k_i, z_k_j)
    """
    emb = np.load(emb_path, mmap_mode="r")
    K, N, D = emb.shape

    A_sum = torch.zeros((N, N), device=device, dtype=torch.float32)
    seen = 0

    for st in range(0, K, batch_windows):
        ed = min(K, st + batch_windows)
        z = torch.from_numpy(np.array(emb[st:ed], copy=True)).to(device)  # [B,N,D]
        z = z / (torch.norm(z, dim=-1, keepdim=True) + 1e-12)

        A_batch = torch.einsum("bnd,bmd->nm", z, z)  # [N,N]
        A_sum += A_batch
        seen += (ed - st)

    A_cos = (A_sum / max(seen, 1)).detach().cpu().numpy()
    A_cos = 0.5 * (A_cos + A_cos.T)
    np.fill_diagonal(A_cos, 1.0)
    return A_cos


def main():
    project_root = Path(__file__).resolve().parents[2]  # Src/graph/.. -> 项目根
    emb_dir = project_root / "Src" / "runs" / "embeddings"
    out_dir = project_root / "Src" / "runs" / "graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_train = emb_dir / "emb_train.npy"
    if not emb_train.exists():
        raise FileNotFoundError(f"Missing {emb_train}. Run: python -m Src.embed_cae")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # 超参：先用这一组跑通
    topk = 8
    hops = 4
    alpha = 0.3
    gamma = 0.5

    A_cos_raw = build_average_cosine(str(emb_train), device=device, batch_windows=512)
    # 关键：把余弦相似变成非负 affinity
    A_cos = sanitize_affinity(A_cos_raw)

    A_topk = topk_sparsify(A_cos, k=topk)
    S_diff = diffusion_similarity(A_topk, hops=hops, alpha=alpha)

    A_fused = gamma * A_cos + (1.0 - gamma) * S_diff
    A_fused = sanitize_affinity(A_fused)  # 再清一次，确保无 NaN/负值

    # Debug 信息（建议保留，能快速排雷）
    print("A_fused min/max:", float(A_fused.min()), float(A_fused.max()))
    print("A_fused nan count:", int(np.isnan(A_fused).sum()))

    np.save(str(out_dir / "A_cos.npy"), A_cos.astype(np.float32))
    np.save(str(out_dir / "A_topk.npy"), A_topk.astype(np.float32))
    np.save(str(out_dir / "A_fused.npy"), A_fused.astype(np.float32))

    meta = {"topk": topk, "hops": hops, "alpha": alpha, "gamma": gamma}
    with open(out_dir / "graph_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("saved:", str(out_dir))
    print("A_fused shape:", A_fused.shape, "topk:", topk)


if __name__ == "__main__":
    main()