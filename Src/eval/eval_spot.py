import argparse
from pathlib import Path
import numpy as np
from scipy.stats import genpareto


def f1_pr_recall(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1, tp, fp, fn


def score_from_global_res(global_node_res: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """node-wise zscore + max score"""
    z = (global_node_res - mu) / (sigma + 1e-6)
    return z.max(axis=1)


def fit_gpd(excess: np.ndarray):
    """
    Fit GPD to excess (x-u) where x>u.
    Force loc=0 for stability.
    Returns (shape, scale).
    """
    if len(excess) < 50:
        return 0.0, float(np.std(excess) + 1e-6)
    c, loc, scale = genpareto.fit(excess, floc=0.0)
    scale = float(max(scale, 1e-6))
    c = float(c)
    return c, scale


def gpd_threshold(u: float, q: float, p_u: float, shape: float, scale: float) -> float:
    """
    Want P(X > t) = q.
    With POT: P(X>t) ~= p_u * (1 - GPD_CDF(t-u)).
    So 1 - GPD_CDF(t-u) = q / p_u.
    t = u + GPD_PPF(1 - q/p_u).
    """
    if p_u <= 0:
        return float("inf")
    ratio = q / p_u
    ratio = min(max(ratio, 1e-12), 0.999999)  # keep in (0,1)
    p = 1.0 - ratio
    # genpareto.ppf expects p in [0,1]
    return float(u + genpareto.ppf(p, c=shape, loc=0.0, scale=scale))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_dir", default="Src/runs/prep/swat_T100_S10_H1")
    ap.add_argument("--sub_dir", default="Src/runs/subgraph_models")
    ap.add_argument("--init_q", type=float, default=0.98, help="initial threshold quantile for POT (on val)")
    ap.add_argument("--risk", type=float, default=1e-3, help="target tail probability q (e.g. 1e-3 ~ 0.999)")
    ap.add_argument("--refit_every", type=int, default=200, help="refit GPD every N new excesses")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    prep_dir = (project_root / args.prep_dir).resolve()
    sub_dir = (project_root / args.sub_dir).resolve()

    val_path = sub_dir / "global_node_res_val.npy"
    test_path = sub_dir / "global_node_res_test.npy"
    y_path = prep_dir / "test_win_labels.npy"

    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing global_node_res_*.npy. Run: python -m Src.train_subgraph_models")
    if not y_path.exists():
        raise FileNotFoundError("Missing test_win_labels.npy. Run: python -m Src.data.make_cache ...")

    val = np.load(str(val_path)).astype(np.float64)    # [num_val, N] (normal)
    test = np.load(str(test_path)).astype(np.float64)  # [num_test, N]
    y_true = np.load(str(y_path)).astype(np.int8)

    # --- node-wise z-score statistics from VAL(normal) ---
    mu = val.mean(axis=0)
    sigma = val.std(axis=0) + 1e-6

    # --- scores ---
    score_val = score_from_global_res(val, mu, sigma)
    score_test = score_from_global_res(test, mu, sigma)

    # --- POT init ---
    u = float(np.quantile(score_val, args.init_q))
    excess = score_val[score_val > u] - u
    p_u = float(len(excess) / len(score_val))
    shape, scale = fit_gpd(excess)

    print("init_q:", args.init_q, "risk:", args.risk)
    print("u:", u, "p_u:", p_u, "excess_count:", len(excess))
    print("GPD shape:", shape, "scale:", scale)

    # dynamic threshold over test
    thresholds = np.zeros_like(score_test, dtype=np.float64)
    y_pred = np.zeros_like(score_test, dtype=np.int8)

    excess_list = list(excess.tolist())
    new_excess = 0

    for i, x in enumerate(score_test):
        # current threshold computed from current GPD fit
        t = gpd_threshold(u=u, q=args.risk, p_u=p_u, shape=shape, scale=scale)
        thresholds[i] = t

        if x > t:
            y_pred[i] = 1  # anomaly
            # 不用异常点更新阈值（常见做法：异常点不纳入正常尾部建模）
            continue

        # non-anomalous update
        if x > u:
            excess_list.append(float(x - u))
            new_excess += 1

            # update p_u with streaming count
            # (approx) keep p_u stable; you can also update u periodically if you want
            p_u = float(len(excess_list) / (len(score_val) + i + 1))

            # refit occasionally
            if new_excess >= args.refit_every:
                arr = np.array(excess_list, dtype=np.float64)
                shape, scale = fit_gpd(arr)
                new_excess = 0

    p, r, f1, tp, fp, fn = f1_pr_recall(y_true[:len(y_pred)], y_pred)
    print("\n=== SPOT Results (zscore + max score) ===")
    print(f"P={p:.4f} R={r:.4f} F1={f1:.4f} | TP={tp} FP={fp} FN={fn}")

    out_dir = sub_dir / "spot"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / "score_val.npy"), score_val.astype(np.float32))
    np.save(str(out_dir / "score_test.npy"), score_test.astype(np.float32))
    np.save(str(out_dir / "thresholds.npy"), thresholds.astype(np.float32))
    np.save(str(out_dir / "pred_test.npy"), y_pred.astype(np.int8))
    print("saved:", str(out_dir))


if __name__ == "__main__":
    main()