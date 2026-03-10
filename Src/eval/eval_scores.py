# Src/eval/eval_scores.py
import argparse
from pathlib import Path
import numpy as np


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


def compute_score(z: np.ndarray, mode: str) -> np.ndarray:
    """
    z: [num_windows, N]
    """
    mode = mode.lower()
    if mode == "max":
        return z.max(axis=1)
    if mode == "mean":
        return z.mean(axis=1)
    if mode.startswith("p"):
        # p95, p99 ...
        p = float(mode[1:])
        return np.percentile(z, p, axis=1)
    raise ValueError(f"Unknown mode: {mode}. Use one of: max, mean, p95, p99, p90...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_dir", default="Src/runs/prep/swat_T100_S10_H1", help="prep cache dir")
    ap.add_argument("--sub_dir", default="Src/runs/subgraph_models", help="subgraph_models output dir")
    ap.add_argument("--modes", default="max,p99,p95,mean", help="comma-separated score modes")
    ap.add_argument("--quantiles", default="0.990,0.995,0.997,0.999", help="comma-separated threshold quantiles")
    ap.add_argument("--use_zscore", action="store_true", help="apply node-wise z-score using VAL stats")
    ap.add_argument("--topk", type=int, default=10, help="print top-k settings by F1")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]  # Src/eval/.. -> project root
    prep_dir = (project_root / args.prep_dir).resolve()
    sub_dir = (project_root / args.sub_dir).resolve()

    val_path = sub_dir / "global_node_res_val.npy"
    test_path = sub_dir / "global_node_res_test.npy"
    y_path = prep_dir / "test_win_labels.npy"

    if not val_path.exists():
        raise FileNotFoundError(f"Missing {val_path}. Run: python -m Src.train_subgraph_models")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}. Run: python -m Src.train_subgraph_models")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing {y_path}. Run: python -m Src.data.make_cache ...")

    val = np.load(str(val_path))   # [num_val, N]
    test = np.load(str(test_path)) # [num_test, N]
    y_true = np.load(str(y_path)).astype(np.int8)  # [num_test]

    # optional: node-wise z-score normalize based on val (normal)
    if args.use_zscore:
        mu = val.mean(axis=0)
        sigma = val.std(axis=0) + 1e-6
        val = (val - mu) / sigma
        test = (test - mu) / sigma

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    qs = [float(x.strip()) for x in args.quantiles.split(",") if x.strip()]

    results = []
    for mode in modes:
        score_val = compute_score(val, mode)
        score_test = compute_score(test, mode)

        for q in qs:
            thr = float(np.quantile(score_val, q))
            y_pred = (score_test > thr).astype(np.int8)

            p, r, f1, tp, fp, fn = f1_pr_recall(y_true[:len(y_pred)], y_pred)
            results.append({
                "mode": mode,
                "q": q,
                "thr": thr,
                "precision": p,
                "recall": r,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })

    # sort by f1 desc, then precision desc
    results.sort(key=lambda d: (d["f1"], d["precision"]), reverse=True)

    print("\n=== Eval Results ===")
    print("use_zscore:", bool(args.use_zscore))
    print("val shape:", val.shape, "test shape:", test.shape, "y_true shape:", y_true.shape)
    print("\nTop settings by F1:")
    for i, r in enumerate(results[:args.topk], start=1):
        print(
            f"{i:02d}. mode={r['mode']:<4} q={r['q']:.3f} thr={r['thr']:.6f} | "
            f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} | "
            f"TP={r['tp']} FP={r['fp']} FN={r['fn']}"
        )

    # also show best for each mode
    print("\nBest per mode:")
    for mode in modes:
        best = next(x for x in results if x["mode"] == mode)
        print(
            f"mode={mode:<4} q={best['q']:.3f} thr={best['thr']:.6f} | "
            f"P={best['precision']:.4f} R={best['recall']:.4f} F1={best['f1']:.4f}"
        )


if __name__ == "__main__":
    main()