# Src/eval/eval_stage_fusion.py
import argparse
import json
from pathlib import Path
import numpy as np


def f1_pr_recall(y_true, y_pred):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1, tp, fp, fn


def point_adjust(y_true, y_pred):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32).copy()
    n = len(y_true)
    i = 0
    while i < n:
        if y_true[i] == 1:
            s = i
            while i < n and y_true[i] == 1:
                i += 1
            e = i
            if np.any(y_pred[s:e] == 1):
                y_pred[s:e] = 1
        else:
            i += 1
    return y_pred


def event_segments(y):
    y = y.astype(np.int32)
    segs = []
    n = len(y)
    i = 0
    while i < n:
        if y[i] == 1:
            s = i
            while i < n and y[i] == 1:
                i += 1
            segs.append((s, i))
        else:
            i += 1
    return segs


def event_level_metrics(y_true, y_pred):
    true_segs = event_segments(y_true)
    pred_segs = event_segments(y_pred)

    matched_true = set()
    matched_pred = set()

    for pi, (ps, pe) in enumerate(pred_segs):
        for ti, (ts, te) in enumerate(true_segs):
            if not (pe <= ts or te <= ps):
                matched_true.add(ti)
                matched_pred.add(pi)

    tp = len(matched_true)
    fn = len(true_segs) - tp
    fp = len(pred_segs) - len(matched_pred)

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1, tp, fp, fn, len(true_segs), len(pred_segs)


def apply_hysteresis(y_pred, H):
    if H <= 1:
        return y_pred.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    out = np.zeros_like(y_pred)
    n = len(y_pred)
    i = 0
    while i < n:
        if y_pred[i] == 1:
            s = i
            while i < n and y_pred[i] == 1:
                i += 1
            e = i
            if (e - s) >= H:
                out[s:e] = 1
        else:
            i += 1
    return out


def fuse_scores(score_mat, mode):
    mode = mode.lower()
    if mode == "max":
        return score_mat.max(axis=1)
    if mode == "mean":
        return score_mat.mean(axis=1)
    if mode == "top2mean":
        part = np.partition(score_mat, -2, axis=1)[:, -2:]
        return part.mean(axis=1)
    if mode == "top3mean":
        k = min(3, score_mat.shape[1])
        part = np.partition(score_mat, -k, axis=1)[:, -k:]
        return part.mean(axis=1)
    if mode == "p95":
        return np.percentile(score_mat, 95, axis=1)
    raise ValueError(f"unknown mode={mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_dir", default="Src/runs/prep/swat_T100_S10_H1")
    ap.add_argument("--sub_dir", default="Src/runs/subgraph_models_stage")
    ap.add_argument("--source", default="mean", choices=["mean", "max"])
    ap.add_argument("--use_zscore", action="store_true")
    ap.add_argument("--mode", default="max", choices=["max", "mean", "top2mean", "top3mean", "p95"])
    ap.add_argument("--quantile", type=float, default=0.999)
    ap.add_argument("--hysteresis", type=int, default=1)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    prep_dir = (root / args.prep_dir).resolve()
    sub_dir = (root / args.sub_dir).resolve()

    y_true = np.load(str(prep_dir / "test_win_labels.npy")).astype(np.int32)

    if args.source == "mean":
        val = np.load(str(sub_dir / "subgraph_score_val_mean.npy")).astype(np.float64)
        test = np.load(str(sub_dir / "subgraph_score_test_mean.npy")).astype(np.float64)
    else:
        val = np.load(str(sub_dir / "subgraph_score_val_max.npy")).astype(np.float64)
        test = np.load(str(sub_dir / "subgraph_score_test_max.npy")).astype(np.float64)

    if args.use_zscore:
        mu = val.mean(axis=0)
        sigma = val.std(axis=0) + 1e-6
        val = (val - mu) / sigma
        test = (test - mu) / sigma

    score_val = fuse_scores(val, args.mode)
    score_test = fuse_scores(test, args.mode)

    thr = float(np.quantile(score_val, args.quantile))
    pred_raw = (score_test > thr).astype(np.int32)
    pred_h = apply_hysteresis(pred_raw, args.hysteresis)

    p1, r1, f1_1, tp1, fp1, fn1 = f1_pr_recall(y_true, pred_h)

    pred_pa = point_adjust(y_true, pred_h)
    p2, r2, f1_2, tp2, fp2, fn2 = f1_pr_recall(y_true, pred_pa)

    p3, r3, f1_3, tp3, fp3, fn3, evt_true, evt_pred = event_level_metrics(y_true, pred_pa)

    print("\n=== Stage Fusion Setting ===")
    print(
        f"source={args.source}, use_zscore={args.use_zscore}, "
        f"mode={args.mode}, q={args.quantile}, hysteresis={args.hysteresis}, thr={thr:.6f}"
    )

    print("\n=== Point-wise ===")
    print(f"P={p1:.4f} R={r1:.4f} F1={f1_1:.4f} | TP={tp1} FP={fp1} FN={fn1}")

    print("\n=== Point-Adjust ===")
    print(f"P={p2:.4f} R={r2:.4f} F1={f1_2:.4f} | TP={tp2} FP={fp2} FN={fn2}")

    print("\n=== Event-level ===")
    print(
        f"P={p3:.4f} R={r3:.4f} F1={f1_3:.4f} | "
        f"TP_evt={tp3} FP_evt={fp3} FN_evt={fn3} | "
        f"true_evt={evt_true} pred_evt={evt_pred}"
    )

    out_dir = sub_dir / "stage_fusion_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / "score_val.npy"), score_val.astype(np.float32))
    np.save(str(out_dir / "score_test.npy"), score_test.astype(np.float32))
    np.save(str(out_dir / "pred_raw.npy"), pred_raw.astype(np.int8))
    np.save(str(out_dir / "pred_hysteresis.npy"), pred_h.astype(np.int8))
    np.save(str(out_dir / "pred_point_adjust.npy"), pred_pa.astype(np.int8))
    print("saved:", str(out_dir))


if __name__ == "__main__":
    main()