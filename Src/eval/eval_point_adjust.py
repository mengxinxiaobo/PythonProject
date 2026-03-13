# Src/eval/eval_point_adjust.py
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


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Point-adjust / event-adjust:
    只要在一个真实异常区间内命中任意一个点，
    就把该真实异常区间全部置为预测异常。
    """
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32).copy()

    n = len(y_true)
    i = 0
    while i < n:
        if y_true[i] == 1:
            start = i
            while i < n and y_true[i] == 1:
                i += 1
            end = i  # [start, end)

            # 如果该真实异常区间中，预测命中了至少一个点
            if np.any(y_pred[start:end] == 1):
                y_pred[start:end] = 1
        else:
            i += 1

    return y_pred


def event_segments(y: np.ndarray):
    """
    把 0/1 序列转成异常区间 [(start,end), ...]，end 为开区间
    """
    y = y.astype(np.int32)
    segs = []
    n = len(y)
    i = 0
    while i < n:
        if y[i] == 1:
            s = i
            while i < n and y[i] == 1:
                i += 1
            e = i
            segs.append((s, e))
        else:
            i += 1
    return segs


def event_level_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    事件级评估：
    - 一个真实异常段，只要被命中一次，就算 TP_event
    - 没命中的真实异常段算 FN_event
    - 预测出来但不和任何真实异常段重叠的预测事件算 FP_event
    """
    true_segs = event_segments(y_true)
    pred_segs = event_segments(y_pred)

    matched_true = set()
    matched_pred = set()

    for pi, (ps, pe) in enumerate(pred_segs):
        for ti, (ts, te) in enumerate(true_segs):
            # 区间有交集
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


def apply_hysteresis(y_pred: np.ndarray, H: int) -> np.ndarray:
    """
    迟滞/持续性门控：
    连续 H 个点为 1，才触发该段报警。
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_dir", default="Src/runs/prep/swat_T100_S10_H1")
    ap.add_argument("--sub_dir", default="Src/runs/subgraph_models")
    ap.add_argument("--quantile", type=float, default=0.999, help="threshold quantile on score_val")
    ap.add_argument("--hysteresis", type=int, default=1, help="consecutive alarms needed, e.g. 1/3/5")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    prep_dir = (project_root / args.prep_dir).resolve()
    sub_dir = (project_root / args.sub_dir).resolve()

    score_val_path = sub_dir / "score_val.npy"
    score_test_path = sub_dir / "score_test.npy"
    y_true_path = prep_dir / "test_win_labels.npy"

    if not score_val_path.exists():
        raise FileNotFoundError(f"Missing {score_val_path}")
    if not score_test_path.exists():
        raise FileNotFoundError(f"Missing {score_test_path}")
    if not y_true_path.exists():
        raise FileNotFoundError(f"Missing {y_true_path}")

    score_val = np.load(str(score_val_path)).astype(np.float64)
    score_test = np.load(str(score_test_path)).astype(np.float64)
    y_true = np.load(str(y_true_path)).astype(np.int32)

    # 固定阈值（基于 val）
    thr = float(np.quantile(score_val, args.quantile))
    y_pred_raw = (score_test > thr).astype(np.int32)

    # 迟滞
    y_pred_h = apply_hysteresis(y_pred_raw, args.hysteresis)

    # point-wise
    p1, r1, f11, tp1, fp1, fn1 = f1_pr_recall(y_true, y_pred_h)

    # point-adjust
    y_pred_pa = point_adjust(y_true, y_pred_h)
    p2, r2, f12, tp2, fp2, fn2 = f1_pr_recall(y_true, y_pred_pa)

    # event-level
    p3, r3, f13, tp3, fp3, fn3, n_true_evt, n_pred_evt = event_level_metrics(y_true, y_pred_pa)

    print("\n=== Threshold Setting ===")
    print(f"quantile={args.quantile:.3f}, threshold={thr:.6f}, hysteresis={args.hysteresis}")

    print("\n=== Point-wise ===")
    print(f"P={p1:.4f} R={r1:.4f} F1={f11:.4f} | TP={tp1} FP={fp1} FN={fn1}")

    print("\n=== Point-Adjust ===")
    print(f"P={p2:.4f} R={r2:.4f} F1={f12:.4f} | TP={tp2} FP={fp2} FN={fn2}")

    print("\n=== Event-level ===")
    print(
        f"P={p3:.4f} R={r3:.4f} F1={f13:.4f} | "
        f"TP_evt={tp3} FP_evt={fp3} FN_evt={fn3} | "
        f"true_evt={n_true_evt} pred_evt={n_pred_evt}"
    )

    # 保存结果
    out_dir = sub_dir / "point_adjust_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(out_dir / "pred_raw.npy"), y_pred_raw.astype(np.int8))
    np.save(str(out_dir / "pred_hysteresis.npy"), y_pred_h.astype(np.int8))
    np.save(str(out_dir / "pred_point_adjust.npy"), y_pred_pa.astype(np.int8))

    print("\nsaved:", str(out_dir))


if __name__ == "__main__":
    main()