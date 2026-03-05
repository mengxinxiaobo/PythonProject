# Src/data/make_cache.py
import os
import json
import argparse
import numpy as np
import pandas as pd


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_csv(path: str):
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    # Timestamp 在清洗脚本里已转为 ISO 字符串；这里不强制解析，必要时你可解析再排序
    # 但建议仍按 Timestamp 排一下，防止数据源异常
    if "Timestamp" in df.columns:
        # 尝试解析，失败也不报错（NaT 会排到最后）
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.assign(_ts=ts).sort_values("_ts").drop(columns=["_ts"]).reset_index(drop=True)
    return df


def label_to_01(s: pd.Series) -> np.ndarray:
    lab = s.astype(str).str.strip()
    return (lab == "Attack").astype(np.int8).to_numpy()


def compute_scaler(train_values: np.ndarray, eps: float = 1e-6):
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std = np.where(std < eps, 1.0, std)  # 避免除0
    return mean, std


def make_starts(length: int, T: int, horizon: int, stride: int):
    max_start = length - T - horizon + 1
    if max_start <= 0:
        return np.zeros((0,), dtype=np.int64)
    return np.arange(0, max_start, stride, dtype=np.int64)


def make_window_labels(point_labels: np.ndarray, starts: np.ndarray, T: int, horizon: int):
    # 窗口标签：窗口覆盖范围内出现过 Attack 就标 1
    # 覆盖区间： [start, start+T+horizon-1]
    out = np.zeros((len(starts),), dtype=np.int8)
    w = T + horizon
    for i, st in enumerate(starts):
        out[i] = int(point_labels[st: st + w].max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", default="dataset_clean/normal_clean.csv")
    ap.add_argument("--merged", default="dataset_clean/merged_clean.csv")
    ap.add_argument("--out_dir", default="Src/runs/prep")
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    args = ap.parse_args()

    T = args.T
    stride = args.stride
    horizon = args.horizon

    run_dir = os.path.join(args.out_dir, f"swat_T{T}_S{stride}_H{horizon}")
    ensure_dir(run_dir)

    normal_df = load_csv(args.normal)
    merged_df = load_csv(args.merged)

    # 固定特征列顺序：除 Timestamp / Normal/Attack
    drop_cols = {"Timestamp", "Normal/Attack"}
    feat_cols = [c for c in normal_df.columns if c not in drop_cols]
    if set(feat_cols) != set([c for c in merged_df.columns if c not in drop_cols]):
        raise ValueError("Feature columns mismatch between normal and merged!")

    # 提取数值矩阵
    normal_x = normal_df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    merged_x = merged_df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    # point label（merged用来做测试窗口标签）
    if "Normal/Attack" not in merged_df.columns:
        raise ValueError("merged csv must contain Normal/Attack column for evaluation.")
    merged_y = label_to_01(merged_df["Normal/Attack"])

    # 切分 train/val（在 normal 上按时间顺序）
    n_total = len(normal_x)
    n_train = int(n_total * args.train_ratio)
    train_raw = normal_x[:n_train]

    mean, std = compute_scaler(train_raw)
    normal_x = (normal_x - mean) / std
    merged_x = (merged_x - mean) / std

    # 生成滑窗起点索引（train/val/test）
    # train/val 在 normal 上分别做成两个“独立序列”
    train_len = n_train
    val_len = n_total - n_train

    train_starts = make_starts(train_len, T, horizon, stride)
    val_starts = make_starts(val_len, T, horizon, stride)
    # val 的 starts 需要偏移到 normal_x 的后半段
    val_starts_global = val_starts + n_train

    test_starts = make_starts(len(merged_x), T, horizon, stride)
    test_win_labels = make_window_labels(merged_y, test_starts, T, horizon)

    # 保存
    np.save(os.path.join(run_dir, "normal_values.npy"), normal_x.astype(np.float32))
    np.save(os.path.join(run_dir, "merged_values.npy"), merged_x.astype(np.float32))
    np.save(os.path.join(run_dir, "train_starts.npy"), train_starts)
    np.save(os.path.join(run_dir, "val_starts.npy"), val_starts_global)
    np.save(os.path.join(run_dir, "test_starts.npy"), test_starts)
    np.save(os.path.join(run_dir, "test_win_labels.npy"), test_win_labels)

    with open(os.path.join(run_dir, "scaler.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"mean": mean.tolist(), "std": std.tolist(), "T": T, "stride": stride, "horizon": horizon},
            f, ensure_ascii=False, indent=2
        )

    with open(os.path.join(run_dir, "features.json"), "w", encoding="utf-8") as f:
        json.dump({"features": feat_cols}, f, ensure_ascii=False, indent=2)

    print("Saved cache to:", run_dir)
    print("normal_x:", normal_x.shape, "merged_x:", merged_x.shape)
    print("train windows:", len(train_starts), "val windows:", len(val_starts_global), "test windows:", len(test_starts))
    print("test positive windows:", int(test_win_labels.sum()))


if __name__ == "__main__":
    main()