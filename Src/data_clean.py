import os
import json
import argparse
import pandas as pd
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_best_datetime_parse(s: pd.Series) -> pd.Series:
    """
    SWaT 的 Timestamp 常见是日/月/年 + AM/PM。
    为了稳健：分别尝试 dayfirst=True/False，选择 NaT 更少的解析结果。
    """
    s_str = s.astype(str).str.strip()
    dt1 = pd.to_datetime(s_str, errors="coerce", dayfirst=True)
    dt2 = pd.to_datetime(s_str, errors="coerce", dayfirst=False)

    nat1 = int(dt1.isna().sum())
    nat2 = int(dt2.isna().sum())

    return dt1 if nat1 <= nat2 else dt2


def clean_label_col(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    lab = df[label_col].astype(str).str.strip()

    # 合并多余空白
    lab = lab.str.replace(r"\s+", " ", regex=True)

    # 修复常见脏 token
    lab = lab.replace(
        {
            "A ttack": "Attack",
            "Att ack": "Attack",
            "attack": "Attack",
            "normal": "Normal",
            "NORMAL": "Normal",
            "ATTACK": "Attack",
        }
    )

    df[label_col] = lab
    return df


def summarize_nan(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    nan_total = int(df[feat_cols].isna().sum().sum())
    size_total = int(df[feat_cols].size)
    nan_ratio = float(nan_total) / float(size_total) if size_total > 0 else 0.0
    top_nan_cols = (
        df[feat_cols].isna().sum().sort_values(ascending=False).head(10).to_dict()
    )
    return {
        "nan_total": nan_total,
        "nan_ratio": nan_ratio,
        "top_nan_cols": top_nan_cols,
    }


def clean_one_file(in_path: str, out_path: str) -> dict:
    df = pd.read_csv(in_path, encoding="utf-8-sig", low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    report = {
        "input_path": in_path,
        "output_path": out_path,
        "shape_before": [int(df.shape[0]), int(df.shape[1])],
        "columns": df.columns.tolist(),
    }

    # 找 Timestamp / 标签列
    if "Timestamp" not in df.columns:
        raise ValueError(f'No "Timestamp" column found in {in_path}')

    label_col = "Normal/Attack" if "Normal/Attack" in df.columns else None

    # 标签清洗
    if label_col is not None:
        df = clean_label_col(df, label_col)

        # 统计标签 token
        tokens = df[label_col].value_counts(dropna=False).to_dict()
        report["label_value_counts_before_mapping"] = {str(k): int(v) for k, v in tokens.items()}

        # 检查未知 token（不强制报错，但会记录）
        known = {"Normal", "Attack"}
        unknown = sorted(set(df[label_col].unique()) - known)
        report["label_unknown_tokens"] = unknown

    # 解析 Timestamp（自动选 dayfirst）
    df["Timestamp"] = pick_best_datetime_parse(df["Timestamp"])
    ts_nat = int(df["Timestamp"].isna().sum())
    report["timestamp_nat_before_drop"] = ts_nat

    # 删除 Timestamp 解析失败的行
    if ts_nat > 0:
        df = df[df["Timestamp"].notna()].copy()

    # 先删除整行完全重复（你现在 normal 的重复就是这种）
    before_dedup = len(df)
    df = df.drop_duplicates()
    report["rows_removed_by_drop_duplicates"] = int(before_dedup - len(df))

    # 排序
    df = df.sort_values("Timestamp").reset_index(drop=True)
    report["timestamp_monotonic_after_sort"] = bool(df["Timestamp"].is_monotonic_increasing)

    # 如果 Timestamp 仍重复，按 Timestamp 聚合取 last（更符合日志“最后写入”语义）
    ts_dup = int(df["Timestamp"].duplicated().sum())
    report["dup_timestamps_after_drop_duplicates"] = ts_dup

    if ts_dup > 0:
        feat_cols_tmp = [c for c in df.columns if c not in ["Timestamp"]]
        agg = {c: "last" for c in feat_cols_tmp}
        df = df.groupby("Timestamp", as_index=False).agg(agg)
        df = df.sort_values("Timestamp").reset_index(drop=True)
        report["rows_after_groupby_timestamp_last"] = int(df.shape[0])
        report["dup_timestamps_after_groupby"] = int(df["Timestamp"].duplicated().sum())

    # 特征列（排除 Timestamp 和 标签）
    drop_cols = ["Timestamp"]
    if label_col is not None:
        drop_cols.append(label_col)

    feat_cols = [c for c in df.columns if c not in drop_cols]
    report["feature_count"] = int(len(feat_cols))

    # 转数值
    # （为了避免 object 混入导致 NaN，先强制转）
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")

    # NaN 概况（补全前）
    report["nan_before_fill"] = summarize_nan(df, feat_cols)

    # 缺失值补全：ffill + bfill
    df[feat_cols] = df[feat_cols].ffill().bfill()

    # 若某列仍全 NaN（例如整列都无法解析成数值），用 0 填充并记录
    all_nan_cols = [c for c in feat_cols if df[c].isna().all()]
    if all_nan_cols:
        df[all_nan_cols] = 0.0
    report["all_nan_cols_filled_with_zero"] = all_nan_cols

    # NaN 概况（补全后）
    report["nan_after_fill"] = summarize_nan(df, feat_cols)

    # 输出前把 Timestamp 转回字符串（ISO 格式，便于后续读取一致）
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # 保存
    df.to_csv(out_path, index=False, encoding="utf-8")
    report["shape_after"] = [int(df.shape[0]), int(df.shape[1])]

    # 常量列统计（不默认删除，只报告；你后续训练时可选择 drop）
    nunique = df[feat_cols].nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    report["constant_feature_cols_count"] = int(len(const_cols))
    report["constant_feature_cols_sample"] = const_cols[:20]

    return report


def main():
    parser = argparse.ArgumentParser(description="Clean SWaT CSVs and save to dataset_clean/")
    parser.add_argument("--in_dir", default="dataset", help="Input dir containing normal.csv/attack.csv/merged.csv")
    parser.add_argument("--out_dir", default="dataset_clean", help="Output dir to write cleaned csvs")
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)

    # 你当前的文件名
    inputs = {
        "normal": os.path.join(in_dir, "normal.csv"),
        "attack": os.path.join(in_dir, "attack.csv"),
        "merged": os.path.join(in_dir, "merged.csv"),
    }

    for k, p in inputs.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    outputs = {
        "normal": os.path.join(out_dir, "normal_clean.csv"),
        "attack": os.path.join(out_dir, "attack_clean.csv"),
        "merged": os.path.join(out_dir, "merged_clean.csv"),
    }

    all_reports = {}
    for name in ["normal", "attack", "merged"]:
        print(f"\n=== Cleaning {name} ===")
        rep = clean_one_file(inputs[name], outputs[name])
        all_reports[name] = rep

        # 控制台摘要
        print("before:", rep["shape_before"], "after:", rep["shape_after"])
        print("removed by drop_duplicates:", rep["rows_removed_by_drop_duplicates"])
        print("timestamp NaT dropped:", rep["timestamp_nat_before_drop"])
        print("dup timestamps after dedup:", rep["dup_timestamps_after_drop_duplicates"])
        print("NaN ratio before fill:", rep["nan_before_fill"]["nan_ratio"])
        print("NaN ratio after  fill:", rep["nan_after_fill"]["nan_ratio"])
        print("constant cols (count):", rep["constant_feature_cols_count"])

        if rep.get("label_unknown_tokens"):
            if len(rep["label_unknown_tokens"]) > 0:
                print("WARNING: unknown label tokens:", rep["label_unknown_tokens"])

    # 保存清洗报告
    report_path = os.path.join(out_dir, "cleaning_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    print(f"\nSaved cleaned CSVs to: {out_dir}/")
    print(f"Saved report to: {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()