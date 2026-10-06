"""把 output/analysis_time_diff.csv 的 EEG时间_更新 / 来源 / 测试-EEG(天) 并入 output/eeg_analysis_joined.csv
(按 分析数据编号 = 编号)。可重复运行; join_cluster_outcome.py 重新生成 joined 表后需再运行一次。
先运行 time_diff.py 与 join_cluster_outcome.py。"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "output"
NEW = ["EEG时间_更新", "EEG时间来源", "EEG时间被更改", "测试-EEG(天)"]


def main():
    j = pd.read_csv(OUT / "eeg_analysis_joined.csv", dtype=str, encoding="utf-8-sig").drop(columns=NEW, errors="ignore")
    t = pd.read_csv(OUT / "analysis_time_diff.csv", dtype=str, encoding="utf-8-sig")[["编号"] + NEW].rename(columns={"编号": "分析数据编号"})
    out = j.merge(t, on="分析数据编号", how="left", validate="m:1")
    out.to_csv(OUT / "eeg_analysis_joined.csv", index=False, encoding="utf-8-sig")
    d = pd.to_numeric(out["测试-EEG(天)"])
    print(f"行数 {len(out)}; 有差值 {d.notna().sum()}; 无 {d.isna().sum()} (多为未匹配到分析数据); 中位 {d.median():.0f} 天, 范围 {d.min():.0f}~{d.max():.0f}")
    print("新增列:", NEW)


if __name__ == "__main__":
    main()
