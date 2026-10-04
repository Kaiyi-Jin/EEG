"""Step 2: 按 姓名+年龄+性别 匹配 分析数据.xlsx(sheet=分析数据)，
把其中的"编号"作为新列 分析数据编号 写入 eeg_reports.csv，便于后续 join。"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "output" / "eeg_reports.csv"
KEYS = ["姓名", "年龄", "性别"]
AGE_TOL = 2


def main():
    eeg = pd.read_csv(CSV, dtype=str, encoding="utf-8-sig").drop(columns=["分析数据编号"], errors="ignore")
    ana = pd.read_excel(ROOT / "分析数据.xlsx", sheet_name="分析数据", dtype=str)
    ana = ana[KEYS + ["编号"]].rename(columns={"编号": "分析数据编号"})
    for d in (eeg, ana):
        for k in KEYS:
            d[k] = d[k].str.strip()
    print("分析数据中 姓名+年龄+性别 重复:", ana.duplicated(KEYS, keep=False).sum())
    ana = ana.drop_duplicates(KEYS, keep=False)  # 不唯一的不匹配, 避免错配
    out = eeg.merge(ana, on=KEYS, how="left")
    out["匹配方式"] = out["分析数据编号"].notna().map({True: "姓名+年龄+性别", False: ""})
    # 兜底: 分析数据里的年龄多为测评时年龄, 与脑电报告常差 1~2 岁;
    # 姓名+性别唯一对应且年龄相差<=AGE_TOL 时也匹配, 并在"匹配方式"里标注, 可自行筛掉
    ana["_age"] = pd.to_numeric(ana["年龄"], errors="coerce")
    for i in out.index[out["分析数据编号"].isna()]:
        r = out.loc[i]
        if pd.isna(r["姓名"]):
            continue
        c = ana[(ana["姓名"] == r["姓名"]) & (ana["性别"] == r["性别"])
                & ((ana["_age"] - float(r["年龄"])).abs() <= AGE_TOL)]
        if len(c) == 1:
            out.loc[i, "分析数据编号"] = c["分析数据编号"].iloc[0]
            out.loc[i, "匹配方式"] = f"姓名+性别+年龄差<={AGE_TOL}"
    out.to_csv(CSV, index=False, encoding="utf-8-sig")
    miss = out[out["分析数据编号"].isna()]
    print(f"匹配 {out['分析数据编号'].notna().sum()}/{len(out)}; 未匹配 {len(miss)}")
    if len(miss):
        print(miss[KEYS + ["脑电编号"]].to_string())


if __name__ == "__main__":
    main()
