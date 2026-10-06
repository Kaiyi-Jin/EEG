"""分析数据.xlsx(sheet=分析数据): 测试时间 - EEG时间 (天)。
EEG时间 先用脑电 PDF 的检查日期更新(按 分析数据编号 对应); PDF 里没有的行保留 xlsx 原值(若格式合法)。
不修改原 xlsx。输出: output/analysis_time_diff.csv"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"


def main():
    a = pd.read_excel(ROOT / "分析数据.xlsx", sheet_name="分析数据", dtype={"编号": str})
    rep = pd.read_csv(OUT / "eeg_reports.csv", dtype=str, encoding="utf-8-sig")
    rep = rep.dropna(subset=["分析数据编号"]).drop_duplicates("分析数据编号")[["分析数据编号", "脑电编号", "检查日期"]]
    df = a[["编号", "姓名", "年龄", "性别", "测试时间", "EEG时间"]].merge(rep, left_on="编号", right_on="分析数据编号", how="left").drop(columns="分析数据编号")
    t = pd.to_datetime(df["测试时间"])
    old = pd.to_datetime(df["EEG时间"], errors="coerce")       # 非法格式 -> NaT
    pdf = pd.to_datetime(df["检查日期"], errors="coerce")
    new = pdf.fillna(old)
    df["EEG时间_原"] = df["EEG时间"].astype(str)
    df["EEG时间_更新"] = new.dt.date
    df["EEG时间来源"] = pdf.notna().map({True: "PDF检查日期", False: ""})
    df.loc[pdf.isna() & old.notna(), "EEG时间来源"] = "xlsx原值"
    df.loc[new.isna(), "EEG时间来源"] = "缺失"
    df["测试时间"] = t.dt.date
    df["测试-EEG(天)"] = (t - new).dt.days
    df["EEG时间被更改"] = (pdf.notna() & (pdf != old)).astype(int)
    df = df.drop(columns=["EEG时间", "检查日期"])
    df.to_csv(OUT / "analysis_time_diff.csv", index=False, encoding="utf-8-sig")

    d = df["测试-EEG(天)"].dropna()
    print(f"共 {len(df)} 行; 来源: {dict(df['EEG时间来源'].value_counts())}; 无法计算 {int(df['测试-EEG(天)'].isna().sum())}")
    print(f"EEG时间被 PDF 改动的行 {int(df['EEG时间被更改'].sum())} (其中 xlsx 原值非法 {int((pdf.notna() & old.isna()).sum())})")
    print(f"测试时间-EEG时间(天): n={len(d)}, 均值 {d.mean():.1f}, 中位 {d.median():.0f}, sd {d.std():.1f}, 范围 {d.min():.0f}~{d.max():.0f}")
    print("分位数:", d.quantile([.05, .25, .5, .75, .95]).round(0).to_dict())
    print(f"负值(测试早于EEG) {int((d < 0).sum())}; 0 天 {int((d == 0).sum())}; >30 天 {int((d > 30).sum())}; >90 天 {int((d > 90).sum())}")
    print(df.loc[(df["测试-EEG(天)"] < 0) | (df["测试-EEG(天)"] > 30) | df["测试-EEG(天)"].isna(),
                 ["编号", "姓名", "测试时间", "EEG时间_原", "EEG时间_更新", "EEG时间来源", "测试-EEG(天)"]].to_string())
    ch = df[df["EEG时间被更改"] == 1]
    print(f"\n被改动的 {len(ch)} 行:")
    print(ch[["编号", "姓名", "EEG时间_原", "EEG时间_更新", "测试-EEG(天)"]].to_string())


if __name__ == "__main__":
    main()
