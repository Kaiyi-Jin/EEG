"""分析数据.xlsx(sheet=分析数据): 测试时间 - EEG时间 (天)。
EEG时间中格式非法的值(如 2025-0-16)不猜测, 留空并标记; 同时用脑电 PDF 的检查日期做交叉核对。
输出: output/analysis_time_diff.csv"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"


def main():
    a = pd.read_excel(ROOT / "分析数据.xlsx", sheet_name="分析数据", dtype={"编号": str})
    t = pd.to_datetime(a["测试时间"], errors="coerce")
    e = pd.to_datetime(a["EEG时间"], errors="coerce")
    df = a[["编号", "姓名", "年龄", "性别", "测试时间", "EEG时间"]].copy()
    df["测试时间"], df["EEG时间_解析"] = t.dt.date, e.dt.date
    df["测试-EEG(天)"] = (t - e).dt.days
    df["备注"] = ""
    df.loc[e.isna(), "备注"] = "EEG时间格式非法, 未计算"
    # 交叉核对: 脑电 PDF 检查日期
    rep = pd.read_csv(OUT / "eeg_reports.csv", dtype=str, encoding="utf-8-sig")
    rep = rep.dropna(subset=["分析数据编号"]).drop_duplicates("分析数据编号")[["分析数据编号", "脑电编号", "检查日期"]]
    df = df.merge(rep, left_on="编号", right_on="分析数据编号", how="left").drop(columns="分析数据编号")
    df["PDF检查日期"] = pd.to_datetime(df.pop("检查日期"), errors="coerce")
    df["测试-PDF检查日期(天)"] = (t.values - df["PDF检查日期"].values) / pd.Timedelta(days=1)
    df["EEG时间-PDF检查日期(天)"] = (e.values - df["PDF检查日期"].values) / pd.Timedelta(days=1)
    df["PDF检查日期"] = df["PDF检查日期"].dt.date
    df.to_csv(OUT / "analysis_time_diff.csv", index=False, encoding="utf-8-sig")

    d = df["测试-EEG(天)"].dropna()
    print(f"共 {len(df)} 行; 可计算 {len(d)}; 无法计算 {df['备注'].ne('').sum()}: {df.loc[df['备注'] != '', ['编号', '姓名', 'EEG时间']].values.tolist()}")
    print(f"测试时间-EEG时间(天): 均值 {d.mean():.1f}, 中位 {d.median():.0f}, sd {d.std():.1f}, 范围 {d.min():.0f}~{d.max():.0f}")
    print("分位数:", d.quantile([.05, .25, .5, .75, .95]).round(0).to_dict())
    print(f"负值(测试早于EEG) {int((d < 0).sum())} 行; 0天 {int((d == 0).sum())} 行; >30天 {int((d > 30).sum())}; >90天 {int((d > 90).sum())}")
    x = df["EEG时间-PDF检查日期(天)"].dropna()
    print(f"交叉核对: EEG时间与 PDF 检查日期 可比 {len(x)} 行, 完全一致 {int((x == 0).sum())} 行, 相差>3天 {int((x.abs() > 3).sum())} 行")
    print(df.loc[df["EEG时间-PDF检查日期(天)"].abs() > 3, ["编号", "姓名", "EEG时间_解析", "PDF检查日期", "EEG时间-PDF检查日期(天)"]].to_string())
    print(df.loc[df["备注"] != "", ["编号", "姓名", "EEG时间", "PDF检查日期", "测试-PDF检查日期(天)"]].to_string())


if __name__ == "__main__":
    main()
