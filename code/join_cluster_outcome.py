"""join eeg_reports.csv 与 分析数据.xlsx(sheet=分析数据), 并入每人的 GMM 聚类结果,
检验聚类与 睡眠减分率 / 总分减分率 的关联。
输出: output/eeg_analysis_joined.csv, output/clustering/outcome_association.txt"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
OUTCOMES = ["睡眠减分率", "总分减分率"]
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def main():
    eeg = pd.read_csv(OUT / "eeg_reports.csv", dtype=str, encoding="utf-8-sig")
    ana = pd.read_excel(ROOT / "分析数据.xlsx", sheet_name="分析数据", dtype={"编号": str})
    clu = pd.read_csv(OUT / "clustering" / "cluster_assignments.csv", dtype={"脑电编号": str})
    clu = clu[["脑电编号", "cluster", "max_prob", "entropy", "PC1", "PC2"]]
    clu["cluster_prob1"] = np.nan
    ana = ana.rename(columns={"姓名": "姓名_分析", "年龄": "年龄_分析", "性别": "性别_分析", "编号": "分析数据编号"})
    df = eeg.merge(ana, on="分析数据编号", how="left", validate="m:1").merge(clu, on="脑电编号", how="left")
    df = df.drop(columns="cluster_prob1")
    df.to_csv(OUT / "eeg_analysis_joined.csv", index=False, encoding="utf-8-sig")
    log(f"join: EEG {len(eeg)} 行; 匹配到分析数据 {df['姓名_分析'].notna().sum()}; 有聚类结果 {df['cluster'].notna().sum()}; "
        f"两者都有 {(df['cluster'].notna() & df['姓名_分析'].notna()).sum()}")

    d = df[df["cluster"].notna() & df[OUTCOMES[0]].notna()].copy()
    d["cluster"] = d["cluster"].astype(int)
    d["age"] = pd.to_numeric(d["年龄"])
    d["male"] = (d["性别"] == "男").astype(int)
    log(f"\n分析样本 n={len(d)}; 聚类大小 {dict(d['cluster'].value_counts().sort_index())}")
    log("  (聚类 0 = 低 α/平谱/年长组, 聚类 1 = 高 α 组, 见 cluster_profile.csv)")
    for o in OUTCOMES:
        d[o] = pd.to_numeric(d[o])
        log(f"\n=== {o} ===  取值: 均值 {d[o].mean():.2f}, 中位 {d[o].median():.2f}, 范围 {d[o].min():.2f}~{d[o].max():.2f}")
        g = [d.loc[d.cluster == c, o] for c in sorted(d.cluster.unique())]
        for c, x in zip(sorted(d.cluster.unique()), g):
            log(f"  聚类{c}: n={len(x)} 均值 {x.mean():.3f} 中位 {x.median():.3f} sd {x.std():.3f}")
        u, p = stats.mannwhitneyu(g[0], g[1], alternative="two-sided")
        auc = u / (len(g[0]) * len(g[1]))
        rb, pr = stats.pointbiserialr(d["cluster"], d[o])
        log(f"  Mann-Whitney U p={p:.3f} (聚类0 > 聚类1 的概率 AUC={auc:.2f}); 点二列相关 r={rb:.3f} p={pr:.3f}")
        # 排列检验
        rng = np.random.default_rng(0)
        obs = abs(g[0].mean() - g[1].mean())
        perm = [abs(a[:len(g[0])].mean() - a[len(g[0]):].mean()) for a in (rng.permutation(d[o].values) for _ in range(5000))]
        log(f"  均值差 {g[0].mean() - g[1].mean():+.3f}, 排列检验 p={(np.sum(np.array(perm) >= obs) + 1) / 5001:.3f}")
        # 与连续量的关系
        for v, lab in [("max_prob", "聚类后验概率"), ("PC1", "PC1")]:
            rho, pp = stats.spearmanr(d[v], d[o])
            log(f"  Spearman({lab}, {o}): rho={rho:.3f} p={pp:.3f}")
        # 校正年龄、性别
        m = smf.ols(f"Q('{o}') ~ cluster + age + male", data=d).fit()
        b, ci, pv = m.params["cluster"], m.conf_int().loc["cluster"].values, m.pvalues["cluster"]
        log(f"  OLS 校正年龄+性别: cluster 系数 {b:+.3f} (95%CI {ci[0]:+.3f}~{ci[1]:+.3f}) p={pv:.3f}; 年龄 p={m.pvalues['age']:.3f}")
        rho, pa = stats.spearmanr(d["age"], d[o])
        log(f"  参考: 年龄与 {o} Spearman rho={rho:.3f} p={pa:.3f}; 年龄与聚类 点二列 r={stats.pointbiserialr(d['cluster'], d['age'])[0]:.2f}")
    log("\n注意: 减分率为有界、部分离散的变量(睡眠减分率多取 0/1, 总分减分率有 -3.5 的极端值), 以上以秩检验/排列检验为主。")
    (OUT / "clustering" / "outcome_association.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
