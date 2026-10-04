"""连续 EEG 特征 vs 睡眠减分率 / 总分减分率 (相关 + 回归 + 交叉验证预测)。
输出: output/clustering/continuous_outcome.txt, continuous_univariate.csv, continuous_volcano.png"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gmm_single_page import BANDS, REGIONS, design  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
OUTCOMES = ["睡眠减分率", "总分减分率"]
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def load():
    F = pd.read_csv(OUT / "clustering" / "features.csv", dtype={"脑电编号": str})
    J = pd.read_csv(OUT / "eeg_analysis_joined.csv", dtype={"脑电编号": str})
    X = design(F)
    for k in BANDS:  # 全脑平均相对功率(各脑区均值, 作为易解释的总体指标)
        X[f"global_rel_{k}"] = F[[f"rel_{k}_{r}" for r in REGIONS]].mean(1)
    X.insert(0, "脑电编号", F["脑电编号"])
    d = X.merge(J[["脑电编号", "年龄", "性别"] + OUTCOMES], on="脑电编号")
    d["age"] = pd.to_numeric(d["年龄"])
    d["male"] = (d["性别"] == "男").astype(int)
    d = d.dropna(subset=OUTCOMES).reset_index(drop=True)
    feats = [c for c in X.columns if c != "脑电编号"]
    return d, feats


def rank_z(x):
    return pd.Series(stats.rankdata(x), index=x.index).pipe(lambda r: (r - r.mean()) / r.std())


def univariate(d, feats, o):
    rows = []
    yr = rank_z(d[o])
    for f in feats:
        x = d[f]
        rho, p = stats.spearmanr(x, d[o])
        # 校正年龄+性别: 秩化结局 ~ z(特征) + 年龄 + 性别, HC3 稳健标准误
        M = sm.add_constant(pd.DataFrame({"x": (x - x.mean()) / x.std(), "age": d["age"], "male": d["male"]}))
        m = sm.OLS(yr, M).fit(cov_type="HC3")
        rows.append(dict(feature=f, rho=rho, p=p, beta_adj=m.params["x"], p_adj=m.pvalues["x"]))
    r = pd.DataFrame(rows)
    r["q"] = multipletests(r["p"], method="fdr_bh")[1]
    r["q_adj"] = multipletests(r["p_adj"], method="fdr_bh")[1]
    r["outcome"] = o
    return r


def cv_predict(d, feats, o, extra_cols, tag, n_perm=200):
    """岭回归重复 10 折 CV 的 R² (秩化结局), 并用排列检验给 p"""
    y = rank_z(d[o]).to_numpy()
    X = d[feats + extra_cols].to_numpy()
    rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30)))

    def score(yv):
        sc = []
        for tr, te in rkf.split(X):
            mdl.fit(X[tr], yv[tr])
            sc.append(mdl.predict(X[te]))
        # 汇总各折预测的 R² (相对训练集均值基线)
        r2 = []
        for (tr, te), pr in zip(rkf.split(X), sc):
            r2.append(1 - np.sum((yv[te] - pr) ** 2) / np.sum((yv[te] - yv[tr].mean()) ** 2))
        return np.mean(r2)

    obs = score(y)
    rng = np.random.default_rng(0)
    null = np.array([score(rng.permutation(y)) for _ in range(n_perm)])
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    log(f"  {tag:34s} CV R²={obs:+.3f}  (排列零分布均值 {null.mean():+.3f}, p={p:.3f})")
    return obs


def main():
    d, feats = load()
    log(f"样本 n={len(d)}, 特征 {len(feats)} 个(CLR 区域相对功率 + 补充特征 + 全脑相对功率)")
    log(f"年龄 {d['age'].min():.0f}-{d['age'].max():.0f}, 男 {d['male'].mean():.0%}")
    res = []
    for o in OUTCOMES:
        log(f"\n{'=' * 50}\n{o}\n{'=' * 50}")
        r = univariate(d, feats, o)
        res.append(r)
        rho_a, p_a = stats.spearmanr(d["age"], d[o])
        log(f"  年龄 vs {o}: rho={rho_a:.3f} p={p_a:.3f}")
        log(f"  单变量: 名义 p<0.05 的特征 {int((r.p < .05).sum())}/{len(r)} (期望约 {0.05 * len(r):.1f}); "
            f"FDR q<0.05: {int((r.q < .05).sum())}; 校正年龄性别后 名义 p<0.05: {int((r.p_adj < .05).sum())}, FDR q<0.05: {int((r.q_adj < .05).sum())}")
        top = r.reindex(r["p"].sort_values().index).head(8)
        log("  Spearman 最强的 8 个特征(rho / p / q | 校正后 β(秩化sd) / p_adj / q_adj):")
        for _, t in top.iterrows():
            log(f"    {t.feature:24s} {t.rho:+.3f} {t.p:.3f} {t.q:.2f} | {t.beta_adj:+.3f} {t.p_adj:.3f} {t.q_adj:.2f}")
        log("  预先指定的 6 个特征(不做搜索):")
        for f in ["global_rel_alpha", "log_theta_beta", "log_occ_front_alpha", "paf", "ap_exp_all", "ap_offset_all"]:
            t = r[r.feature == f].iloc[0]
            log(f"    {f:22s} rho={t.rho:+.3f} p={t.p:.3f} | 校正后 β={t.beta_adj:+.3f} p={t.p_adj:.3f}")
        # 稳健性: 去掉总分减分率极端值
        if o == "总分减分率":
            keep = d[o] > -2
            n_out = int((~keep).sum())
            log(f"  [稳健性] 去掉 {n_out} 个极端值(<-2)后最小 p={min(stats.spearmanr(d.loc[keep, f], d.loc[keep, o])[1] for f in feats):.3f}"
                " (Spearman 本身对极端值不敏感)")
        log("  多变量(岭回归, 重复10折CV, 结局秩化):")
        cv_predict(d, [], o, ["age", "male"], "仅 年龄+性别")
        cv_predict(d, feats, o, [], "仅 EEG 特征(全部)")
        cv_predict(d, feats, o, ["age", "male"], "EEG 特征 + 年龄+性别")
        pcs = [c for c in feats if c in ("paf", "ap_exp_all", "ap_offset_all", "log_theta_beta", "log_occ_front_alpha")]
        cv_predict(d, pcs, o, ["age", "male"], "5 个补充特征 + 年龄+性别")
    R = pd.concat(res)
    R.to_csv(OUT / "clustering" / "continuous_univariate.csv", index=False, encoding="utf-8-sig")
    log("\n解读: CV R²≈0 或为负表示 EEG 特征对该结局没有样本外预测力; 排列 p 用于判断是否优于随机。")
    (OUT / "clustering" / "continuous_outcome.txt").write_text("\n".join(lines), encoding="utf-8")
    plot(R)


def plot(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for a, (o, g) in zip(ax, R.groupby("outcome")):
        a.scatter(g["rho"], -np.log10(g["p"]), s=18)
        a.axhline(-np.log10(0.05), ls="--", c="gray")
        for _, t in g.sort_values("p").head(4).iterrows():
            a.annotate(t.feature, (t.rho, -np.log10(t.p)), fontsize=7)
        a.set_title(o); a.set_xlabel("Spearman rho"); a.set_ylabel("-log10 p")
    fig.tight_layout()
    fig.savefig(OUT / "clustering" / "continuous_volcano.png", dpi=110)


if __name__ == "__main__":
    main()
