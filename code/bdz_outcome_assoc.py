"""(1) BDZ 使用组 K=6 各簇的 睡眠减分率/总分减分率 差异;
(2) 64 维(16导×4频带 CLR) 与两个减分率的相关/回归/交叉验证预测 (全体、BDZ 使用组、未用组)。
输出: output/clustering_bdz_64d/outcome_assoc.txt, univariate64_*.csv"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmm_bdz_64d as m64  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
D64 = OUT / "clustering_bdz_64d"
OUTCOMES = ["睡眠减分率", "总分减分率"]
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def rank_z(x):
    r = pd.Series(stats.rankdata(x), index=x.index)
    return (r - r.mean()) / r.std()


# ------------------------------------------------------------ 1. K=6 簇间差异
def clusters_test():
    c = pd.read_csv(D64 / "BDZ使用" / "cluster_assignments_K6.csv", dtype={"脑电编号": str})
    J = pd.read_csv(OUT / "eeg_analysis_joined.csv", dtype={"脑电编号": str})
    d = c[["脑电编号", "cluster", "年龄", "性别"]].merge(J[["脑电编号"] + OUTCOMES], on="脑电编号", suffixes=("", "_j"))
    d["age"], d["male"] = pd.to_numeric(d["年龄"]), (d["性别"] == "男").astype(int)
    d = d.dropna(subset=OUTCOMES)
    log("=" * 60 + f"\n1. BDZ 使用组 K=6 (64 维, diag GMM) 簇间差异  n={len(d)}\n" + "=" * 60)
    log("簇大小: " + str(dict(d["cluster"].value_counts().sort_index())) + "; 簇平均年龄: "
        + str(d.groupby("cluster")["age"].mean().round(1).to_dict()))
    for o in OUTCOMES:
        log(f"\n--- {o} ---")
        g = d.groupby("cluster")[o].agg(["count", "mean", "median", "std"]).round(3)
        log(g.to_string())
        groups = [x[o].values for _, x in d.groupby("cluster")]
        H, p = stats.kruskal(*groups)
        eps2 = (H - len(groups) + 1) / (len(d) - len(groups))
        rng = np.random.default_rng(0)
        lab = d["cluster"].values
        obs = H
        perm = np.array([stats.kruskal(*[v[lab == k] for k in np.unique(lab)]).statistic
                         for v in (rng.permutation(d[o].values) for _ in range(2000))])
        log(f"Kruskal-Wallis H={H:.2f} p={p:.3f} (排列 p={(np.sum(perm >= obs) + 1) / 2001:.3f}); ε²={eps2:.3f}")
        yr = rank_z(d[o])
        a = smf.ols("y ~ C(cluster) + age + male", data=d.assign(y=yr)).fit()
        a0 = smf.ols("y ~ age + male", data=d.assign(y=yr)).fit()
        f, pf, _ = a.compare_f_test(a0)
        log(f"校正年龄+性别后簇效应(秩化结局, 嵌套 F 检验): F={f:.2f} p={pf:.3f}")
        pairs = list(combinations(sorted(d["cluster"].unique()), 2))
        pv = [stats.mannwhitneyu(d.loc[d.cluster == i, o], d.loc[d.cluster == j, o]).pvalue for i, j in pairs]
        q = multipletests(pv, method="holm")[1]
        best = sorted(zip(pv, q, pairs))[:3]
        log("两两比较(Mann-Whitney, Holm 校正) 最小 3 对: " + "; ".join(f"{i}vs{j}: p={a_:.3f}, p_holm={b_:.3f}" for a_, b_, (i, j) in best)
            + f"; 共 {len(pairs)} 对, Holm 后 p<0.05: {int((q < .05).sum())}")


# ------------------------------------------------------------ 2. 64 维 vs 减分率
def cv_r2(X, y, n_perm, seed=0):
    rkf = RepeatedKFold(n_splits=10, n_repeats=3, random_state=seed)
    splits = list(rkf.split(X))
    mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 12)))

    def score(yv):
        r2 = []
        for tr, te in splits:
            mdl.fit(X[tr], yv[tr])
            r2.append(1 - np.sum((yv[te] - mdl.predict(X[te])) ** 2) / np.sum((yv[te] - yv[tr].mean()) ** 2))
        return np.mean(r2)
    obs = score(y)
    if n_perm == 0:
        return obs, None, None
    rng = np.random.default_rng(seed)
    null = np.array([score(rng.permutation(y)) for _ in range(n_perm)])
    return obs, null.mean(), (np.sum(null >= obs) + 1) / (n_perm + 1)


def assoc64():
    F = pd.read_csv(D64 / "features64.csv", dtype={"脑电编号": str})
    J = pd.read_csv(OUT / "eeg_analysis_joined.csv", dtype={"脑电编号": str})
    X = m64.design64(F)
    X.insert(0, "脑电编号", F["脑电编号"])
    d = X.merge(J[["脑电编号", "年龄", "性别", "BDZ是否使用"] + OUTCOMES], on="脑电编号").dropna(subset=OUTCOMES + ["BDZ是否使用"])
    d["age"], d["male"] = pd.to_numeric(d["年龄"]), (d["性别"] == "男").astype(int)
    d["bdz"] = pd.to_numeric(d["BDZ是否使用"]).astype(int)
    feats = [c for c in X.columns if c != "脑电编号"]
    samples = {"全体": d, "BDZ使用": d[d.bdz == 1], "未使用BDZ": d[d.bdz == 0]}
    allres = []
    for sname, s in samples.items():
        s = s.reset_index(drop=True)
        cov = ["age", "male"] + (["bdz"] if sname == "全体" else [])
        log("\n" + "=" * 60 + f"\n2. 64 维 CLR vs 减分率 — {sname} (n={len(s)}; 校正 {'+'.join(cov)})\n" + "=" * 60)
        for o in OUTCOMES:
            yr = rank_z(s[o])
            rows = []
            for f in feats:
                x = s[f]
                rho, p = stats.spearmanr(x, s[o])
                M = sm.add_constant(pd.concat([((x - x.mean()) / x.std()).rename("x"), s[cov]], axis=1))
                fit = sm.OLS(yr, M).fit(cov_type="HC3")
                rows.append(dict(sample=sname, outcome=o, feature=f, rho=rho, p=p, beta_adj=fit.params["x"], p_adj=fit.pvalues["x"]))
            r = pd.DataFrame(rows)
            r["q"] = multipletests(r["p"], method="fdr_bh")[1]
            r["q_adj"] = multipletests(r["p_adj"], method="fdr_bh")[1]
            allres.append(r)
            log(f"\n--- {o} ---")
            log(f"单变量(64 个特征): 名义 p<0.05 {int((r.p < .05).sum())} (随机期望约 3.2); FDR q<0.05 {int((r.q < .05).sum())}; "
                f"校正后 名义 p<0.05 {int((r.p_adj < .05).sum())}, FDR q<0.05 {int((r.q_adj < .05).sum())}")
            for _, t in r.sort_values("p").head(5).iterrows():
                log(f"   {t.feature:18s} rho={t.rho:+.3f} p={t.p:.3f} q={t.q:.2f} | 校正 β={t.beta_adj:+.3f} p={t.p_adj:.3f} q={t.q_adj:.2f}")
            y = yr.to_numpy()
            b0 = cv_r2(s[cov].to_numpy(float), y, 0)[0]
            obs, nm, p = cv_r2(s[feats + cov].to_numpy(float), y, 100)
            obs2, nm2, p2 = cv_r2(s[feats].to_numpy(float), y, 100)
            log(f"岭回归 10 折×3 CV R²: 仅协变量 {b0:+.3f}; 64维+协变量 {obs:+.3f} (排列零均值 {nm:+.3f}, p={p:.3f}); 仅64维 {obs2:+.3f} (排列 p={p2:.3f})")
    pd.concat(allres).to_csv(D64 / "univariate64_outcome.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    clusters_test()
    assoc64()
    (D64 / "outcome_assoc.txt").write_text("\n".join(lines), encoding="utf-8")
