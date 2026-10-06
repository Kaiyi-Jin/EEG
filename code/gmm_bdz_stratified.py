"""分支 bdz-stratified-gmm: 按分析数据的【BDZ是否使用】分组, 组内分别做单页 GMM 聚类。
特征、PCA、GMM 设置与 gmm_single_page.py 完全一致(CLR 区域相对功率 + 补充特征 -> 标准化 -> PCA -> diag/tied GMM)。
输出: output/clustering_bdz/{BDZ使用,未使用BDZ}/ 与 output/clustering_bdz/summary.txt"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmm_single_page as g  # noqa: E402

OUT = g.OUT
BASE = OUT / "clustering_bdz"
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def explore_k(d, name, F, meta, Z, age, k, cov, out, forced):
    log(f"\n--- K={k} ({'探索性强制' if forced else 'BIC 选定'}, 协方差 {cov}) ---")
    gm = g.fit_gmm(Z, k, cov)
    lab = gm.predict(Z)
    prob = gm.predict_proba(Z)
    res = meta.assign(cluster=lab, max_prob=prob.max(1))
    for i in range(Z.shape[1]):
        res[f"PC{i + 1}"] = Z[:, i]
    res.to_csv(d / f"cluster_assignments_K{k}.csv", index=False, encoding="utf-8-sig")
    log(f"聚类大小 {dict(pd.Series(lab).value_counts().sort_index())}; 平均最大后验概率 {prob.max(1).mean():.2f}")
    ari = g.bootstrap_stability(Z, k, cov)
    log(f"bootstrap(100) ARI 均值 {ari.mean():.2f}, p5 {np.percentile(ari, 5):.2f}")
    out[f"K{k}_boot_ARI"] = ari.mean()
    for nm, kw in {"去掉θ": dict(drop_theta=True), "去掉补充特征": dict(extras=False)}.items():
        Zs, _, _ = g.reduce(g.design(F, **kw))
        log("  敏感性 " + nm + ": " + ", ".join(f"{c2}: ARI={adjusted_rand_score(lab, g.fit_gmm(Zs, k, c2).predict(Zs)):.2f}" for c2 in ("diag", "tied")))
    log("  换种子 ARI: " + str(np.round([adjusted_rand_score(lab, g.fit_gmm(Z, k, cov, seed=s).predict(Z)) for s in range(1, 6)], 2)))
    reg = pd.DataFrame({f"cluster{c}": F.loc[lab == c, [f"rel_alpha_{r}" for r in g.REGIONS]].mean().values
                        for c in range(k)}, index=list(g.REGIONS))
    log("各聚类平均相对 α(脑区):\n" + reg.round(2).to_string())
    log("补充特征均值:\n" + F[g.EXTRA].groupby(lab).mean().round(2).to_string())
    ag = pd.DataFrame({"age_mean": age.groupby(lab).mean(), "male_frac": (meta["性别"] == "男").groupby(lab).mean(),
                       "n": pd.Series(lab).value_counts().sort_index()}).round(2)
    log("年龄/性别:\n" + ag.to_string())
    log("聚类 × BDZ 具体药物:\n" + pd.crosstab(lab, meta["BDZ具体药物"].fillna("无")).to_string())
    F.assign(cluster=lab).groupby("cluster").mean(numeric_only=True).T.to_csv(d / f"cluster_profile_K{k}.csv", encoding="utf-8-sig")


def run_group(name, F, meta):
    d = BASE / name
    d.mkdir(parents=True, exist_ok=True)
    log(f"\n{'=' * 60}\n{name}  n={len(F)}\n{'=' * 60}")
    age = pd.to_numeric(meta["年龄"])
    log(f"年龄 {age.mean():.1f}±{age.std():.1f}, 男 {(meta['性别'] == '男').mean():.0%}")
    X = g.design(F)
    Z, pca, cum = g.reduce(X)
    log(f"{X.shape[1]} 维 -> 聚类输入 {Z.shape[1]} 维" + ("" if pca is None else f" (PCA, 累计方差 {cum[Z.shape[1] - 1]:.1%})"))
    g.K_RANGE = range(1, 7 if len(Z) < 100 else 9)
    bt = g.bic_table(Z)
    bt.to_csv(d / "bic.csv", index=False)
    log("BIC:\n" + bt.pivot(index="K", columns="cov", values="BIC").round(1).to_string())
    best = bt.loc[bt.groupby("cov")["BIC"].idxmin()].set_index("cov")["K"].to_dict()
    cov = bt.loc[bt["BIC"].idxmin(), "cov"]
    k = int(best[cov])
    b = bt[bt["cov"] == cov].set_index("K")["BIC"]
    log(f"各协方差最优 K: {best} -> 选用 {cov} K={k}; 相对 K=1 的 ΔBIC = {b[1] - b[k]:.1f}"
        + ("; 相对次优 K 的 ΔBIC = %.1f" % (b.drop(k).min() - b[k]) if len(b) > 1 else ""))
    out = {"n": len(Z), "K_BIC": k, "cov": cov}
    if k == 1:
        log("BIC 最优 K=1: 组内没有统计证据支持多聚类; 以下 K=2/3 为探索性(强制)结果, 请结合稳定性谨慎解读")
    for kk in sorted({k, 2, 3} - {1}):
        explore_k(d, name, F, meta, Z, age, kk, cov, out, forced=(kk != k))
    if k == 1:
        F.assign(cluster=0).groupby("cluster").mean(numeric_only=True).T.to_csv(d / "cluster_profile_K1.csv", encoding="utf-8-sig")
    pd.Series(out).to_csv(d / "summary.csv", header=False)
    return out


def main():
    BASE.mkdir(exist_ok=True)
    F = pd.read_csv(OUT / "clustering" / "features.csv", dtype={"脑电编号": str})
    J = pd.read_csv(OUT / "eeg_analysis_joined.csv", dtype={"脑电编号": str})
    cols = ["脑电编号", "姓名", "年龄", "性别", "BDZ是否使用", "BDZ具体药物", "BDZ剂量"]
    meta = F[["脑电编号"]].merge(J[cols], on="脑电编号", how="left")
    log(f"单页样本 {len(F)}; BDZ 是否使用: {dict(meta['BDZ是否使用'].value_counts(dropna=False))} (NaN=未匹配到分析数据, 排除)")
    use = pd.to_numeric(meta["BDZ是否使用"])
    # 组间 EEG 特征差异(简要): 全脑相对功率
    for k in g.BANDS:
        v = F[[f"rel_{k}_{r}" for r in g.REGIONS]].mean(1)
        p = stats.mannwhitneyu(v[use == 1], v[use == 0]).pvalue
        log(f"  全脑相对 {k}: 用BDZ {v[use == 1].mean():.3f} vs 未用 {v[use == 0].mean():.3f}  (MWU p={p:.3f})")
    summ = {}
    for name, flag in (("BDZ使用", 1), ("未使用BDZ", 0)):
        m = (use == flag).to_numpy()
        summ[name] = run_group(name, F[m].reset_index(drop=True), meta[m].reset_index(drop=True))
    (BASE / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
