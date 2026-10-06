"""分支 kmeans-bdz-64d: 64 维(16 导联 × δθαβ 相对功率 CLR, 标准化, 不降维) + K-means。
按 BDZ 使用与否分组(并附全体作参考), 组内聚类; K=2..8 用 轮廓系数/CH/DB/Gap/肘部 选 K;
bootstrap 稳定性 + 去 θ 敏感性 + 与两个减分率的差异检验。输出 output/kmeans_bdz_64d/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmm_bdz_64d as m64  # noqa: E402
import gmm_single_page as g  # noqa: E402

OUT = g.OUT
BASE = OUT / "kmeans_bdz_64d"
OUTCOMES = ["睡眠减分率", "总分减分率"]
KS = range(2, 9)
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def km(Z, k, seed=0, n_init=50):
    return KMeans(k, n_init=n_init, random_state=seed).fit(Z)


def gap_stat(Z, ks, B=20, seed=0):
    """Gap 统计量: 参考分布 = 主成分对齐的均匀盒 (Tibshirani 2001)"""
    rng = np.random.default_rng(seed)
    Zc = Z - Z.mean(0)
    V = PCA().fit(Zc).components_
    Zp = Zc @ V.T
    lo, hi = Zp.min(0), Zp.max(0)
    gaps, sk = [], []
    for k in ks:
        w = np.log(km(Z, k, n_init=10).inertia_)
        wr = np.array([np.log(km((rng.uniform(lo, hi, Zp.shape)) @ V, k, n_init=3).inertia_) for _ in range(B)])
        gaps.append(wr.mean() - w)
        sk.append(wr.std(ddof=0) * np.sqrt(1 + 1 / B))
    gaps, sk = np.array(gaps), np.array(sk)
    ks = list(ks)
    best = next((ks[i] for i in range(len(ks) - 1) if gaps[i] >= gaps[i + 1] - sk[i + 1]), ks[-1])
    return gaps, best


def boot_ari(Z, k, B=100):
    base = km(Z, k).labels_
    rng = np.random.default_rng(0)
    out = []
    for b in range(B):
        i = rng.integers(0, len(Z), len(Z))
        out.append(adjusted_rand_score(base, KMeans(k, n_init=10, random_state=b).fit(Z[i]).predict(Z)))
    return np.array(out)


def outcome_tests(lab, meta, J):
    d = meta[["脑电编号", "年龄", "性别"]].assign(cluster=lab).merge(J[["脑电编号"] + OUTCOMES], on="脑电编号").dropna(subset=OUTCOMES)
    d["age"], d["male"] = pd.to_numeric(d["年龄"]), (d["性别"] == "男").astype(int)
    for o in OUTCOMES:
        grp = [x[o].values for _, x in d.groupby("cluster")]
        H, p = stats.kruskal(*grp)
        rk = pd.Series(stats.rankdata(d[o]), index=d.index)
        rk = (rk - rk.mean()) / rk.std()
        a = smf.ols("y ~ C(cluster) + age + male", data=d.assign(y=rk)).fit()
        a0 = smf.ols("y ~ age + male", data=d.assign(y=rk)).fit()
        f, pf, _ = a.compare_f_test(a0)
        means = d.groupby("cluster")[o].mean().round(2).to_dict()
        log(f"  {o}: 均值 {means}; Kruskal H={H:.2f} p={p:.3f}; 校正年龄性别 F 检验 p={pf:.3f}")


def run(name, F, meta, J):
    d = BASE / name
    d.mkdir(parents=True, exist_ok=True)
    log(f"\n{'=' * 60}\n{name}  n={len(F)}\n{'=' * 60}")
    age = pd.to_numeric(meta["年龄"])
    log(f"年龄 {age.mean():.1f}±{age.std():.1f}, 男 {(meta['性别'] == '男').mean():.0%}")
    Z = StandardScaler().fit_transform(m64.design64(F))
    rows = []
    for k in KS:
        m = km(Z, k)
        rows.append(dict(K=k, inertia=m.inertia_, silhouette=silhouette_score(Z, m.labels_),
                         CH=calinski_harabasz_score(Z, m.labels_), DB=davies_bouldin_score(Z, m.labels_),
                         min_size=np.bincount(m.labels_).min()))
    T = pd.DataFrame(rows)
    gaps, gbest = gap_stat(Z, KS)
    T["gap"] = gaps
    T.to_csv(d / "k_selection.csv", index=False)
    log("K 选择(轮廓系数越大/CH 越大/DB 越小/Gap 越大越好):\n" + T.round(3).to_string(index=False))
    ks_sil = int(T.loc[T["silhouette"].idxmax(), "K"])
    log(f"轮廓系数最优 K={ks_sil}; CH 最优 K={int(T.loc[T['CH'].idxmax(), 'K'])}; DB 最优 K={int(T.loc[T['DB'].idxmin(), 'K'])}; Gap(1-SE 规则) K={gbest}")
    log(f"最大轮廓系数 {T['silhouette'].max():.3f} (<0.25 通常表示几乎没有聚类结构, 0.25-0.5 弱)")
    for k in sorted({2, 3, 6, ks_sil}):
        m = km(Z, k)
        lab = m.labels_
        log(f"\n--- K={k}{' (轮廓系数最优)' if k == ks_sil else ''} ---")
        log(f"聚类大小 {dict(pd.Series(lab).value_counts().sort_index())}")
        ari = boot_ari(Z, k)
        log(f"bootstrap(100) ARI 均值 {ari.mean():.2f}, p5 {np.percentile(ari, 5):.2f}")
        Zs = StandardScaler().fit_transform(m64.design64(F, drop_theta=True))
        log(f"  去掉θ后 ARI={adjusted_rand_score(lab, km(Zs, k).labels_):.2f}; 换种子 ARI: "
            + str(np.round([adjusted_rand_score(lab, km(Z, k, seed=s).labels_) for s in range(1, 6)], 2)))
        ag = pd.DataFrame({"age_mean": age.groupby(lab).mean().round(1), "male_frac": (meta["性别"] == "男").groupby(lab).mean().round(2),
                           "n": pd.Series(lab).value_counts().sort_index()})
        log("年龄/性别:\n" + ag.to_string())
        alpha = pd.DataFrame({f"c{c}": F.loc[lab == c, [f"rel_alpha_{ch}" for ch in m64.CH]].mean().values for c in range(k)},
                             index=m64.CH)
        log("各簇平均相对α(导联):\n" + alpha.round(2).to_string())
        log("补充特征均值:\n" + F[g.EXTRA].groupby(lab).mean().round(2).to_string())
        if name != "未使用BDZ":
            log("聚类 × BDZ 具体药物:\n" + pd.crosstab(lab, meta["BDZ具体药物"].fillna("无")).to_string())
        log("与减分率:")
        outcome_tests(lab, meta, J)
        meta.assign(cluster=lab, silhouette=__import__("sklearn.metrics", fromlist=["x"]).silhouette_samples(Z, lab)).to_csv(
            d / f"cluster_assignments_K{k}.csv", index=False, encoding="utf-8-sig")
        F.assign(cluster=lab).groupby("cluster").mean(numeric_only=True).T.to_csv(d / f"cluster_profile_K{k}.csv", encoding="utf-8-sig")


def main():
    BASE.mkdir(exist_ok=True)
    F = pd.read_csv(OUT / "clustering_bdz_64d" / "features64.csv", dtype={"脑电编号": str})
    J = pd.read_csv(OUT / "eeg_analysis_joined.csv", dtype={"脑电编号": str})
    cols = ["脑电编号", "姓名", "年龄", "性别", "BDZ是否使用", "BDZ具体药物", "BDZ剂量"]
    meta = F[["脑电编号"]].merge(J[cols], on="脑电编号", how="left")
    use = pd.to_numeric(meta["BDZ是否使用"])
    log(f"单页样本 {len(F)}; BDZ: 用 {int((use == 1).sum())}, 未用 {int((use == 0).sum())}, 未匹配 {int(use.isna().sum())}(分组分析中排除)")
    for name, mask in (("BDZ使用", use == 1), ("未使用BDZ", use == 0), ("全体(参考)", use.notna())):
        m = mask.to_numpy()
        run(name, F[m].reset_index(drop=True), meta[m].reset_index(drop=True), J)
    (BASE / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
