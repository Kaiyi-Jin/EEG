"""分支 gmm-single-page: 单页(每人 1 个 8s 静息片段) 特征提取 + GMM 聚类。

特征:
  * 相对功率(1-30Hz 内 δ/θ/α/β 之和为 1) 按 8 个脑区合并 -> CLR 变换
  * 补充: α 峰频率(重心法, 0.5Hz 分辨率仅作参考)、1/f 斜率与截距(specparam, 2-30Hz)、
          log(θ/β)、log(枕区α/额区α)
  * 标准化 -> PCA(5-10 个主成分) -> GMM(diag / tied, 不用 full), BIC 选 K
  * 敏感性: 去掉 θ、去掉补充特征、不同协方差、bootstrap 稳定性(ARI)
输出: output/clustering/
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
CLU = OUT / "clustering"
FS = 100
FMIN, FMAX = 1.0, 30.0           # 只用 1-30Hz
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
# 8 个脑区(左右 × 额/中央/颞/顶枕); T5/T6 归顶枕
REGIONS = {
    "F_L": ["FP1-A1", "F3-A1", "F7-A1"], "F_R": ["FP2-A2", "F4-A2", "F8-A2"],
    "C_L": ["C3-A1"], "C_R": ["C4-A2"],
    "T_L": ["T3-A1"], "T_R": ["T4-A2"],
    "PO_L": ["P3-A1", "O1-A1", "T5-A1"], "PO_R": ["P4-A2", "O2-A2", "T6-A2"],
}
K_RANGE = range(1, 9)
SEED = 0


# ---------------------------------------------------------------- 选页
def select_pages():
    """每人 1 页: 背景 + A1/A2 参考导联 + 有处理长度 + 有灵敏度 + 无候选坏导联; 多页取起点最早的一页"""
    P = pd.read_csv(OUT / "qc_pages.csv", dtype={"脑电编号": str})
    C = pd.read_csv(OUT / "qc_channels.csv", dtype={"脑电编号": str})
    rep = pd.read_csv(OUT / "eeg_reports.csv", dtype=str)
    no_len = {json.loads(p.read_text(encoding="utf-8"))["脑电编号"]
              for p in (OUT / "waveform_info").glob("*.json")
              if not json.loads(p.read_text(encoding="utf-8"))["处理长度"]}
    ref = set(map(tuple, C[C["ch"] == "FP1-A1"][["脑电编号", "page"]].values))
    P["ref"] = [(a, b) in ref for a, b in zip(P["脑电编号"], P["page"])]
    # 候选坏导联: 平线/削顶/按导联稳健 z(log std)>5
    ls = np.log10(C["std_uV"])
    med = ls.groupby(C["ch"]).transform("median")
    mad = (ls - med).abs().groupby(C["ch"]).transform("median") * 1.4826
    C["bad"] = (C["std_uV"] < 1) | (C["flat_frac"] > 0.5) | (C["clip_frac"] > 0.01) | (((ls - med) / mad).abs() > 5)
    badp = set(map(tuple, C[C["bad"]][["脑电编号", "page"]].values))
    P["bad"] = [(a, b) in badp for a, b in zip(P["脑电编号"], P["page"])]
    log = {"总页数": len(P), "总人数": P["脑电编号"].nunique()}
    el = P[(P["stage"] == "背景") & P["ref"] & ~P["sens_missing"] & ~P["脑电编号"].isin(no_len)]
    log["背景+参考导联+有灵敏度+有处理长度"] = (len(el), el["脑电编号"].nunique())
    el2 = el[~el["bad"]]
    log["再去掉含候选坏导联的页"] = (len(el2), el2["脑电编号"].nunique())
    one = el2.sort_values(["脑电编号", "起点秒", "page"]).drop_duplicates("脑电编号", keep="first")
    log["最终(每人1页)"] = len(one)
    print(log)
    return one.merge(rep[["脑电编号", "姓名", "年龄", "性别", "临床诊断", "药物", "检查日期"]], on="脑电编号", how="left")


# ---------------------------------------------------------------- 特征
def band_power(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return np.trapezoid(p[..., m], f[m], axis=-1)


def clr(x):
    lx = np.log(x)
    return lx - lx.mean(-1, keepdims=True)


def page_features(sig, names):
    names = list(names)
    sig = sig - sig.mean(1, keepdims=True)
    f, p = welch(sig, fs=FS, nperseg=200, noverlap=100, detrend="linear")   # 2s 窗, 0.5Hz
    idx = {n: i for i, n in enumerate(names)}
    feat, comp = {}, {}
    reg_psd = {}
    for r, chs in REGIONS.items():
        reg_psd[r] = p[[idx[c] for c in chs]].mean(0)       # 区内通道平均 PSD
        b = np.array([band_power(f, reg_psd[r], *v) for v in BANDS.values()])
        comp[r] = b / b.sum()                                # 1-30Hz 内相对功率, 和为 1
    for r in REGIONS:
        for k, v in zip(BANDS, comp[r]):
            feat[f"rel_{k}_{r}"] = v
    # 补充特征
    occ = (reg_psd["PO_L"] + reg_psd["PO_R"]) / 2
    fro = (reg_psd["F_L"] + reg_psd["F_R"]) / 2
    allm = p.mean(0)
    sel = (f >= 7) & (f <= 13)
    feat["paf"] = float((f[sel] * occ[sel]).sum() / occ[sel].sum())     # α 重心频率 (0.5Hz 分辨率, 仅作参考)
    feat.update(aperiodic(f, allm, "all"))
    feat.update(aperiodic(f, occ, "occ"))
    bt = lambda q: band_power(f, q, 4, 8) / band_power(f, q, 13, 30)
    feat["log_theta_beta"] = float(np.log(bt(allm)))
    feat["log_occ_front_alpha"] = float(np.log(band_power(f, occ, 8, 13) / band_power(f, fro, 8, 13)))
    return feat


def aperiodic(f, psd, tag):
    from specparam import SpectralModel
    m = SpectralModel(peak_width_limits=(1, 8), max_n_peaks=4, verbose=False)
    try:
        m.fit(f[f > 0], psd[f > 0], [2, 30])
        off, expo = m.results.params.aperiodic.params
        r2 = m.results.metrics.results["gof_rsquared"]
    except Exception:
        off, expo, r2 = np.nan, np.nan, np.nan
    return {f"ap_offset_{tag}": off, f"ap_exp_{tag}": expo, f"ap_r2_{tag}": r2}


def build_features(pages):
    rows = []
    for _, r in pages.iterrows():
        z = np.load(OUT / "signals" / f"{r['脑电编号']}.npz")
        feats = page_features(z[f"p{r['page']}"].astype(float), z[f"p{r['page']}_ch"])
        rows.append({"脑电编号": r["脑电编号"], "page": r["page"], **feats})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 矩阵
def clr_block(F, drop_theta=False):
    keys = [k for k in BANDS if not (drop_theta and k == "theta")]
    cols, mats = [], []
    for r in REGIONS:
        x = F[[f"rel_{k}_{r}" for k in keys]].to_numpy()
        x = x / x.sum(1, keepdims=True)         # 去 θ 后重新闭合
        mats.append(clr(x))
        cols += [f"clr_{k}_{r}" for k in keys]
    return pd.DataFrame(np.hstack(mats), columns=cols, index=F.index)


EXTRA = ["paf", "ap_exp_all", "ap_offset_all", "log_theta_beta", "log_occ_front_alpha"]


def design(F, drop_theta=False, extras=True):
    X = clr_block(F, drop_theta)
    if extras:
        E = F[[c for c in EXTRA if not (drop_theta and c == "log_theta_beta")]]
        X = pd.concat([X, E], axis=1)
    return X


def reduce(X, var_target=0.8, lo=5, hi=10):
    Z = StandardScaler().fit_transform(X)
    full = PCA().fit(Z)
    cum = np.cumsum(full.explained_variance_ratio_)
    n = int(np.clip(np.searchsorted(cum, var_target) + 1, lo, hi))
    pca = PCA(n_components=n, random_state=SEED)
    return pca.fit_transform(Z), pca, cum


def fit_gmm(Z, k, cov, seed=SEED):
    return GaussianMixture(k, covariance_type=cov, n_init=20, reg_covar=1e-4, random_state=seed).fit(Z)


def bic_table(Z):
    rows = []
    for cov in ("diag", "tied"):
        for k in K_RANGE:
            g = fit_gmm(Z, k, cov)
            rows.append(dict(cov=cov, K=k, BIC=g.bic(Z), AIC=g.aic(Z), converged=g.converged_))
    return pd.DataFrame(rows)


def bootstrap_stability(Z, k, cov, B=100):
    """bootstrap 重拟合 GMM, 在全体样本上预测, 与原聚类比 ARI"""
    base = fit_gmm(Z, k, cov).predict(Z)
    rng = np.random.default_rng(SEED)
    ari = []
    for b in range(B):
        i = rng.integers(0, len(Z), len(Z))
        g = GaussianMixture(k, covariance_type=cov, n_init=5, reg_covar=1e-4, random_state=b).fit(Z[i])
        ari.append(adjusted_rand_score(base, g.predict(Z)))
    return np.array(ari)


# ---------------------------------------------------------------- 主流程
def main():
    CLU.mkdir(exist_ok=True)
    pages = select_pages()
    F = build_features(pages)
    meta = pages[["脑电编号", "姓名", "年龄", "性别", "临床诊断", "药物", "检查日期"]].reset_index(drop=True)
    F.to_csv(CLU / "features.csv", index=False, encoding="utf-8-sig")
    print("特征矩阵", F.shape, " specparam R² 中位数 all/occ:", F["ap_r2_all"].median().round(3), F["ap_r2_occ"].median().round(3))
    bad_fit = F["ap_exp_all"].isna().sum()
    print("specparam 拟合失败:", bad_fit)

    # 主分析
    X = design(F)
    ok = X.notna().all(1)
    X, F, meta = X[ok].reset_index(drop=True), F[ok].reset_index(drop=True), meta[ok].reset_index(drop=True)
    Z, pca, cum = reduce(X)
    print(f"主分析: {X.shape[1]} 维 -> PCA {Z.shape[1]} 个主成分 (累计方差 {cum[Z.shape[1] - 1]:.1%})")
    bt = bic_table(Z)
    bt.to_csv(CLU / "bic.csv", index=False)
    print(bt.pivot(index="K", columns="cov", values="BIC").round(1))
    best = bt.loc[bt.groupby("cov")["BIC"].idxmin()].set_index("cov")["K"].to_dict()
    cov = bt.loc[bt["BIC"].idxmin(), "cov"]
    k = int(best[cov])
    print("各协方差下 BIC 最优 K:", best, "-> 选用", cov, k)

    g = fit_gmm(Z, k, cov)
    lab = g.predict(Z)
    prob = g.predict_proba(Z)
    ent = -(prob * np.log(prob + 1e-12)).sum(1) / np.log(max(k, 2))
    res = meta.assign(cluster=lab, max_prob=prob.max(1), entropy=ent)
    for i in range(Z.shape[1]):
        res[f"PC{i + 1}"] = Z[:, i]
    res.to_csv(CLU / "cluster_assignments.csv", index=False, encoding="utf-8-sig")
    print("聚类大小:", dict(pd.Series(lab).value_counts().sort_index()), "; 平均最大后验概率", prob.max(1).mean().round(3))

    # 稳定性与敏感性
    summ = {"K": k, "cov": cov, "n": len(Z), "n_pc": Z.shape[1]}
    ari = bootstrap_stability(Z, k, cov)
    summ["bootstrap_ARI_mean"], summ["bootstrap_ARI_p5"] = ari.mean(), np.percentile(ari, 5)
    print(f"bootstrap(100) ARI 均值 {ari.mean():.2f}, p5 {np.percentile(ari, 5):.2f}")
    sens = {}
    for name, kw in {"去掉θ": dict(drop_theta=True), "去掉补充特征": dict(extras=False),
                     "仅补充特征(无CLR)": None}.items():
        if kw is None:
            Xs = F[EXTRA]
        else:
            Xs = design(F, **kw)
        Zs, _, _ = reduce(Xs, lo=min(5, Xs.shape[1]), hi=10)
        for c2 in ("diag", "tied"):
            gs = fit_gmm(Zs, k, c2)
            sens[f"{name}|{c2}"] = adjusted_rand_score(lab, gs.predict(Zs))
    for c2 in ("diag", "tied"):
        if c2 != cov:
            sens[f"主特征|{c2}"] = adjusted_rand_score(lab, fit_gmm(Z, k, c2).predict(Z))
    print("敏感性(与主分析聚类的 ARI):")
    for a, b in sens.items():
        print(f"  {a:22s} {b:.2f}")
    # 样本量种子敏感: 换随机种子
    seeds = [adjusted_rand_score(lab, fit_gmm(Z, k, cov, seed=s).predict(Z)) for s in range(1, 6)]
    print("不同随机种子 ARI:", np.round(seeds, 2))
    pd.Series({**summ, **{f"sens_{a}": b for a, b in sens.items()}}).to_csv(CLU / "stability.csv", header=False)

    # 聚类画像
    prof_cols = [c for c in F.columns if c.startswith("rel_")] + EXTRA
    prof = F[prof_cols].groupby(lab).mean().T
    prof.columns = [f"cluster{c}" for c in prof.columns]
    prof.to_csv(CLU / "cluster_profile.csv", encoding="utf-8-sig")
    reg_alpha = pd.DataFrame({f"cluster{c}": F.loc[lab == c, [f"rel_alpha_{r}" for r in REGIONS]].mean().values
                              for c in range(k)}, index=list(REGIONS))
    print("\n各聚类平均相对 α 功率(脑区):\n", reg_alpha.round(2))
    print("各聚类补充特征均值:\n", F[EXTRA].groupby(lab).mean().round(2))
    age = pd.to_numeric(meta["年龄"])
    print("年龄 均值±sd / 男比例 / 样本数:")
    print(pd.DataFrame({"age_mean": age.groupby(lab).mean(), "age_sd": age.groupby(lab).std(),
                        "male_frac": (meta["性别"] == "男").groupby(lab).mean(), "n": pd.Series(lab).value_counts().sort_index()}).round(2))
    pcs = pd.DataFrame(pca.components_.T[:, :3], index=X.columns, columns=["PC1", "PC2", "PC3"])
    print("PC1-3 载荷最大的特征:", {c: pcs[c].abs().sort_values(ascending=False).head(4).index.tolist() for c in pcs})
    plots(Z, lab, bt, cum, reg_alpha, age)


def plots(Z, lab, bt, cum, reg_alpha, age):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for c, g in bt.groupby("cov"):
        ax[0].plot(g["K"], g["BIC"], marker="o", label=c)
    ax[0].set_xlabel("K"); ax[0].set_ylabel("BIC"); ax[0].legend()
    ax[1].plot(range(1, len(cum) + 1), cum, marker="o"); ax[1].set_xlabel("n PCs"); ax[1].set_ylabel("cum. var")
    sc = ax[2].scatter(Z[:, 0], Z[:, 1], c=lab, cmap="tab10", s=18)
    ax[2].set_xlabel("PC1"); ax[2].set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(CLU / "gmm_overview.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(reg_alpha.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(reg_alpha.shape[1])); ax.set_xticklabels(reg_alpha.columns)
    ax.set_yticks(range(len(reg_alpha))); ax.set_yticklabels(reg_alpha.index)
    fig.colorbar(im, label="relative alpha"); fig.tight_layout()
    fig.savefig(CLU / "cluster_alpha_by_region.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
