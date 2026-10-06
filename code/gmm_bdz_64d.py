"""分支 bdz-64d-gmm: 64 维(16 导联 × δθαβ 相对功率, 1-30Hz) 逐导联 CLR, 不降维, 按 BDZ 分组聚类。
相对功率: 每个导联在 1-30Hz 内 δ/θ/α/β 之和为 1, 再做 CLR(每导联 4 维和为 0, 共线, 秩=48); 标准化后全部 64 维进 GMM。
流程/输出格式复用 gmm_bdz_stratified.py。输出: output/clustering_bdz_64d/
注意: 去掉补充特征的敏感性在本分支不适用(特征本来就只有 64 维), 对应一行结果无意义, 可忽略。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmm_bdz_stratified as b  # noqa: E402
import gmm_single_page as g  # noqa: E402

CH = [c for c in ["FP1-A1", "FP2-A2", "F3-A1", "F4-A2", "C3-A1", "C4-A2", "P3-A1", "P4-A2",
                  "O1-A1", "O2-A2", "F7-A1", "F8-A2", "T3-A1", "T4-A2", "T5-A1", "T6-A2"]]


def channel_features(sig, names):
    names = list(names)
    sig = sig - sig.mean(1, keepdims=True)
    f, p = welch(sig, fs=g.FS, nperseg=200, noverlap=100, detrend="linear")
    feat = {}
    for c in CH:
        bp = np.array([g.band_power(f, p[names.index(c)], *v) for v in g.BANDS.values()])
        for k, v in zip(g.BANDS, bp / bp.sum()):
            feat[f"rel_{k}_{c}"] = v
    return feat


def build():
    pages = g.select_pages()
    base = g.build_features(pages)            # 含 5 个补充特征(此处不进聚类, 仅用于聚类画像)
    rows = []
    for _, r in pages.iterrows():
        z = np.load(g.OUT / "signals" / f"{r['脑电编号']}.npz")
        rows.append(channel_features(z[f"p{r['page']}"].astype(float), z[f"p{r['page']}_ch"]))
    F = pd.concat([base[["脑电编号", "page"] + g.EXTRA], pd.DataFrame(rows)], axis=1)
    return F


def design64(F, drop_theta=False, extras=True):
    keys = [k for k in g.BANDS if not (drop_theta and k == "theta")]
    cols, mats = [], []
    for c in CH:
        x = F[[f"rel_{k}_{c}" for k in keys]].to_numpy()
        mats.append(g.clr(x / x.sum(1, keepdims=True)))
        cols += [f"clr_{k}_{c}" for k in keys]
    return pd.DataFrame(np.hstack(mats), columns=cols, index=F.index)


def standardize_only(X, **kw):
    Z = StandardScaler().fit_transform(X)
    return Z, None, np.ones(Z.shape[1])


def diag_best_k(F):
    J = pd.read_csv(g.OUT / "eeg_analysis_joined.csv", dtype={"脑电编号": str})
    cols = ["脑电编号", "姓名", "年龄", "性别", "BDZ是否使用", "BDZ具体药物", "BDZ剂量"]
    meta = F[["脑电编号"]].merge(J[cols], on="脑电编号", how="left")
    use = pd.to_numeric(meta["BDZ是否使用"])
    for name, flag in (("BDZ使用", 1), ("未使用BDZ", 0)):
        m = (use == flag).to_numpy()
        Fg, mg = F[m].reset_index(drop=True), meta[m].reset_index(drop=True)
        Z, _, _ = standardize_only(design64(Fg))
        bt = pd.read_csv(b.BASE / name / "bic.csv")
        k = int(bt[bt["cov"] == "diag"].sort_values("BIC").iloc[0]["K"])
        b.log(f"\n##### {name}: diag 协方差 BIC 最优 K={k} (全 64 维) #####")
        b.explore_k(b.BASE / name, name, Fg, mg, Z, pd.to_numeric(mg["年龄"]), k, "diag", {}, forced=True)
    (b.BASE / "summary.txt").write_text("\n".join(b.lines), encoding="utf-8")


if __name__ == "__main__":
    F = build()
    b.BASE = g.OUT / "clustering_bdz_64d"
    b.BASE.mkdir(exist_ok=True)
    F.to_csv(b.BASE / "features64.csv", index=False, encoding="utf-8-sig")
    # 让 bdz 脚本使用 64 维特征
    g.reduce = standardize_only
    g.design = design64
    g.REGIONS = {c: [c] for c in CH}          # 聚类画像按单导联显示 α
    orig_read = pd.read_csv
    pd.read_csv = lambda p, *a, **k: orig_read(b.BASE / "features64.csv" if str(p).endswith("features.csv") else p, *a, **k)
    b.main()
    pd.read_csv = orig_read
    diag_best_k(F)
