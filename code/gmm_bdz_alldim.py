"""分支 bdz-alldim-gmm: 不做 PCA, 保留全部 37 维标准化特征, 按 BDZ 使用与否分组, 组内 GMM 聚类。
与 gmm_bdz_stratified.py 的唯一区别: 降维步骤 PCA 被替换为"只标准化"。
注意: 维数(37)相对样本量(未用BDZ组仅56)很大, CLR 各脑区 4 个成分和为 0 (共线), diag 协方差把它们当独立维度会加权;
tied 协方差在 n=56 时有 37*38/2 个参数, 估计很不稳。结果请与 PCA 版本对照。
输出: output/clustering_bdz_alldim/"""
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmm_bdz_stratified as b  # noqa: E402
import gmm_single_page as g  # noqa: E402


def standardize_only(X, **kw):
    Z = StandardScaler().fit_transform(X)
    return Z, None, np.ones(Z.shape[1])  # 与 reduce() 同签名; 累计方差恒为 100%


def diag_best_k():
    """全维时 tied 的 BIC 因 CLR 共线(协方差近奇异)而失真, 额外对 diag 协方差 BIC 最优 K 做聚类"""
    F = g.pd.read_csv(g.OUT / "clustering" / "features.csv", dtype={"脑电编号": str})
    J = g.pd.read_csv(g.OUT / "eeg_analysis_joined.csv", dtype={"脑电编号": str})
    cols = ["脑电编号", "姓名", "年龄", "性别", "BDZ是否使用", "BDZ具体药物", "BDZ剂量"]
    meta = F[["脑电编号"]].merge(J[cols], on="脑电编号", how="left")
    use = g.pd.to_numeric(meta["BDZ是否使用"])
    for name, flag in (("BDZ使用", 1), ("未使用BDZ", 0)):
        m = (use == flag).to_numpy()
        Fg, mg = F[m].reset_index(drop=True), meta[m].reset_index(drop=True)
        Z, _, _ = standardize_only(g.design(Fg))
        bt = g.pd.read_csv(b.BASE / name / "bic.csv")
        k = int(bt[bt["cov"] == "diag"].sort_values("BIC").iloc[0]["K"])
        b.log(f"\n##### {name}: diag 协方差 BIC 最优 K={k} (全 37 维) #####")
        b.explore_k(b.BASE / name, name, Fg, mg, Z, g.pd.to_numeric(mg["年龄"]), k, "diag", {}, forced=True)
    (b.BASE / "summary.txt").write_text("\n".join(b.lines), encoding="utf-8")


if __name__ == "__main__":
    g.reduce = standardize_only
    b.BASE = g.OUT / "clustering_bdz_alldim"
    b.main()
    diag_best_k()
