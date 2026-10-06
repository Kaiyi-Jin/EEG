"""分支 predict-64d-kfold: 64 维(16导×δθαβ 相对功率 CLR) -> 睡眠减分率 / 总分减分率 预测模型。
按 BDZ 使用与否分组(并附全体作参考), 重复 K 折交叉验证 (10 折 × 3 次重复, 训练折内标准化 + 内层 CV 调参, 无信息泄漏):
  回归(结局秩化): Ridge / ElasticNet / PLS / RandomForest / SVR, 另有 仅协变量(年龄+性别) 基线 与 64维+协变量
  分类: 睡眠改善(减分率>0) 与 总分有效(减分率>=0.5): L2 逻辑回归 / RandomForest, 指标 AUC
  每个模型做排列检验 (打乱结局重复整套 CV)。输出 output/predict_64d/"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmm_bdz_64d as m64  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
D = OUT / "predict_64d"
OUTCOMES = ["睡眠减分率", "总分减分率"]
N_SPLITS, N_REPEATS, SEED = 10, 2, 0
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def rank_z(y):
    r = stats.rankdata(y)
    return (r - r.mean()) / r.std()


# ------------------------------------------------------------------ 模型
def reg_models():
    alphas = np.logspace(-1, 4, 20)
    return {
        "Ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
        "ElasticNet": lambda: make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.5], alphas=np.logspace(-2.5, 0.5, 10), cv=3, max_iter=2000)),
        "PLS": lambda: GridSearchCV(make_pipeline(StandardScaler(), PLSRegression()), {"plsregression__n_components": [1, 2, 3, 5]},
                                    cv=3, scoring="neg_mean_squared_error"),
        "RandomForest": lambda: make_pipeline(StandardScaler(), RandomForestRegressor(100, min_samples_leaf=5, max_features=0.3, random_state=0, n_jobs=1)),
        "SVR(RBF)": lambda: GridSearchCV(make_pipeline(StandardScaler(), SVR()), {"svr__C": [.1, 1, 10], "svr__gamma": ["scale", 0.003]},
                                         cv=3, scoring="neg_mean_squared_error"),
    }


def clf_models():
    return {
        "Logistic(L2)": lambda: make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-4, 0, 6), cv=3, max_iter=2000, scoring="roc_auc")),
        "RandomForest": lambda: make_pipeline(StandardScaler(), RandomForestClassifier(100, min_samples_leaf=5, max_features=0.3, random_state=0, n_jobs=1)),
    }


# ------------------------------------------------------------------ CV
def cv_reg(make, X, y, splits):
    r2, oof = [], np.zeros((N_REPEATS, len(y)))
    for i, (tr, te) in enumerate(splits):
        m = make().fit(X[tr], y[tr])
        p = np.ravel(m.predict(X[te]))
        oof[i // N_SPLITS, te] = p
        r2.append(1 - np.sum((y[te] - p) ** 2) / np.sum((y[te] - y[tr].mean()) ** 2))
    return float(np.mean(r2)), oof.mean(0)


def cv_clf(make, X, y, splits):
    auc, oof = [], np.zeros((N_REPEATS, len(y)))
    for i, (tr, te) in enumerate(splits):
        m = make().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        oof[i // N_SPLITS, te] = p
        auc.append(roc_auc_score(y[te], p))
    return float(np.mean(auc)), oof.mean(0)


def perm_p(fn, make, X, y, splits, obs, n_perm, stratified=False):
    def one(s):
        rng = np.random.default_rng(1000 + s)
        yp = rng.permutation(y)
        sp = splits
        if stratified:  # 分类: 打乱后的标签需重新做分层划分, 否则可能出现单一类别的测试折
            sp = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(X, yp))
        return fn(make, X, yp, sp)[0]
    null = np.array(Parallel(n_jobs=8)(delayed(one)(s) for s in range(n_perm)))
    return null.mean(), (np.sum(null >= obs) + 1) / (n_perm + 1)


def evaluate(sname, s, feats, cov, rows):
    s = s.reset_index(drop=True)
    X64 = s[feats].to_numpy(float)
    C = s[cov].to_numpy(float)
    XC = np.hstack([X64, C])
    log("\n" + "=" * 70 + f"\n{sname}  n={len(s)}  (协变量: {'+'.join(cov)})\n" + "=" * 70)
    for o in OUTCOMES:
        y = rank_z(s[o].to_numpy())
        splits = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(X64))
        log(f"\n--- 回归: {o} (秩化; 10 折×2 重复; 嵌套调参) ---")
        log(f"{'模型':28s} {'CV R²':>8s} {'排列零均值':>10s} {'排列p':>7s}")
        specs = [("仅协变量(基线, Ridge)", C, reg_models()["Ridge"], 0)]
        for nm, mk in reg_models().items():
            n_perm = {"Ridge": 100, "PLS": 50}.get(nm, 25)
            specs.append((f"64维 {nm}", X64, mk, n_perm))
        for nm in ("Ridge", "ElasticNet"):
            specs.append((f"64维+协变量 {nm}", XC, reg_models()[nm], 100 if nm == "Ridge" else 25))
        for label, X, mk, npm in specs:
            obs, oof = cv_reg(mk, X, y, splits)
            nm_, p = perm_p(cv_reg, mk, X, y, splits, obs, npm) if npm else (np.nan, np.nan)
            log(f"{label:28s} {obs:+8.3f} {nm_:+10.3f} {p:7.3f}")
            rows.append(dict(sample=sname, task="回归", outcome=o, model=label, metric="R2", value=obs, perm_null=nm_, perm_p=p))
    ybin = {"睡眠改善(减分率>0)": (s["睡眠减分率"] > 0).astype(int).to_numpy(),
            "总分有效(减分率>=0.5)": (s["总分减分率"] >= 0.5).astype(int).to_numpy()}
    for tname, yb in ybin.items():
        splits = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(X64, yb))
        log(f"\n--- 分类: {tname}, 阳性比例 {yb.mean():.2f} (AUC; 10 折×2 重复分层) ---")
        log(f"{'模型':28s} {'CV AUC':>8s} {'排列零均值':>10s} {'排列p':>7s}")
        specs = [("仅协变量(基线, Logistic)", C, clf_models()["Logistic(L2)"], 0)]
        for nm, mk in clf_models().items():
            specs.append((f"64维 {nm}", X64, mk, 50 if nm.startswith("Log") else 25))
        specs.append(("64维+协变量 Logistic(L2)", XC, clf_models()["Logistic(L2)"], 50))
        for label, X, mk, npm in specs:
            obs, _ = cv_clf(mk, X, yb, splits)
            nm_, p = perm_p(cv_clf, mk, X, yb, splits, obs, npm, stratified=True) if npm else (np.nan, np.nan)
            log(f"{label:28s} {obs:8.3f} {nm_:10.3f} {p:7.3f}")
            rows.append(dict(sample=sname, task="分类", outcome=tname, model=label, metric="AUC", value=obs, perm_null=nm_, perm_p=p))


def main():
    D.mkdir(exist_ok=True)
    F = pd.read_csv(OUT / "clustering_bdz_64d" / "features64.csv", dtype={"脑电编号": str})
    J = pd.read_csv(OUT / "eeg_analysis_joined.csv", dtype={"脑电编号": str})
    X = m64.design64(F)
    X.insert(0, "脑电编号", F["脑电编号"])
    d = X.merge(J[["脑电编号", "年龄", "性别", "BDZ是否使用"] + OUTCOMES], on="脑电编号").dropna(subset=OUTCOMES + ["BDZ是否使用"])
    d["age"], d["male"] = pd.to_numeric(d["年龄"]), (d["性别"] == "男").astype(int)
    d["bdz"] = pd.to_numeric(d["BDZ是否使用"]).astype(int)
    feats = [c for c in X.columns if c != "脑电编号"]
    rows = []
    for sname, s, cov in (("BDZ使用", d[d.bdz == 1], ["age", "male"]), ("未使用BDZ", d[d.bdz == 0], ["age", "male"]),
                          ("全体(参考)", d, ["age", "male", "bdz"])):
        evaluate(sname, s, feats, cov, rows)
        pd.DataFrame(rows).to_csv(D / "cv_results.csv", index=False, encoding="utf-8-sig")
    (D / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
