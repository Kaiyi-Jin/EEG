"""分支 predict-37d-kfold: 37 维(8 脑区 × δθαβ 相对功率 CLR = 32 维 + α峰频率/1/f斜率/截距/log(θ/β)/log(枕/额α) 5 维)
-> 睡眠减分率 / 总分减分率 预测模型, BDZ 分组, 重复 K 折 CV。模型与评估流程同 predict_outcome_64d.py。
输出 output/predict_37d/"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gmm_single_page as g  # noqa: E402
import predict_outcome_64d as p  # noqa: E402

OUT = g.OUT
D = OUT / "predict_37d"
p.FEAT_LABEL = "37维"


def main():
    D.mkdir(exist_ok=True)
    F = pd.read_csv(OUT / "clustering" / "features.csv", dtype={"脑电编号": str})
    J = pd.read_csv(OUT / "eeg_analysis_joined.csv", dtype={"脑电编号": str})
    X = g.design(F)                                   # 32 维区域 CLR + 5 个补充特征 = 37 维
    X.insert(0, "脑电编号", F["脑电编号"])
    d = X.merge(J[["脑电编号", "年龄", "性别", "BDZ是否使用"] + p.OUTCOMES], on="脑电编号").dropna(subset=p.OUTCOMES + ["BDZ是否使用"])
    d["age"], d["male"] = pd.to_numeric(d["年龄"]), (d["性别"] == "男").astype(int)
    d["bdz"] = pd.to_numeric(d["BDZ是否使用"]).astype(int)
    feats = [c for c in X.columns if c != "脑电编号"]
    p.log(f"特征维数 {len(feats)}")
    rows = []
    for sname, s, cov in (("BDZ使用", d[d.bdz == 1], ["age", "male"]), ("未使用BDZ", d[d.bdz == 0], ["age", "male"]),
                          ("全体(参考)", d, ["age", "male", "bdz"])):
        p.evaluate(sname, s, feats, cov, rows)
        pd.DataFrame(rows).to_csv(D / "cv_results.csv", index=False, encoding="utf-8-sig")
    (D / "summary.txt").write_text("\n".join(p.lines), encoding="utf-8")


if __name__ == "__main__":
    main()
