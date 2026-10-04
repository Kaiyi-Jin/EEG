"""Step 3: 汇总分析 output/waveform_info/*.json (波形页信息)，结果打印并写入 output/waveform_summary.txt"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WAVE_DIR = ROOT / "output" / "waveform_info"
OUT = ROOT / "output" / "waveform_summary.txt"
sys.path.insert(0, str(Path(__file__).resolve().parent))
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def sec(x):
    m = re.search(r"\d+", x) if isinstance(x, str) else None
    return int(m.group()) if m else None


def dist(c, n, top=8):
    for k, v in c.most_common(top):
        log(f"    {v:4d} ({v / n:5.1%})  {k}")
    if len(c) > top:
        log(f"    ... 另有 {len(c) - top} 种")


def pct(x, qs=(1, 5, 50, 95, 99)):
    x = pd.Series(x).dropna()
    return "  ".join(f"p{q}={x.quantile(q / 100):.3g}" for q in qs) + f"  (min={x.min():.3g}, max={x.max():.3g})"


def robust_out(df, col, k=4, log=False):
    """按导联名分组的稳健 z (中位数/MAD) 超过 k 的行"""
    v = np.log10(df[col].clip(lower=1e-6)) if log else df[col]
    g = v.groupby(df["ch"])
    z = (v - g.transform("median")) / (1.4826 * g.transform(lambda a: (a - a.median()).abs().median()) + 1e-12)
    return df[z.abs() > k], z


def signal_qc(recs):
    """信号层面 QC + 状态检查; 数据来自 qc_signals.py (output/qc_*.csv)"""
    if not (OUT.parent / "qc_pages.csv").exists():
        import qc_signals
        qc_signals.run()
    P = pd.read_csv(OUT.parent / "qc_pages.csv", dtype={"脑电编号": str})
    C = pd.read_csv(OUT.parent / "qc_channels.csv", dtype={"脑电编号": str})
    T = pd.read_csv(OUT.parent / "qc_other_text.csv", dtype={"脑电编号": str})
    P["sec0"] = pd.to_numeric(P["起点秒"], errors="coerce")
    P["montage"] = P["脑电编号"].map(lambda i: None)  # 占位, 下面按导联名判断
    mont = {}
    for r in recs:
        for k, w in enumerate(r["waveforms"]):
            ch = w["channels"]
            mont[(r["脑电编号"], k)] = ("A1/A2参考" if ch and ch[0].endswith("A1") else "REF" if ch and "REF" in ch[0] else "双极")
    P["pg_idx"] = P.groupby("脑电编号").cumcount()
    P["montage"] = [mont.get((a, b)) for a, b in zip(P["脑电编号"], P["pg_idx"])]
    C = C.merge(P[["脑电编号", "page", "stage", "sensitivity", "montage"]], on=["脑电编号", "page"])
    npg = len(P)
    log("\n" + "=" * 60 + "\n一、信号层面 QC\n" + "=" * 60)

    log("\n【1.1 等效采样率】(每条曲线的点数 / 8 秒显示窗口)")
    log(f"  每页每导联点数: {dict(Counter(P['n_points']))}")
    fl = P["fs_len"].dropna()
    log(f"  点数/8秒: 取值 {sorted(set(fl.round(2)))} Hz (页眉的'处理长度'是地形图分析长度, 与波形窗口无关)")
    log(f"  按 3 厘米/秒 走纸与点间距反推: {sorted(set(P['fs_paper'].round(1)))} Hz (点间距 x_step={sorted(set(P['x_step_pt'].round(3)))} pt)")
    log(f"  => 所有页等效采样率一致 (约 100 Hz, Nyquist=50 Hz): {'是' if P['n_points'].nunique() == 1 and P['x_step_pt'].round(3).nunique() == 1 else '否, 需先重采样'}")
    log("  注意: 显示曲线只有 100 Hz, 这是打印用的抽点, 原始采样率不可知; 50 Hz 工频刚好落在 Nyquist 上, "
        "无法在此数据中直接检出/评估 50 Hz 陷波.")
    log(f"  间接证据(低通/抗混叠): 40-49.5 Hz 功率占 0.5-49.5 Hz 的比例 中位数 {C['hf_40_49'].median():.2e}, p99 {C['hf_40_49'].quantile(.99):.2e};"
        f" 30-49.5Hz/8-13Hz 功率比 中位数 {C['p_hf_ratio'].median():.3f}")

    log("\n【1.2 滤波设置】")
    ft = T[T["text"].str.contains("滤波|陷波|采样|Hz", na=False)]
    log(f"  页面文字里写有滤波/采样信息的: {ft['脑电编号'].nunique()}/{len(recs)} 人" +
        (f": {sorted(ft['脑电编号'].unique())}  内容 {sorted(ft['text'].unique())}" if len(ft) else ""))
    log("  其余人页眉仅有 '起点/处理长度/灵敏度/走纸速度', 未记录高频/低频滤波、50Hz 陷波、采样率 => 无法核实是否一致。")
    log("  (已扫描所有波形页上的全部文字; 第一页文字报告里也没有滤波描述.)")
    log("  建议: 向设备/科室确认默认滤波; 用频谱形状做事后核查 (见下), 并加入协变量.")
    rob, _ = robust_out(C, "p_hf_ratio", 4, log=True)
    log(f"  频谱形状离群(30-49Hz/8-13Hz 功率比, 按导联稳健 z>4): {len(rob)} 个导联-页, 涉及 {rob[['脑电编号','page']].drop_duplicates().shape[0]} 页")
    rob2, _ = robust_out(C, "p_lo_ratio", 4, log=True)
    log(f"  低频形状离群(0.5-1Hz/2-4Hz 功率比): {len(rob2)} 个导联-页 (>0 表示高通/基线漂移不同)")

    log("\n【1.3 幅值标定】")
    for sens, g in P.groupby(P["sensitivity"].fillna("(空)")):
        log(f"  {sens:>10}: {len(g):3d} 页, 1pt={g['uV_per_pt'].iloc[0]:.3f} uV, 最小 y 步长 {g['y_min_step_pt'].min():.2f} pt = {g['y_res_uV'].min():.2f} uV")
    log(f"  公式: uV = Δy(pt) × 灵敏度(uV) / (灵敏度(mm) × 2.8346 pt/mm); 3.5mm 页校正系数 5/3.5 = {5 / 3.5:.2f}")
    log(f"  灵敏度为空: {P[P['sens_missing']][['脑电编号','page']].values.tolist()} -> 按 50uV/5mm 假设处理, 需人工核对")
    log(f"  PDF 坐标稳定性: 行距(pt) {sorted(set(P['row_spacing_pt'].round(2)))}, 点间距 {sorted(set(P['x_step_pt'].round(3)))};"
        f" 行距标准差 {P['row_spacing_pt'].std():.3f} pt => {'稳定' if P['row_spacing_pt'].std() < 0.1 else '不稳定'}")
    log("  (页面内没有定标方波/标尺, 1mm 对应长度只能用 页面缩放(2.8346 pt/mm) 与 走纸点间距 间接验证)")
    a = C[C["montage"] == "A1/A2参考"]
    log("  校正自检(参考导联, 各灵敏度的中位 std uV, 校正后应与5mm接近): " +
        ", ".join(f"{k}: {v:.1f}" for k, v in a.groupby("sensitivity")["std_uV"].median().items()))

    log("\n【1.4 坏导联 / 伪迹】(共 %d 个 导联-页)" % len(C))
    for c, lab in [("std_uV", "标准差 uV"), ("ptp_uV", "峰峰值 uV"), ("flat_frac", "相邻点相等比例(平线)"),
                   ("clip_frac", "削顶(连续>=3点停在极值)比例"), ("zcr_hz", "过零率 Hz"),
                   ("hf_40_49", "40-49Hz功率占比(高频/肌电)"), ("off_center_pt", "曲线均值偏离行中心 pt")]:
        log(f"  {lab}: {pct(C[c])}")
    flat = C[(C["std_uV"] < 1) | (C["flat_frac"] > 0.5)]
    sat = C[C["clip_frac"] > 0.01]
    over = C[C["overrange"]]
    log(f"  拟用阈值 -> 平线(std<1uV 或 平线比例>0.5): {len(flat)}; 削顶(>1%): {len(sat)} {sat[['脑电编号','page','ch']].values.tolist()[:5]};"
        f" 曲线幅度超过行距(与邻导联重叠): {len(over)} ({len(over) / len(C):.1%}), 涉及 {over[['脑电编号','page']].drop_duplicates().shape[0]} 页")
    log(f"  显示范围: 峰峰值最大 {C['ptp_pt'].max():.1f} pt (行距 {P['row_spacing_pt'].median():.0f} pt), 大幅曲线与邻行重叠而非被行边界裁切; "
        f"仅 {len(sat)} 个导联-页出现平顶(见上, 均需人工看图确认是否为设备削顶)")
    hi, _ = robust_out(C, "std_uV", 5, log=True)
    log(f"  std 按导联稳健 z>5 的离群导联-页: {len(hi)} (候选坏导联/大幅伪迹), 例: {hi.sort_values('std_uV', ascending=False)[['脑电编号','page','ch','std_uV']].head(5).values.tolist()}")
    zh, _ = robust_out(C, "zcr_hz", 5)
    log(f"  过零率离群(z>5): {len(zh)}; 高频占比 p99 以上: {(C['hf_40_49'] > C['hf_40_49'].quantile(.99)).sum()} 个导联-页")
    bp = C.assign(bad=C.index.isin(set(flat.index) | set(sat.index) | set(hi.index))).groupby(["脑电编号", "page"])["bad"].sum()
    log(f"  每页存在候选坏导联数分布: {dict(Counter(bp))}")
    log("  各导联 std 中位数(uV): " + ", ".join(f"{k}:{v:.1f}" for k, v in C.groupby('ch')['std_uV'].median().sort_values(ascending=False).head(8).items()) + " ...")

    log("\n【1.5 PDF 精度】")
    log(f"  y 坐标量化步长: {sorted(set(P['y_min_step_pt'].round(3)))} pt (等于 {P['y_res_uV'].min():.2f}-{P['y_res_uV'].max():.2f} uV); x 以 {P['x_step_pt'].iloc[0]:.2f} pt 为栅格")
    lv = C["n_levels"]
    log(f"  每条曲线实际出现的不同 y 取值数: {pct(lv)}")
    qn = (P['y_res_uV'] / np.sqrt(12)).median()
    small = C[C["std_uV"] < 5 * (C["ptp_uV"] / C["ptp_pt"].clip(lower=1e-3)).median() * 0.12]
    log(f"  量化噪声约 {qn:.2f} uV rms; 对 std 很小(< ~2uV)的导联-页 {len(small)} 个, 高频/小幅信号会明显失真")
    log(f"  坐标原始小数位(y, 抽样): {dict(Counter({k: v for r in P['dec_hist'].dropna() for k, v in json.loads(r).items()}).most_common(4))} (多余位数是浮点噪声, 有效精度即上面的量化步长)")

    log("\n" + "=" * 60 + "\n二、状态\n" + "=" * 60)
    log("\n【2.1 '标记'通道】")
    log(f"  含 '标记' 行的页: {npg}/{npg}; 跳变次数分布: {dict(Counter(P['marker_transitions']).most_common(6))}")
    log(f"  规则方波(每 0.5s 翻转一次, 实为时间标尺, 不含事件信息): {int(P['marker_regular'].sum())} 页; 不规则(有额外短脉冲, 疑似真实事件标记): {int((~P['marker_regular']).sum())} 页")
    for _, r in P[~P["marker_regular"]].iterrows():
        extra = [int(x) for x in r["marker_positions"].split() if int(x) % 50 != 48]
        log(f"    {r['脑电编号']} p{r['page']} ({r['stage']}): 额外跳变样本点(秒) {[round(e / 100, 2) for e in extra]}")
    txt = T[~T["text"].str.contains("滤波", na=False)]
    log(f"  PDF 中没有'睁眼/闭眼/眨眼/肌电'等文字标注 (页面内其余未识别文字 {len(txt)} 条)。眼状态只能由信号推断(枕区 α 功率, 前额低频眨眼)")

    log("\n【2.2 多页(背景)的病人】")
    ref = C[C["montage"] == "A1/A2参考"]
    occ = ref[ref["ch"].isin(["O1-A1", "O2-A2"])].groupby(["脑电编号", "page"])[["rel_alpha", "abs_alpha"]].mean()
    fro = ref[ref["ch"].isin(["FP1-A1", "FP2-A2"])].groupby(["脑电编号", "page"])["abs_delta"].mean()
    multi = P[P.groupby("脑电编号")["page"].transform("count") > 1]
    log(f"  {multi['脑电编号'].nunique()} 人有多页; 每行: 起点(秒) | 状态 | 导联 | 枕区α相对功率 | 枕区α绝对功率 | 前额0.5-4Hz功率")
    for pid, g in multi.groupby("脑电编号"):
        best = occ.loc[pid]["rel_alpha"].idxmax() if pid in occ.index.get_level_values(0) else None
        for _, r in g.iterrows():
            k = (pid, r["page"])
            o = occ.loc[k] if k in occ.index else None
            log(f"    {pid} p{r['page']}  t={r['sec0']}s  {r['stage']}  {r['montage']}  " +
                (f"α_rel={o['rel_alpha']:.2f}  α_abs={o['abs_alpha']:.0f}  FP_delta={fro.get(k, float('nan')):.0f}" if o is not None else "(非参考导联)") +
                ("   <- α相对功率最高, 疑似闭目" if best == r["page"] else ""))
    log("  结论: 多页间的 α 差异可帮助判断睁/闭目, 但需结合各页起点人工确认; 起点相近(<8s)的页会时间重叠。")
    ov, same = [], []
    for pid, g in multi.groupby("脑电编号"):
        for m, gm in g.groupby("montage"):
            v = sorted(gm["sec0"].dropna())
            if any(b - a < 8 for a, b in zip(v, v[1:])):
                ov.append(pid)
        if g.groupby("sec0")["montage"].nunique().max() > 1:
            same.append(pid)
    log(f"  同一导联方式内起点间隔<8s(片段重叠)的人: {sorted(set(ov))}")
    log(f"  同一起点同时以不同导联方式(参考/双极)显示的人(即同一片段的重复展示, 取一种即可): {same}")
    log("\n【2.3 8 秒片段的起点与开头瞬态】")
    st = pd.Series([sec(r["起点"]) for r in recs])
    log(f"  页眉'起点'缺失: {int(st.isna().sum())}/{len(recs)} 人; 缺失者中各页在记录中的位置(秒)分布: "
        f"{dict(Counter(P[P['脑电编号'].isin([r['脑电编号'] for r, v in zip(recs, st) if pd.isna(v)])]['sec0'].fillna(-1).astype(int)).most_common(5))}")
    miss = [r["脑电编号"] for r in recs if not sec(r["处理长度"])]
    log(f"  处理长度缺失: {len(miss)} 人 {miss} -> 建议排除")
    first = C.merge(P[["脑电编号", "page", "sec0"]], on=["脑电编号", "page"])
    first = first[(first["sec0"] == 0) & (first["montage"] == "A1/A2参考")]
    pr = first.groupby(["脑电编号", "page"]).agg(rms=("start_rms_ratio", "median"), slope=("start_slope_ratio", "median"))
    log(f"  位于记录开头(0s)的页: {len(pr)} 页 ({len(pr) / npg:.0%})")
    log(f"  前 0.5s RMS / 其余 std 的中位比: {pct(pr['rms'])}")
    log(f"  前 0.5s 最大斜率 / 其余 99 分位斜率: {pct(pr['slope'])}")
    bad = pr[(pr["rms"] > 2) | (pr["slope"] > 3)]
    log(f"  疑似开头瞬态(RMS比>2 或 斜率比>3): {len(bad)} 页 {bad.index.tolist()[:10]}; 建议统一丢弃前 1s 或按此标记")

    log("\n【2.4 频谱方差与分裂半区稳定性】(8 秒 @100Hz)")
    log(f"  Welch: 建议 nperseg=200(2s), 50% 重叠 => 7 段, 频率分辨率 0.5Hz; 本次 QC 即按此计算")
    log(f"  α 功率在 Welch 分段间的变异系数 CV: {pct(C['alpha_seg_cv'])}")
    rr = C[C["montage"] == "A1/A2参考"].groupby(["脑电编号", "page"])[[f"{h}_{b}" for h in ("h1", "h2") for b in ("delta", "theta", "alpha", "beta")]].median()
    for b in ("delta", "theta", "alpha", "beta"):
        x, y = rr[f"h1_{b}"], rr[f"h2_{b}"]
        log(f"  {b:5s} 前4s vs 后4s 相对功率(导联中位数): 跨页 Pearson r={np.corrcoef(x, y)[0, 1]:.2f}, Spearman={x.corr(y, method='spearman'):.2f}, 中位|Δ|={np.median(np.abs(x - y)):.3f}")
    low = [b for b in ("delta", "theta", "alpha", "beta") if np.corrcoef(rr[f"h1_{b}"], rr[f"h2_{b}"])[0, 1] < 0.7]
    log(f"  r<0.7 的频带: {low or '无'}")
    log("  (r 越高说明同一人的特征在 8s 内越稳定; 若 r<0.7 的频带不宜单独用作聚类特征)")

    elig = P[(P["stage"] == "背景") & (P["montage"] == "A1/A2参考") & P["处理长度秒"].notna() & ~P["sens_missing"]]
    log(f"\n【2.5 聚类可用(静息背景+参考导联+有处理长度+有灵敏度)】 {len(elig)} 页, {elig['脑电编号'].nunique()} 人 / 共 {len(recs)} 人")
    log("  仍需: 睁/闭目确认、开头瞬态、多页取舍(见 2.2)")
    _plots(P, C)


def _plots(P, C):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = OUT.parent / "qc_plots"
    d.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, (c, t, lg) in zip(ax.ravel(), [("std_uV", "channel std (uV)", True), ("ptp_uV", "peak-to-peak (uV)", True),
                                          ("flat_frac", "flat fraction", False), ("zcr_hz", "zero-crossing rate (Hz)", False),
                                          ("hf_40_49", "40-49.5Hz power fraction", True), ("p_hf_ratio", "P(30-49)/P(8-13)", True)]):
        v = C[c].replace(0, np.nan).dropna()
        a.hist(np.log10(v) if lg else v, bins=60)
        a.set_xlabel(("log10 " if lg else "") + t)
    fig.tight_layout()
    fig.savefig(d / "channel_qc_hist.png", dpi=110)
    plt.close(fig)
    # 平均 PSD(各页中位) 以检查滤波形状是否一致
    from scipy.signal import welch
    ps = []
    for f in (OUT.parent / "signals").glob("*.npz"):
        z = np.load(f)
        for k in z.files:
            if k.endswith("_ch") or not k.startswith("p"):
                continue
            if not str(z[k + "_ch"][0]).endswith("A1"):
                continue
            ff, pp = welch(z[k] - z[k].mean(1, keepdims=True), fs=100, nperseg=200, noverlap=100, detrend="linear")
            ps.append(np.median(pp, axis=0))
    ps = np.array(ps)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ff[1:], ps[:, 1:].T, color="C0", alpha=.08, lw=.6)
    ax.loglog(ff[1:], np.median(ps, 0)[1:], color="k", lw=2, label="median")
    ax.set_xlabel("Hz"); ax.set_ylabel("PSD (uV^2/Hz)"); ax.legend()
    fig.tight_layout()
    fig.savefig(d / "psd_overlay.png", dpi=110)
    plt.close(fig)
    log(f"\n图: {d}/channel_qc_hist.png, psd_overlay.png")


def main():
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(WAVE_DIR.glob("*.json"))]
    n = len(recs)
    log(f"共 {n} 份波形信息\n")

    # 1. 波形页数
    log("【每人波形页数】")
    dist(Counter(len(r["waveforms"]) for r in recs), n)

    # 2. 导联
    montages = Counter()
    per_page = Counter()
    chan_cnt = Counter()
    pat_patterns = []
    for r in recs:
        pats = []
        for w in r["waveforms"]:
            key = ",".join(w["channels"])
            montages[key] += 1
            per_page[len(w["channels"])] += 1
            chan_cnt.update(w["channels"])
            pats.append(key)
        pat_patterns.append(tuple(pats))
    log("\n【导联】")
    log(f"  每页导联数分布:")
    dist(per_page, sum(per_page.values()))
    log(f"  导联组合共 {len(montages)} 种 (按页统计):")
    for i, (k, v) in enumerate(montages.most_common(5), 1):
        ch = k.split(",")
        kind = "参考导联(A1/A2)" if any(c.endswith(("A1", "A2")) for c in ch) else "双极导联" if ch else "无"
        log(f"    组合{i}: {v} 页 ({v / sum(montages.values()):.1%}), {len(ch)} 导, {kind}\n      {k}")
    log("  各导联出现占比(按病人):")
    pc = Counter()
    for r in recs:
        pc.update({c for w in r["waveforms"] for c in w["channels"]})
    log("    " + ", ".join(f"{k}:{v / n:.0%}" for k, v in pc.most_common()))
    same = len(set(pat_patterns)) == 1
    log(f"  所有病人导联(含页序)是否完全一致: {'是' if same else '否'}; 不同模式 {len(set(pat_patterns))} 种")
    dist(Counter(" | ".join(f"{len(set(p.split(',')))}导" for p in pp) for pp in pat_patterns), n, 5)
    sets = Counter(frozenset(c for w in r["waveforms"] for c in w["channels"]) for r in recs)
    log(f"  按病人合并后的导联集合: {len(sets)} 种, 最常见覆盖 {sets.most_common(1)[0][1]}/{n} 人")

    # 3. 脑电时长 / 起点 / 处理长度
    df = pd.DataFrame([{
        "编号": r["脑电编号"], "起点秒": sec(r["起点"]), "处理长度秒": sec(r["处理长度"]),
        "页数": len(r["waveforms"]),
        "波形段起点秒": [sec(w["time_in_record"]) for w in r["waveforms"]],
        "检查者": r["检查者"],
    } for r in recs])
    log("\n【处理长度 / 起点】")
    for c in ["处理长度秒", "起点秒"]:
        s = df[c].dropna()
        log(f"  {c}: 有值 {len(s)}/{n}, 均值 {s.mean():.1f}, 中位 {s.median():.0f}, 范围 {s.min():.0f}-{s.max():.0f}")
    log("  处理长度分布:")
    dist(Counter(df["处理长度秒"].dropna().astype(int)), n, 5)
    log("  (注: pdf 仅含截取的波形片段, 非完整记录时长, 无法得到整段脑电总时长)")
    pos = [t for ts in df["波形段起点秒"] for t in ts if t is not None]
    if pos:
        log(f"  各波形页在记录中的位置(秒): 中位 {pd.Series(pos).median():.0f}, 范围 {min(pos)}-{max(pos)}")

    # 4. 背景/状态、灵敏度、速度
    ws = [w for r in recs for w in r["waveforms"]]
    for name, key in [("状态(背景/闪光刺激等)", "stage"), ("灵敏度", "sensitivity"), ("走纸速度", "speed")]:
        log(f"\n【{name}】(按页, 共 {len(ws)} 页)")
        dist(Counter(w[key] or "(空)" for w in ws), len(ws), 6)
        log(f"  每位病人各页该项是否一致: " +
            f"{sum(len({w[key] for w in r['waveforms']}) <= 1 for r in recs)}/{n} 人一致")
        log(f"  是否所有病人一致: {'是' if len({w[key] for w in ws}) == 1 else '否'}")
    log("\n【各病人含有的状态组合】")
    dist(Counter(" + ".join(w["stage"] or "(空)" for w in r["waveforms"]) for r in recs), n, 8)
    log(f"\n【标记行】有'标记'通道的页占比: {sum(w['marker_row'] for w in ws) / len(ws):.1%}")

    # 5. 检查者
    log("\n【检查者】")
    dist(Counter(df["检查者"].replace("", "(空)")), n, 6)

    # 6. 异常检查
    log("\n【数据质量提示】")
    log(f"  无波形页: {sum(not r['waveforms'] for r in recs)} 人; 无处理长度: {df['处理长度秒'].isna().sum()} 人")
    signal_qc(recs)
    OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
