"""Step 3a: 从 PDF 第二页起的矢量波形还原信号并做信号层面 QC。
输出(均在 output/, 已 gitignore):
  signals/<脑电编号>.npz   每个波形页的信号(uV), 通道名, 元信息
  qc_pages.csv             每页一行: 点数/等效采样率/坐标精度/行距/起点瞬态/半区稳定性 ...
  qc_channels.csv          每页每导联一行: std/ptp/平线/削顶/过零率/频带功率 ...
  qc_markers.csv           标记通道上的事件和页面上的其余文字
  qc_plots/*.png           分布图
"""
import json
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "脑电"
OUT = ROOT / "output"
SIG_DIR = OUT / "signals"
PT_PER_MM = 72 / 25.4
DISPLAY_SEC = 8.0
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "hf": (30, 49.5)}
CH_RE = re.compile(r"^[A-Za-z0-9]+-[A-Za-z0-9]+$")
KNOWN_RE = re.compile(r"^(智能脑电|-|贵州医科大学附属医院|标记|\d+秒|\d\d:\d\d:\d\d|背景|闪光刺激|过度换气|"
                      r"\d+uV/[\d.]+mm|\d厘米/秒|编号:\d+|[A-Za-z0-9]+-[A-Za-z0-9]+)$")


def parse_sens(s):
    m = re.match(r"(\d+(?:\.\d+)?)uV/(\d+(?:\.\d+)?)mm", s or "")
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def band_power(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return np.trapezoid(p[..., m], f[m], axis=-1)


def extract_page(pg):
    """返回 (names, y[ch, n] 单位 pt(向下为正), centers[ch], x 坐标, 坐标精度信息) 或 None"""
    words = pg.extract_words()
    labels = [(w["text"], (w["top"] + w["bottom"]) / 2) for w in words if w["x0"] < 100 and
              (CH_RE.match(w["text"]) or w["text"] == "标记")]
    labels.sort(key=lambda t: t[1])
    segs = [l for l in pg.lines if abs(l["pts"][0][0] - l["pts"][1][0]) > 0.1]
    if not labels or len(segs) < 500:
        return None
    names = [t for t, _ in labels]
    centers = np.array([c for _, c in labels])
    cols = {}
    for l in segs:  # 按 x 分列, 保持 PDF 绘制顺序
        cols.setdefault(round(l["pts"][0][0], 2), []).append(l)
    xs = sorted(cols)
    n_rows = len(names)
    y = np.full((n_rows, len(xs) + 1), np.nan)
    ambiguous = 0
    for j, x in enumerate(xs):
        col = cols[x]
        if len(col) == n_rows:  # 绘制顺序即通道顺序
            for i, l in enumerate(col):
                y[i, j] = l["pts"][0][1]
                y[i, j + 1] = l["pts"][1][1]
        else:  # 回退: 按起点最近的基线分配
            ambiguous += 1
            for l in col:
                i = int(np.argmin(np.abs(centers - l["pts"][0][1])))
                y[i, j] = l["pts"][0][1]
                y[i, j + 1] = l["pts"][1][1]
    # 端点缺失用线性插值补
    for i in range(n_rows):
        bad = np.isnan(y[i])
        if bad.any() and (~bad).sum() > 2:
            y[i, bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), y[i, ~bad])
    coords = np.array([p for l in segs for p in l["pts"]])
    dec = Counter(len(f"{v:.6f}".rstrip("0").split(".")[1]) if "." in f"{v:.6f}".rstrip("0") else 0
                  for v in coords[:, 1][:4000])
    ystep = np.diff(np.unique(np.round(coords[:, 1], 3)))
    xstep = np.diff(np.array(xs))
    words_other = [w["text"] for w in words if not KNOWN_RE.match(w["text"])]
    return dict(names=names, y=y, centers=centers, xs=np.array(xs), ambiguous=ambiguous,
                min_ystep=float(ystep[ystep > 0.0005].min()) if len(ystep) else np.nan,
                med_xstep=float(np.median(xstep)), dec=dec, other_words=words_other)


def analyse_page(pdf_id, page_no, info, meta, ndur):
    """meta: json 中该页的 dict; 返回 (page_row, channel_rows, signals_uV, names)"""
    names, y, centers = info["names"], info["y"], info["centers"]
    sens_uv, sens_mm = parse_sens(meta["sensitivity"])
    sens_missing = sens_uv is None
    if sens_missing:
        sens_uv, sens_mm = 50.0, 5.0
    uv_per_pt = sens_uv / (sens_mm * PT_PER_MM)
    n = y.shape[1]
    fs_len = n / ndur if ndur else np.nan
    fs_paper = 1 / info["med_xstep"] * 30 * PT_PER_MM  # 按 3cm/s 走纸反推
    fs = 100.0 if np.isnan(fs_len) else fs_len
    mk = names.index("标记") if "标记" in names else None
    ch_idx = [i for i, nm in enumerate(names) if nm != "标记"]
    sig = -(y - np.nanmean(y, axis=1, keepdims=True)) * uv_per_pt  # 上正下负, uV
    prow = dict(脑电编号=pdf_id, page=page_no, stage=meta["stage"], sensitivity=meta["sensitivity"],
                sens_missing=sens_missing, n_points=n, 处理长度秒=ndur, fs_len=fs_len, fs_paper=fs_paper,
                x_step_pt=info["med_xstep"], y_min_step_pt=info["min_ystep"],
                y_res_uV=info["min_ystep"] * uv_per_pt, uV_per_pt=uv_per_pt,
                row_spacing_pt=float(np.median(np.diff(centers))), ambiguous_cols=info["ambiguous"],
                n_rows=len(names), 起点秒=meta.get("time_in_record"))
    nper = min(int(2 * fs), n)
    crows = []
    welch_all = []
    for i in ch_idx:
        s = sig[i]
        yy = y[i]
        d = np.diff(yy)
        ptp_pt = yy.max() - yy.min()
        ext = (np.abs(yy - yy.max()) < 0.13) | (np.abs(yy - yy.min()) < 0.13)
        run, clip = 0, 0
        for e in ext:  # 连续>=3 点停在同一极值 -> 疑似削顶
            run = run + 1 if e else 0
            if run == 3:
                clip += 3
            elif run > 3:
                clip += 1
        sc = s - s.mean()
        f, p = welch(sc, fs=fs, nperseg=nper, noverlap=nper // 2, detrend="linear")
        welch_all.append(p)
        tot = band_power(f, p, 0.5, 49.5)
        r = dict(脑电编号=pdf_id, page=page_no, ch=names[i], std_uV=s.std(), ptp_uV=ptp_pt * uv_per_pt,
                 ptp_pt=ptp_pt, flat_frac=float(np.mean(d == 0)), n_levels=len(np.unique(np.round(yy, 3))),
                 clip_frac=clip / len(yy), overrange=ptp_pt > np.median(np.diff(centers)),
                 off_center_pt=float(np.mean(yy) - centers[i]),
                 zcr_hz=float(np.sum(np.diff(np.sign(sc)) != 0) / 2 / (n / fs)),
                 hf_40_49=float(band_power(f, p, 40, 49.5) / tot),
                 p_lo_ratio=float(band_power(f, p, 0.5, 1) / band_power(f, p, 2, 4)),
                 p_hf_ratio=float(band_power(f, p, 30, 49.5) / band_power(f, p, 8, 13)),
                 **{f"rel_{k}": float(band_power(f, p, *v) / tot) for k, v in BANDS.items()},
                 abs_alpha=float(band_power(f, p, 8, 13)), abs_delta=float(band_power(f, p, 0.5, 4)))
        # Welch 分段间 alpha 功率的变异系数
        fs_, _, S = __import__("scipy.signal", fromlist=["spectrogram"]).spectrogram(
            sc, fs=fs, nperseg=nper, noverlap=nper // 2, detrend="linear")
        a = S[(fs_ >= 8) & (fs_ < 13)].sum(0)
        r["alpha_seg_cv"] = float(a.std() / a.mean()) if a.mean() > 0 else np.nan
        # 半区稳定性: 前 4 秒 vs 后 4 秒, 相对频带功率
        h = n // 2
        for tag, seg in (("h1", sc[:h]), ("h2", sc[h:])):
            ff, pp = welch(seg, fs=fs, nperseg=min(int(fs), len(seg)), detrend="linear")
            tt = band_power(ff, pp, 0.5, 49.5)
            for k in ("delta", "theta", "alpha", "beta"):
                r[f"{tag}_{k}"] = float(band_power(ff, pp, *BANDS[k]) / tt)
        # 开头瞬态: 前 0.5 s 对其余部分
        k0 = int(0.5 * fs)
        dy = np.abs(np.diff(sc)) 
        r["start_rms_ratio"] = float(np.sqrt(np.mean((s[:k0] - s[k0:].mean()) ** 2)) / (s[k0:].std() + 1e-9))
        r["start_slope_ratio"] = float(dy[:k0].max() / (np.percentile(dy[k0:], 99) + 1e-9))
        crows.append(r)
    mrow = {}
    if mk is not None:
        m = y[mk]
        tr = np.flatnonzero(np.abs(np.diff(m)) > 0.3)  # 标记行的跳变位置(样本序号)
        d = np.diff(tr)
        # 绝大多数页上它是每 0.5 s 翻转一次的方波(时间标尺); 间隔/次数不规则才可能是真正的事件标记
        regular = len(tr) >= 2 and d.min() >= 45 and d.max() <= 55
        mrow = dict(marker_transitions=len(tr), marker_regular=bool(regular),
                    marker_positions=" ".join(map(str, tr)), marker_levels=len(np.unique(np.round(m, 1))))
    prow.update(mrow)
    return prow, crows, sig[ch_idx].astype(np.float32), [names[i] for i in ch_idx]


def process(jpath):
    meta = json.loads(Path(jpath).read_text(encoding="utf-8"))
    pid = meta["脑电编号"]
    pdf_path = PDF_DIR / meta["源文件"]
    ndur = DISPLAY_SEC  # 波形页固定显示 8 s 窗口(页眉里的"处理长度"是地形图分析长度, 与之无关)
    prows, crows, other, store = [], [], [], {}
    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for pno, pg in enumerate(pdf.pages, 1):
            info = extract_page(pg)
            if info is not None:
                pages.append((pno, info))
        for (pno, info), wm in zip(pages, meta["waveforms"]):
            pr, cr, sg, nm = analyse_page(pid, pno, info, wm, ndur)
            prows.append(pr)
            crows += cr
            store[f"p{pno}"] = sg
            store[f"p{pno}_ch"] = np.array(nm)
            other += [dict(脑电编号=pid, page=pno, text=t) for t in info["other_words"]]
            pr["dec_hist"] = json.dumps(dict(info["dec"]))
    if store:
        np.savez_compressed(SIG_DIR / f"{pid}.npz", **store)
    return prows, crows, other, len(meta["waveforms"]), len(pages)


def run():
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted((OUT / "waveform_info").glob("*.json"))
    with Pool(6) as pool:
        res = pool.map(process, files, chunksize=4)
    P = [r for x in res for r in x[0]]
    C = [r for x in res for r in x[1]]
    O = [r for x in res for r in x[2]]
    mism = [(f.stem, x[3], x[4]) for f, x in zip(files, res) if x[3] != x[4]]
    pd.DataFrame(P).to_csv(OUT / "qc_pages.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(C).to_csv(OUT / "qc_channels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(O, columns=["脑电编号", "page", "text"]).to_csv(OUT / "qc_other_text.csv", index=False, encoding="utf-8-sig")
    print(f"pages {len(P)}, channel-rows {len(C)}, json页数与PDF矢量页数不一致: {mism}")


if __name__ == "__main__":
    run()
