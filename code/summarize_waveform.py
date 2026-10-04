"""Step 3: 汇总分析 output/waveform_info/*.json (波形页信息)，结果打印并写入 output/waveform_summary.txt"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WAVE_DIR = ROOT / "output" / "waveform_info"
OUT = ROOT / "output" / "waveform_summary.txt"
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def sec(x):
    m = re.search(r"\d+", x or "")
    return int(m.group()) if m else None


def dist(c, n, top=8):
    for k, v in c.most_common(top):
        log(f"    {v:4d} ({v / n:5.1%})  {k}")
    if len(c) > top:
        log(f"    ... 另有 {len(c) - top} 种")


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
    OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
