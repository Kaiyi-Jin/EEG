"""Step 1: 提取 脑电/*.pdf 的报告信息 -> output/eeg_reports.csv，
第二页起的脑电波形信息 -> output/waveform_info/<脑电编号>.json"""
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "脑电"
OUT_DIR = ROOT / "output"
WAVE_DIR = OUT_DIR / "waveform_info"


def pdf_pages(path):
    txt = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                         capture_output=True, text=True, check=True).stdout
    return txt.split("\f")[:-1] if txt.endswith("\f") else txt.split("\f")


def clean(s):
    """去掉 pdf 抽取带来的汉字间多余空格(仅保留两侧都是 ASCII 字母数字的空格)"""
    return re.sub(r"(?<![A-Za-z0-9])[ \t]+|[ \t]+(?![A-Za-z0-9])", "", s.strip())


def free_text(block):
    lines = [clean(l) for l in block.splitlines() if l.strip()]
    out = []
    for l in lines:  # 以逗号结尾的行与下一行合并(pdf 自动换行)
        # 以逗号结尾、或本行无"xx："标签且上一行未以句号结束 -> 视为 pdf 自动换行
        if out and (out[-1][-1] in ",，" or ("：" not in l and out[-1][-1] not in "。；;")):
            out[-1] += l
        else:
            out.append(l)
    return "\n".join(out)


def field(pat, text):
    m = re.search(pat, text)
    return clean(m.group(1)) if m else ""


def parse_report(p1):
    # 头部字段只在同一行内匹配(药物等可能为空, 不能跨行取值)
    m = re.search(r"床号[:：][^\n]*\n", p1)
    if m:
        h, body = p1[:m.end()], p1[m.end():]
    else:
        h, _, body = p1.partition("脑电图：")
    body = body.removeprefix("脑电图：")
    # 特殊诱发脑电图报告: 波形导联名/标记叠在文字之上, 先去掉
    body = re.sub(r"\b(?:FP|F|C|P|O|T)\d-[A-Z]+\d\b|标记", "", body)
    eeg_txt, _, rest = body.partition("脑电地形图：")
    topo, _, concl = rest.partition("结论：")
    if not rest:  # 无地形图的报告: 结论紧跟在脑电图文字后
        eeg_txt, _, concl = eeg_txt.partition("结论：")
    concl = re.split(r"仅供参考|起点\d+秒|处理长度", concl)[0]
    eeg_txt = re.sub(r"^-{3,}.*$", "", eeg_txt, flags=re.M)
    sp = r"[ \t]*"
    return {
        "姓名": field(rf"姓名[:：]{sp}(\S*)", h),
        "年龄": field(rf"年龄[:：]{sp}(\d*)", h),
        "性别": field(rf"性别[:：]{sp}(\S*)", h),
        "脑电编号": field(rf"编号[:：]{sp}(\d*)", h),
        "左右利": field(rf"左右利[:：]{sp}(\S*)", h),
        "是否合作": field(rf"合作[:：]{sp}(\S*)", h),
        "检查日期": field(rf"检查日期[:：]{sp}([\d-]*)", h),
        "意识": field(rf"意识[:：]{sp}(\S*)", h),
        "药物": field(rf"药物[:：]{sp}(\S*)", h),
        "临床诊断": field(rf"临床诊断[:：]{sp}(.*)", h),
        "脑电图": free_text(eeg_txt),
        "脑电地形图": free_text(topo),
        "脑电结论": free_text(concl),
    }


def parse_waveform(pages, p1):
    info = {
        "检查者": field(r"检查者[:：]\s*(\S+)", p1),
        "起点": field(r"起点\s*(\d+秒)\s*[\d:]+", p1),
        "起点时间": field(r"起点\s*\d+秒\s*([\d:]+)", p1),
        "处理长度": field(r"处理长度\s*(\d+秒)", p1),
        "waveforms": [],
    }
    for i, pg in enumerate(pages, start=1):
        if "姓名" in pg and "脑电图" in pg:  # 重复的报告页, 非波形页
            continue
        lines = [l.strip() for l in pg.splitlines() if l.strip()]
        chans = [l for l in lines if re.fullmatch(r"[A-Za-z0-9]+-[A-Za-z0-9]+", l)]
        m = re.search(r"(\d+)秒\s+([\d:]+)\s+(\S+)\s+(\S+)\s+(\S+)\s+编号[:：]\s*(\d+)", pg)
        if not chans and not m:
            continue
        info["waveforms"].append({
            "page": i,
            "channels": chans,
            "marker_row": "标记" in lines,
            "time_in_record": m.group(1) if m else "",
            "clock_time": m.group(2) if m else "",
            "stage": m.group(3) if m else "",       # 背景 / 闪光刺激 ...
            "sensitivity": m.group(4) if m else "",
            "speed": m.group(5) if m else "",
        })
    return info


def main():
    WAVE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        pages = pdf_pages(pdf)
        row = parse_report(pages[0])
        if not row["脑电编号"]:  # 无报告页时, 编号取自波形页页脚
            row["脑电编号"] = field(r"编号[:：]\s*(\d+)", "\n".join(pages))
        row["源文件"] = pdf.name
        row["PDF页数"] = len(pages)
        rows.append(row)
        has_report = "姓名" in pages[0]  # 个别 pdf(如夏建香)只有波形页, 没有文字报告
        wf = parse_waveform(pages[1:] if has_report else pages, pages[0])
        wf.update({"脑电编号": row["脑电编号"], "姓名": row["姓名"], "源文件": pdf.name})
        (WAVE_DIR / f"{row['脑电编号']}.json").write_text(
            json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "eeg_reports.csv", index=False, encoding="utf-8-sig")
    print(df.shape, "dup 脑电编号:", df["脑电编号"].duplicated().sum())
    print(df.isna().sum()[lambda s: s > 0])
    print((df == "").sum()[lambda s: s > 0])


if __name__ == "__main__":
    main()
