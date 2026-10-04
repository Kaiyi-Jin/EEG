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
        if out and out[-1][-1] in ",，":
            out[-1] += l
        else:
            out.append(l)
    return "\n".join(out)


def field(pat, text):
    m = re.search(pat, text)
    return clean(m.group(1)) if m else ""


def parse_report(p1):
    h = p1.split("脑电图：")[0]
    eeg = p1.split("脑电图：", 1)[1] if "脑电图：" in p1 else ""
    eeg_txt, _, rest = eeg.partition("脑电地形图：")
    topo, _, concl = rest.partition("结论：")
    concl = concl.split("仅供参考")[0]
    return {
        "姓名": field(r"姓名[:：]\s*(\S+)", h),
        "年龄": field(r"年龄[:：]\s*(\d+)", h),
        "性别": field(r"性别[:：]\s*(\S+)", h),
        "脑电编号": field(r"编号[:：]\s*(\d+)", h),
        "左右利": field(r"左右利[:：]\s*(\S+)", h),
        "是否合作": field(r"合作[:：]\s*(\S+)", h),
        "检查日期": field(r"检查日期[:：]\s*([\d-]+)", h),
        "意识": field(r"意识[:：]\s*(\S+)", h),
        "药物": field(r"药物[:：]\s*(\S+)", h),
        "临床诊断": field(r"临床诊断[:：]\s*(.+)", h),
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
        wf.update({"脑电编号": row["脑电编号"], "姓名": row["姓名"]})
        (WAVE_DIR / f"{row['脑电编号']}.json").write_text(
            json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "eeg_reports.csv", index=False, encoding="utf-8-sig")
    print(df.shape, "dup 脑电编号:", df["脑电编号"].duplicated().sum())
    print(df.isna().sum()[lambda s: s > 0])
    print((df == "").sum()[lambda s: s > 0])


if __name__ == "__main__":
    main()
