# EEG 报告提取

从 `脑电/*.pdf` 提取脑电报告信息并与 `分析数据.xlsx` 对接(仅代码, 不含患者数据)。

```bash
python3 -m venv .venv && .venv/bin/pip install -r code/requirements.txt   # 另需 poppler (brew install poppler)
.venv/bin/python code/extract_eeg.py       # -> output/eeg_reports.csv, output/waveform_info/<脑电编号>.json
.venv/bin/python code/add_analysis_id.py   # 新增 分析数据编号 / 匹配方式 两列, 用于后续 join
```
