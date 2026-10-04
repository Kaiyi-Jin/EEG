# EEG 报告提取

从 `脑电/*.pdf` 提取脑电报告信息并与 `分析数据.xlsx` 对接(仅代码, 不含患者数据)。

```bash
python3 -m venv .venv && .venv/bin/pip install -r code/requirements.txt   # 另需 poppler (brew install poppler)
.venv/bin/python code/extract_eeg.py       # -> output/eeg_reports.csv, output/waveform_info/<脑电编号>.json
.venv/bin/python code/add_analysis_id.py   # 新增 分析数据编号 / 匹配方式 两列, 用于后续 join
```

## 分支 `gmm-single-page`：单页 GMM 聚类
```bash
.venv/bin/python code/qc_signals.py          # 还原信号 + QC (output/signals, qc_*.csv)
.venv/bin/python code/gmm_single_page.py     # -> output/clustering/
```
- 每人 1 页：背景 + A1/A2 参考导联 + 有灵敏度/处理长度 + 无候选坏导联，多页取起点最早的一页。
- 特征：1–30 Hz 相对功率(δθαβ) 按 8 脑区合并 → CLR；加 α 峰频率、1/f 斜率与截距(specparam 2–30 Hz)、log(θ/β)、log(枕/额 α)。
- 标准化 → PCA(5–10 个主成分) → GMM(diag/tied)，BIC 选 K，并做 bootstrap、去 θ、去补充特征、换协方差、换种子的敏感性分析。
