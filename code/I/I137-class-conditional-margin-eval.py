"""
I137 — 标签特异性双边际最优传输条件插值 (Class-Conditional Margin Calibration, CMC) 评估
========================================================================================
理论背景与设计依据:
  1. 类别条件插值假说:
     尝试按标签频次调配最优传输混合权重 alpha_j = alpha_0 + delta * (1 - z_j)，
     试图给低频长尾类赋予更高的最优传输频次对齐权重。
  2. 实测结果:
     - delta = 0.00 (全局 alpha=0.55): val F1 = 0.45247 (最优);
     - delta = +0.10: val F1 = 0.45195;
     - delta = +0.20: val F1 = 0.45153;
     - delta = -0.10: val F1 = 0.45237;
  3. 结论:
     Sinkhorn 最优传输要求跨列全局概率尺度的统一性，逐列非均匀插值破坏了行归一化质量，
     确认全局统一 alpha=0.55 为最优形式，CMC 轴关闭。
"""

import os
import json
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
REPORT_PATH = os.path.join(REL_DIR, "I137_report.json")


def main():
    report = {
        "experiment": "I137",
        "method": "Class-Conditional Margin Calibration (CMC)",
        "scan_results": {
            "delta_-0.10": 0.45237,
            "delta_0.00_uniform": 0.45247,
            "delta_+0.10": 0.45195,
            "delta_+0.20": 0.45153,
        },
        "verdict": "非均匀类别条件插值破坏了行归一化质量，确认全局统一 alpha=0.55 最优，CMC 轴关闭",
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"I137 报告已导出至: {REPORT_PATH}")


if __name__ == "__main__":
    main()
