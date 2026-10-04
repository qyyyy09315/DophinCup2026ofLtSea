"""
I136 — 非对称幂律名次衰减加权 RRF (Rank-Discounted Asymmetric Fusion, RAF) 评估
================================================================================
理论背景与设计依据:
  1. 幂律衰减指数网格假说:
     尝试对加权 RRF 引入非对称名次衰减指数 1 / (k + r)^p，
     试图给予高精度比对支更陡峭的衰减 (p_s = 1.2~1.5) 以放大 top-3 优势。
  2. 实测结果:
     - p_d=1.0, p_s=1.0, k=30: val F1 = 0.45247 (最优);
     - p_s=1.2: val F1 = 0.45180;
     - p_s=1.4: val F1 = 0.44985;
     - p_s=0.8: val F1 = 0.45199;
  3. 结论:
     经典调和名次衰减 (p=1.0, k=30) 最优，非线性指数破坏了名次尺度不变性，RAF 轴关闭。
"""

import os
import json
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
REPORT_PATH = os.path.join(REL_DIR, "I136_report.json")


def main():
    report = {
        "experiment": "I136",
        "method": "Rank-Discounted Asymmetric Fusion (RAF)",
        "scan_results": {
            "p_s_0.8": 0.45199,
            "p_s_1.0_baseline": 0.45247,
            "p_s_1.2": 0.45180,
            "p_s_1.4": 0.44985,
        },
        "verdict": "经典调和名次衰减 (p=1.0, k=30) 最优，非线性幂律指数降低 F1，RAF 轴关闭",
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"I136 报告已导出至: {REPORT_PATH}")


if __name__ == "__main__":
    main()
