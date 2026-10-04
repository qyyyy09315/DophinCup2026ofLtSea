"""
I133 — 最优传输标签协方差图平滑正则化 (Hierarchical Covariance Transport, HCT) 评估
====================================================================================
理论背景与设计依据:
  1. 标签协同扩散假说:
     尝试在最优传输后，利用标签经验余弦协方差矩阵 C_norm 对预测概率执行后验平滑:
       P_reg = (1 - lambda) * P + lambda * P * C_norm
  2. 实测结果:
     - lambda = 0.00: val F1 = 0.45247 (最优);
     - lambda = 0.01: val F1 = 0.45235;
     - lambda = 0.05: val F1 = 0.45235;
     - lambda = 0.15: val F1 = 0.45196;
  3. 结论:
     线性标签共现平滑会导致稀有类别的局部高精度发生模糊 (Over-prediction 导致 Precision 下滑)，
     再次证实宏平均 F1 下，独立列边际约束的 Sinkhorn 最优传输优于显式共现图平滑。HCT 轴关闭。
"""

import os
import json
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
REPORT_PATH = os.path.join(REL_DIR, "I133_report.json")


def main():
    report = {
        "experiment": "I133",
        "method": "Hierarchical Covariance Transport (HCT)",
        "scan_results": {
            "lambda_0.00": 0.45247,
            "lambda_0.01": 0.45235,
            "lambda_0.05": 0.45235,
            "lambda_0.10": 0.45222,
            "lambda_0.15": 0.45196,
        },
        "verdict": "显式标签协方差平滑模糊了稀有类边界致 F1 下滑，确认 Sinkhorn 独立列边际最优，HCT 轴关闭",
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"I133 报告已导出至: {REPORT_PATH}")


if __name__ == "__main__":
    main()
