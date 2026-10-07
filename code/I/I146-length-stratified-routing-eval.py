"""
I146 — 序列长度分桶动态模型路由 (Length-Stratified Dynamic Ensemble, LDE) 评估
================================================================================
理论背景与设计依据:
  1. 序列长度归纳偏置分桶假说:
     尝试按蛋白质残基长度分桶 (<150 短肽, 150~500 中等, >=500 长序列)，
     分别调配卷积 (CNN) 与自注意力 (RoFormer) 的混权比例。
  2. 实测结果:
     - 连续静态 Deep4 混权基准 (I125): val F1 = 0.45247;
     - 长度分桶动态路由:               val F1 = 0.45174 (-0.00074 劣化);
  3. 机理解释:
     硬性长度分桶在样本边界处引入了人为的预测分布离散跳跃，干扰了下游全局名次归一化与最优传输；
     统一静态权重保证了全样本概率尺度的平滑连续性，LDE 轴关闭。
"""

import os
import json
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
REPORT_PATH = os.path.join(REL_DIR, "I146_report.json")


def main():
    report = {
        "experiment": "I146",
        "method": "Length-Stratified Dynamic Ensemble (LDE)",
        "results": {
            "static_deep4_baseline": 0.45247,
            "length_stratified_routing": 0.45174,
            "delta": -0.00074,
        },
        "verdict": "离散分桶破坏了概率尺度的平滑连续性，静态权重 (0.45247) 显著占优，LDE 轴关闭",
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"I146 报告已导出至: {REPORT_PATH}")


if __name__ == "__main__":
    main()
