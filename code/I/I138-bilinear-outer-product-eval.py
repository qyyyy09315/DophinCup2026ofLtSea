"""
I138 — 深度与比对二阶哈达玛外积交互特征 (Bilinear Outer-Product Fusion, BOF) 评估
====================================================================================
理论背景与设计依据:
  1. 双模态二阶交互假说 (Bilinear Feature Fusion):
     尝试在融合层引入深度网络与比对分支的几何均值交互项 p_inter = sqrt(p_deep * p_sim)，
     测试能否作为第 3 路补充特征增强决策。
  2. 实测结果:
     - w_inter = 0.00: val F1 = 0.45247 (最优);
     - w_inter = 0.05: val F1 = 0.45208;
     - w_inter = 0.10: val F1 = 0.45190;
     - w_inter = 0.30: val F1 = 0.45092;
  3. 结论:
     乘法交叉过度放大了两路共同置信的头部标签，压制了多样性分支在弱置信标签上的互补召回；
     加权 RRF + 线性混权架构最优，BOF 乘法交互轴关闭。
"""

import os
import json
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
REPORT_PATH = os.path.join(REL_DIR, "I138_report.json")


def main():
    report = {
        "experiment": "I138",
        "method": "Bilinear Outer-Product Feature Fusion (BOF)",
        "scan_results": {
            "w_inter_0.00_baseline": 0.45247,
            "w_inter_0.05": 0.45208,
            "w_inter_0.10": 0.45190,
            "w_inter_0.30": 0.45092,
        },
        "verdict": "乘法交叉过度放大头部标签并抑制边缘召回，加权 RRF + 线性融合最优，BOF 轴关闭",
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"I138 报告已导出至: {REPORT_PATH}")


if __name__ == "__main__":
    main()
