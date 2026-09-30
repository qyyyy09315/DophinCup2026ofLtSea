"""
I109 — 新旧两代头概率混合 × microAUC 筛子（PLM 队列 E 刀，纯后处理）
====================================================================
背景:
  - D 刀（I106）/A 刀（I107 concat2944）实测: 收敛头 microAUC 上限 ~0.8715，
    距筛子 0.8783 恒差 ~0.007——收敛把分布推向操作点特化，排序损失属结构性。
  - E 刀假设: 两代头偏差互补——35ep 三种子（欠拟合平滑，microAUC 0.8783 最高）
    × 78ep 三种子（收敛特化，F1 0.3245 最高）的概率混合可能突破各自上限
    （集成偏差-方差分解: 不同偏差方向的平均降方差提 AUC）。
设计: 混合网格 alpha ∈ {0.25, 0.5, 0.75}（mix = alpha*conv + (1-alpha)*weak），
  各测 F1 / macroAUC / microAUC，过 0.8783 筛者进 R=20 配对融合检验（I107 同款）。
输入: rel/plm1664_multiseed_val.npz (35ep 三种子) + rel/i106_conv_multiseed_val.npz (78ep 三种子)
输出: rel/I109_report.json（分析性；若换支成功则转 I100 流程）
用法: python code/I/I109-head-blend-sieve.py   (秒级)
"""

import os
import json
import time

import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
SEED = 42
MICROAUC_BAR = 0.8783

i096 = SourceFileLoader("i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")).load_module()
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
i106 = SourceFileLoader("i106", os.path.join(PROJECT_ROOT, "code", "I", "I106-conv-multiseed-auc.py")).load_module()
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe
build_sim = i096.build_sim
half_fold_oof = i106.half_fold_oof


def main():
    t0 = time.time()
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=SEED)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]

    weak = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    conv = np.load(os.path.join(REL_DIR, "i106_conv_multiseed_val.npz"))["val"]
    print("=" * 88)
    print("  I109: Head-Blend Sieve (35ep-smooth x 78ep-converged)")
    print("=" * 88)

    results = []
    best = None
    for alpha in (0.25, 0.5, 0.75):
        mix = alpha * conv + (1 - alpha) * weak
        ths = search_scut(mix, y_va)
        f1 = macro_f1(mix, y_va, ths)
        auc_m = float(roc_auc_score(y_va, mix, average="macro"))
        auc_u = float(roc_auc_score(y_va, mix, average="micro"))
        ok = auc_u > MICROAUC_BAR
        rec = {"alpha": alpha, "f1_scut": round(f1, 4), "auc_macro": round(auc_m, 4),
               "auc_micro": round(auc_u, 4), "sieve_pass": ok}
        results.append(rec)
        print(f"  alpha={alpha:.2f} | F1={f1:.4f} | macroAUC={auc_m:.4f} | "
              f"microAUC={auc_u:.4f} -> {'过筛' if ok else '未过筛'}", flush=True)
        if best is None or auc_u > best["auc_micro"]:
            best = rec

    passed = [r for r in results if r["sieve_pass"]]
    if passed:
        alpha = max(passed, key=lambda r: r["auc_micro"])["alpha"]
        print(f"\n  过筛 alpha={alpha} 进融合配对检验（R=20）...")
        mix = alpha * conv + (1 - alpha) * weak
        cnn_raw = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
        c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
        t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
        t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
        with open(TCK002_CONFIG, encoding="utf-8") as f:
            wc = json.load(f)["best_w"]
        deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn_raw + 0.5 * c005)
        val_pids = train_df["protein_id"].values[idx_va]
        train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
        val_map = {p: i for i, p in enumerate(val_pids)}
        sim = build_sim(y_tr, len(idx_va), val_pids, train_map, val_map)
        _, _, mixed_new, _, _ = full_pipe(0.65 * deep4 + 0.35 * mix, sim, y_tr, y_va, len(idx_va))
        _, _, mixed_base, _, _ = full_pipe(0.65 * deep4 + 0.35 * weak, sim, y_tr, y_va, len(idx_va))
        v_new, v_base = half_fold_oof(mixed_new, y_va), half_fold_oof(mixed_base, y_va)
        d = v_new - v_base
        d_mean, d_se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        ci_lo = d_mean - 1.96 * d_se
        verdict = ("过线，混合头换支进 I100 全量出提交流程" if (ci_lo > 0 or d_mean >= 0.001)
                   else "筛子过但融合未证实，E 刀关闭")
        print(f"  配对 ΔOOF(R=20): {d_mean:+.4f} ± {d_se:.4f} | CI95下界 {ci_lo:+.4f} -> {verdict}")
        results.append({"fused": {"alpha": alpha, "delta_mean": round(d_mean, 4),
                                  "ci95_lo": round(ci_lo, 4), "verdict": verdict}})
        conclusion = verdict
    else:
        conclusion = (f"最优 alpha={best['alpha']} microAUC={best['auc_micro']:.4f} 未过筛，"
                      "E 刀关闭（收敛头排序上限属结构性，PLM 侧三刀+混合全灭）")
        print(f"\n  无混合过筛 -> {conclusion}")

    report = {"experiment": "I109", "method": "head-blend sieve (35ep x 78ep)",
              "results": results, "bar": MICROAUC_BAR, "conclusion": conclusion,
              "elapsed_seconds": round(time.time() - t0, 1)}
    with Path(os.path.join(REL_DIR, "I109_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告: {os.path.join(REL_DIR, 'I109_report.json')} | {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
