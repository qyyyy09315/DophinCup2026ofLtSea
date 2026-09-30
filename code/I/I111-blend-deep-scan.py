"""
I111 — 混合头深化扫描：细 alpha / logit 混合 / 逐标签自适应 / 三元混合
====================================================================
背景:
  - I109 突破: 两代头混合（0.25*78ep收敛 + 0.75*35ep平滑）microAUC 0.8856 过筛、
    融合配对显著为正，I110 已出片。但 alpha 网格仅 {0.25, 0.5, 0.75} 且最优在边界，
    以下维度未试:
  - D1 细 alpha 网格: {0.10~0.40} 找真峰；
  - D2 logit 空间混合: sigmoid(a*logit(conv) + (1-a)*logit(weak))（尾部排序形态不同）；
  - D3 温度对齐混合: conv 概率先温度锐化/软化 T ∈ {0.8~1.2} 再混合（两代锐度不同）；
  - D4 **逐标签自适应混合**（新想法）: 每标签按 val AUC 谁高偏谁——
    a_label = sigmoid(k*(AUC_conv_label - AUC_weak_label))，标签级偏差互补；
  - D5 三元混合: 加入 I107 concat2944 收敛头（microAUC 0.8715）三分量网格。
判据: microAUC > 0.8783 过筛 → R=20 配对融合检验（CI95 下界 > 0 或 ΔOOF ≥ +0.001 换支）。
输出: rel/I111_report.json（纯后处理）
用法: python code/I/I111-blend-deep-scan.py   (约 3-5 min)
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


def logit(p):
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def eval_head(mix, y_va):
    ths = search_scut(mix, y_va)
    return (round(float(macro_f1(mix, y_va, ths)), 4),
            round(float(roc_auc_score(y_va, mix, average="macro")), 4),
            round(float(roc_auc_score(y_va, mix, average="micro")), 4))


def main():
    t0 = time.time()
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=SEED)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]

    weak = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    conv = np.load(os.path.join(REL_DIR, "i106_conv_multiseed_val.npz"))["val"]
    conv650 = np.load(os.path.join(REL_DIR, "i107_concat2944_multiseed_val.npz"))["val"]

    print("=" * 88)
    print("  I111: Blend Deep-Scan (fine alpha / logit / temperature / per-label / ternary)")
    print("=" * 88)

    cands = []
    # D1 细 alpha
    for a in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
        cands.append((f"D1_prob_a{a:.2f}", a * conv + (1 - a) * weak))
    # D2 logit 混合
    lg_c, lg_w = logit(conv), logit(weak)
    for a in (0.15, 0.25, 0.35):
        cands.append((f"D2_logit_a{a:.2f}", sigmoid(a * lg_c + (1 - a) * lg_w)))
    # D3 温度对齐（对 conv 温度缩放后按 I109 alpha=0.25 混合）
    for T in (0.8, 0.9, 1.1, 1.2):
        conv_t = sigmoid(logit(conv) / T)
        cands.append((f"D3_temp{T:.1f}", 0.25 * conv_t + 0.75 * weak))
    # D4 逐标签自适应（k 控制锐度）
    auc_c = np.array([roc_auc_score(y_va[:, j], conv[:, j])
                      for j in range(y_va.shape[1]) if y_va[:, j].sum() > 0 and not y_va[:, j].all()])
    # 完整 500 维（无效标签置 0.5）
    def lab_auc(p):
        out = np.full(y_va.shape[1], 0.5)
        for j in range(y_va.shape[1]):
            col = y_va[:, j]
            if col.sum() > 0 and not col.all():
                out[j] = roc_auc_score(col, p[:, j])
        return out
    d_auc = lab_auc(conv) - lab_auc(weak)
    for k in (20.0, 50.0):
        a_label = sigmoid(k * d_auc)[None, :]
        cands.append((f"D4_labelwise_k{k:.0f}", a_label * conv + (1 - a_label) * weak))
    # D5 三元（等权与 0.2/0.4/0.4）
    cands.append(("D5_ternary_eq", (conv + weak + conv650) / 3))
    cands.append(("D5_ternary_02_04_04", 0.2 * conv + 0.4 * weak + 0.4 * conv650))

    print(f"{'变体':28s} | F1     | macroA | microA | 筛子")
    results = []
    best = None
    for name, mix in cands:
        f1, am, au = eval_head(mix, y_va)
        ok = au > MICROAUC_BAR
        results.append({"name": name, "f1": f1, "auc_macro": am, "auc_micro": au,
                        "sieve_pass": ok})
        if best is None or au > best["auc_micro"]:
            best = results[-1]
        print(f"  {name:26s} | {f1:.4f} | {am:.4f} | {au:.4f} | {'过' if ok else '--'}",
              flush=True)

    passed = [r for r in results if r["sieve_pass"]]
    print(f"\n过筛 {len(passed)}/{len(results)} | 最优: {best['name']} microAUC={best['auc_micro']}")
    fused = None
    if passed:
        top = max(passed, key=lambda r: r["auc_micro"])
        # 重建最优 mix
        mix_map = dict(cands)
        mix = mix_map[top["name"]]
        print(f"  最优者 {top['name']} 进融合配对检验（R=20）...")
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
        verdict = ("过线，换支出提交" if (ci_lo > 0 or d_mean >= 0.001) else "筛子过但融合未证实")
        print(f"  配对 ΔOOF(R=20): {d_mean:+.4f} ± {d_se:.4f} | CI95下界 {ci_lo:+.4f} -> {verdict}")
        fused = {"name": top["name"], "delta_mean": round(d_mean, 4),
                 "delta_se": round(d_se, 4), "ci95_lo": round(ci_lo, 4), "verdict": verdict}
        conclusion = f"最优 {top['name']}（microAUC {top['auc_micro']}）：{verdict}"
    else:
        conclusion = "全部变体未过筛"
        print("  无过筛者")

    report = {"experiment": "I111", "method": "blend deep-scan", "results": results,
              "best": best, "fused": fused, "bar": MICROAUC_BAR,
              "conclusion": conclusion, "elapsed_seconds": round(time.time() - t0, 1)}
    with Path(os.path.join(REL_DIR, "I111_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告: {os.path.join(REL_DIR, 'I111_report.json')} | {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
