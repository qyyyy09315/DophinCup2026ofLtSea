"""
I111a — 混合变体融合传导全景配对（找传导真峰，非 microAUC 峰）
===============================================================
背景:
  - I111 深化扫描 18/18 全过筛（混合家族稳健），但 microAUC 最优者（D5 三元 0.8862）
    融合配对 +0.0001（CI95 下界 −0.0000）反而不如 I109 alpha=0.25（+0.0002 显著）——
    **microAUC 是传导必要条件而非充分条件**（传导还取决于与 deep4/sim 的秩相关结构）。
  - 本刀对代表性变体做统一 R=20 配对融合检验，绘制「变体 → 融合传导」全景，
    找传导真峰（判据同前: CI95 下界 > 0 或 ΔOOF ≥ +0.001）。
变体: I109 a0.25(基准复现) / D1 a0.30 / D1 a0.35 / D2 logit a0.25 / D3 temp0.8 /
      D4 labelwise k20 / D5 ternary_eq / D5 ternary_02_04_04
输出: rel/I111a_report.json（纯后处理）
用法: python code/I/I111a-blend-fusion-panorama.py   (约 15 min)
"""

import os
import json
import time

import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
SEED = 42

i096 = SourceFileLoader("i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")).load_module()
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
i106 = SourceFileLoader("i106", os.path.join(PROJECT_ROOT, "code", "I", "I106-conv-multiseed-auc.py")).load_module()
full_pipe = i064.full_pipe
build_sim = i096.build_sim
half_fold_oof = i106.half_fold_oof


def logit(p):
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


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

    def lab_auc(p):
        from sklearn.metrics import roc_auc_score
        out = np.full(y_va.shape[1], 0.5)
        for j in range(y_va.shape[1]):
            col = y_va[:, j]
            if col.sum() > 0 and not col.all():
                out[j] = roc_auc_score(col, p[:, j])
        return out

    d_auc = lab_auc(conv) - lab_auc(weak)
    a_label = sigmoid(20.0 * d_auc)[None, :]

    mixes = {
        "I109_a0.25(基准)": 0.25 * conv + 0.75 * weak,
        "D1_a0.30": 0.30 * conv + 0.70 * weak,
        "D1_a0.35": 0.35 * conv + 0.65 * weak,
        "D2_logit_a0.25": sigmoid(0.25 * logit(conv) + 0.75 * logit(weak)),
        "D3_temp0.8": 0.25 * sigmoid(logit(conv) / 0.8) + 0.75 * weak,
        "D4_labelwise_k20": a_label * conv + (1 - a_label) * weak,
        "D5_ternary_eq": (conv + weak + conv650) / 3,
        "D5_ternary_020404": 0.2 * conv + 0.4 * weak + 0.4 * conv650,
    }

    print("=" * 88)
    print("  I111a: Blend Fusion Panorama (R=20 paired per variant)")
    print("=" * 88)
    print("[1/2] 构建 deep4/sim 与基线 mixed...")
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

    _, _, mixed_base, _, _ = full_pipe(0.65 * deep4 + 0.35 * weak, sim, y_tr, y_va, len(idx_va))
    v_base = half_fold_oof(mixed_base, y_va)

    print("[2/2] 逐变体 R=20 配对...")
    results = []
    best = None
    for name, mix in mixes.items():
        _, _, mixed_new, _, _ = full_pipe(0.65 * deep4 + 0.35 * mix, sim, y_tr, y_va, len(idx_va))
        v_new = half_fold_oof(mixed_new, y_va)
        d = v_new - v_base
        d_mean, d_se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        ci_lo = d_mean - 1.96 * d_se
        rec = {"name": name, "delta_mean": round(d_mean, 4), "delta_se": round(d_se, 4),
               "ci95_lo": round(ci_lo, 4), "p_gt0": round(float((d > 0).mean()), 3)}
        results.append(rec)
        mark = ""
        if best is None or rec["delta_mean"] > best["delta_mean"]:
            best = rec
            mark = " (*)"
        print(f"  {name:22s} | ΔOOF={d_mean:+.4f} ± {d_se:.4f} | CI95下界={ci_lo:+.4f} "
              f"| P(Δ>0)={rec['p_gt0']:.2f}{mark}", flush=True)

    sig = [r for r in results if r["ci95_lo"] > 0]
    if sig:
        top = max(sig, key=lambda r: r["delta_mean"])
        verdict = f"传导真峰 {top['name']}（ΔOOF {top['delta_mean']:+.4f}，CI95 下界 {top['ci95_lo']:+.4f}）"
    else:
        verdict = f"最优 {best['name']}（ΔOOF {best['delta_mean']:+.4f}）未显著，以 I109/I110 现状为准"
    report = {"experiment": "I111a", "method": "blend fusion panorama R=20 paired",
              "results": results, "best": best, "conclusion": verdict,
              "elapsed_seconds": round(time.time() - t0, 1)}
    with Path(os.path.join(REL_DIR, "I111a_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  结论: {verdict}")
    print(f"  报告: {os.path.join(REL_DIR, 'I111a_report.json')} | {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
