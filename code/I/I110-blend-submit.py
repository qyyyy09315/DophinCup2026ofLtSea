"""
I110 — 新旧两代头混合提交版（I109 过筛信号的平台检验）
============================================================
背景:
  - I109 E 刀: 两代头混合（0.25*78ep收敛三种子 + 0.75*35ep平滑三种子）microAUC 0.8856
    全队列首次过筛（>0.8783），融合配对 ΔOOF +0.0002（CI95 下界 +0.0001 显著为正）。
  - I105→I104 决策链先例: 统计显著微增益（+0.0008）平台兑现 +0.0089。本刀出混合头提交。
链路（I104 同款，仅 PLM 头换混合）:
  val 校准头 = 0.25*rel/i106_conv_multiseed_val + 0.75*rel/plm1664_multiseed_val
  test 头   = 0.25*I097 三种子全量78ep均值 + 0.75*I081 三种子全量35ep均值
  （test 侧两代头都用全量重训版：I081=35ep 全量、I097=78ep 全量——与 val 侧代际结构一致）
  sim 支 = top20 聚合 + 2-hop + kmer（I104 同款）；RRF+Sinkhorn+SCut 同款。
输出: submit/submission_I110_blend.csv (+桌面副本), rel/I110_report.json
判据: 平台提交分 > 45.9725（现锁 I104）
用法: python code/I/I110-blend-submit.py
"""

import os
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
SEED = 42
ALPHA = 0.25  # I109 网格最优

i096 = SourceFileLoader("i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")).load_module()
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
i085 = SourceFileLoader("i085", os.path.join(PROJECT_ROOT, "code", "I", "I085-top100-gated-agg.py")).load_module()
i104 = SourceFileLoader("i104", os.path.join(PROJECT_ROOT, "code", "I", "I104-top20sim-submit.py")).load_module()
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe
ranks, col_norm, sinkhorn = i064.ranks, i064.col_norm, i064.sinkhorn


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print(f"  I110: Blend-Head Platform Submission (alpha={ALPHA} conv + {1-ALPHA} weak)")
    print("=" * 88)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=SEED)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    full_map = {p: i for i, p in enumerate(train_df["protein_id"].values)}
    test_pids = test_df["protein_id"].values

    print("[1/5] 构建混合头（val 校准 + test 代际对齐）...")
    conv_v = np.load(os.path.join(REL_DIR, "i106_conv_multiseed_val.npz"))["val"]
    weak_v = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    blend_v = ALPHA * conv_v + (1 - ALPHA) * weak_v

    plm_te_i097 = i104.mlp_test_mean(device)          # 78ep 全量三种子
    plm_te_i081 = np.load(os.path.join(REL_DIR, "plm1664_fulltrain_test.npz"))["test"]  # 35ep 全量三种子
    blend_te = ALPHA * plm_te_i097 + (1 - ALPHA) * plm_te_i081
    print(f"  blend_val 均值 {blend_v.mean():.4f} | blend_test 均值 {blend_te.mean():.4f}")

    print("[2/5] sim 支: top20 聚合 + 2-hop + kmer...")
    idx100_v, bits100_v = i085.parse_topk(
        os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001_top100.tsv"), val_pids, train_map)
    idx100_t, bits100_t = i085.parse_topk(
        os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001_top100.tsv"), test_pids, full_map)
    sim1_v = i085.agg_topk(idx100_v, bits100_v, y_tr, 20)
    hit_v = idx100_v[:, 0] >= 0
    v2v = i104.parse_2hop(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv"), val_map, val_map)
    sim_v = 0.4 * i104.two_hop(sim1_v, hit_v, v2v, n_va) + 0.6 * i104.build_kmer(y_tr, "val")
    sim1_t = i085.agg_topk(idx100_t, bits100_t, y_full, 20)
    hit_t = idx100_t[:, 0] >= 0
    te_map = {p: i for i, p in enumerate(test_pids)}
    t2t = i104.parse_2hop(os.path.join(CK005A_TMP, "align_test_to_test_e0.001.tsv"), te_map, te_map)
    sim_t = 0.4 * i104.two_hop(sim1_t, hit_t, t2t, len(test_pids)) + 0.6 * i104.build_kmer(y_full, "test")

    print("[3/5] val 阈值管线（混合头口径）...")
    cnn_raw = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1_v = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2_v = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4_v = wc["t001a"] * t1_v + wc["t002"] * t2_v + wc["cnn"] * (0.5 * cnn_raw + 0.5 * c005_v)
    f1_val, _, _, opt_ths, s_v = full_pipe(0.65 * deep4_v + 0.35 * blend_v, sim_v, y_tr, y_va, n_va)
    t_tmp = search_scut(s_v, y_va)
    print(f"  val: F1={f1_val:.4f} (I109 混合口径)")

    print("[4/5] test 融合...")
    deep4_te = (wc["t001a"] * np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"]
                + wc["t002"] * np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
                + wc["cnn"] * (0.5 * np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
                               + 0.5 * np.load(os.path.join(REL_DIR, "C005_test_probs.npz"))["test"]))
    deep5_t = 0.65 * deep4_te + 0.35 * blend_te
    s_te = deep5_t + 0.75 * sim_t + 0.50 * col_norm(
        1.0 / (30 + ranks(deep5_t) + 1) + 1.0 / (30 + ranks(sim_t) + 1))
    row_m = np.maximum((s_te >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_full.mean(0) * len(s_te), 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s_te - s_te.max(axis=1, keepdims=True)) / 0.20)
    p_te = sinkhorn(ker, row_m, col_m)
    final = 0.40 * s_te + 0.60 * p_te
    y_pred = (final >= opt_ths[None, :]).astype(np.int8)

    print("[5/5] 合规校验与落盘...")
    sub = pd.concat([test_df[["protein_id"]].reset_index(drop=True),
                     pd.DataFrame(y_pred, columns=label_cols)], axis=1)
    assert sub.shape == (28450, 501), f"尺寸不匹配: {sub.shape}"
    assert set(np.unique(y_pred)).issubset({0, 1}), "非 0/1"
    out = os.path.join(SUB_DIR, "submission_I110_blend.csv")
    sub.to_csv(out, index=False)
    mb = os.path.getsize(out) / 1e6
    assert mb <= 50, f"超 50MB: {mb}"
    diffs = {}
    for ref_name in ("submission_I104_top20sim.csv", "submission_I103_deep4full.csv",
                     "submission_I097_fulltrain_80ep.csv"):
        ref = os.path.join(SUB_DIR, ref_name)
        if os.path.isfile(ref):
            old = pd.read_csv(ref)[label_cols].values.astype(np.int8)
            diff = int((old != y_pred).sum())
            diffs[ref_name] = diff
            print(f"  与 {ref_name} 差异: {diff} 格 ({diff / old.size:.3%})")
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if os.path.isdir(desktop):
        shutil.copy2(out, os.path.join(desktop, "submission_I110_blend.csv"))
        print("  桌面副本已放")

    report = {"experiment": "I110", "method": f"blend-head submission (alpha={ALPHA})",
              "val_f1": round(f1_val, 4),
              "cardinality": round(float(y_pred.sum(1).mean()), 2),
              "total_pos": int(y_pred.sum()), "file_mb": round(mb, 2),
              "diffs": diffs, "elapsed_seconds": round(time.time() - t0, 1)}
    with Path(os.path.join(REL_DIR, "I110_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 88)
    print(f"  I110 完成! 基数均值 {report['cardinality']} | {out} ({mb:.1f} MB)")
    print("=" * 88)


if __name__ == "__main__":
    main()
