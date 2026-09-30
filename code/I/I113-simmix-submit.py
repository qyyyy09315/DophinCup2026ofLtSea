from pathlib import Path
"""
I113 — sim 代际混合 + 混合头提交版（I112 收官信号出片）
============================================================
背景:
  - I112 网格: sim 代际混合（0.75×top20 + 0.25×top3）是最大杠杆——
    ΔOOF +0.0009（CI95 下界 +0.0008，P(Δ>0)=1.00，全项目最强传导信号），
    且与头选择/权重无关（弱头同样 +0.0007）；混合头 × sim 混合 = 最优组合
    B_mix_b0.75_w0.35。
  - 本刀出提交：I110 链路（混合头 alpha=0.25，w=0.35）+ sim 两代混合
    （beta=0.75：0.75×top20 + 0.25×top3，val/test 同口径）。
输出: submit/submission_I113_simmix.csv (+桌面副本), rel/I113_report.json
判据: 平台提交分 > 45.9725（现锁 I104）
用法: python code/I/I113-simmix-submit.py
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
SEED = 42
ALPHA = 0.25   # 混合头（I109 最优）
BETA = 0.75    # sim 代际混合（I112 最优）

i096 = SourceFileLoader("i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")).load_module()
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
i085 = SourceFileLoader("i085", os.path.join(PROJECT_ROOT, "code", "I", "I085-top100-gated-agg.py")).load_module()
i104 = SourceFileLoader("i104", os.path.join(PROJECT_ROOT, "code", "I", "I104-top20sim-submit.py")).load_module()
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe
ranks, col_norm, sinkhorn = i064.ranks, i064.col_norm, i064.sinkhorn


def sim_mix_align(idx100, bits100, y_src, nbr, n):
    """代际混合比对支：beta*top20 + (1-beta)*top3（2-hop 后混合）"""
    s20 = i085.agg_topk(idx100, bits100, y_src, 20)
    s3 = i085.agg_topk(idx100, bits100, y_src, 3)
    hit = idx100[:, 0] >= 0
    s20 = i104.two_hop(s20, hit, nbr, n)
    s3 = i104.two_hop(s3, hit, nbr, n)
    return BETA * s20 + (1 - BETA) * s3


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print(f"  I113: Sim-Generational-Mix + Blend-Head Submission (alpha={ALPHA}, beta={BETA})")
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

    print("[1/5] 混合头（val 校准 + test 代际对齐）...")
    conv_v = np.load(os.path.join(REL_DIR, "i106_conv_multiseed_val.npz"))["val"]
    weak_v = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    blend_v = ALPHA * conv_v + (1 - ALPHA) * weak_v
    plm_te_i097 = i104.mlp_test_mean(device)
    plm_te_i081 = np.load(os.path.join(REL_DIR, "plm1664_fulltrain_test.npz"))["test"]
    blend_te = ALPHA * plm_te_i097 + (1 - ALPHA) * plm_te_i081

    print("[2/5] sim 代际混合（val/test 同口径）...")
    idx100_v, bits100_v = i085.parse_topk(
        os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001_top100.tsv"), val_pids, train_map)
    idx100_t, bits100_t = i085.parse_topk(
        os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001_top100.tsv"), test_pids, full_map)
    v2v = i104.parse_2hop(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv"), val_map, val_map)
    te_map = {p: i for i, p in enumerate(test_pids)}
    t2t = i104.parse_2hop(os.path.join(CK005A_TMP, "align_test_to_test_e0.001.tsv"), te_map, te_map)
    sim_v = 0.4 * sim_mix_align(idx100_v, bits100_v, y_tr, v2v, n_va) + 0.6 * i104.build_kmer(y_tr, "val")
    sim_t = 0.4 * sim_mix_align(idx100_t, bits100_t, y_full, t2t, len(test_pids)) + 0.6 * i104.build_kmer(y_full, "test")

    print("[3/5] val 阈值管线（I112 最优口径）...")
    cnn_raw = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1_v = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2_v = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4_v = wc["t001a"] * t1_v + wc["t002"] * t2_v + wc["cnn"] * (0.5 * cnn_raw + 0.5 * c005_v)
    f1_val, _, _, opt_ths, s_v = full_pipe(0.65 * deep4_v + 0.35 * blend_v, sim_v, y_tr, y_va, n_va)
    t_tmp = search_scut(s_v, y_va)
    print(f"  val: F1={f1_val:.4f}")

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
    out = os.path.join(SUB_DIR, "submission_I113_simmix.csv")
    sub.to_csv(out, index=False)
    mb = os.path.getsize(out) / 1e6
    assert mb <= 50, f"超 50MB: {mb}"
    for ref_name in ("submission_I110_blend.csv", "submission_I104_top20sim.csv"):
        ref = os.path.join(SUB_DIR, ref_name)
        if os.path.isfile(ref):
            old = pd.read_csv(ref)[label_cols].values.astype(np.int8)
            diff = int((old != y_pred).sum())
            print(f"  与 {ref_name} 差异: {diff} 格 ({diff / old.size:.3%})")
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if os.path.isdir(desktop):
        shutil.copy2(out, os.path.join(desktop, "submission_I113_simmix.csv"))
        print("  桌面副本已放")

    report = {"experiment": "I113", "method": f"sim-mix(beta={BETA}) + blend-head(alpha={ALPHA}) submission",
              "val_f1": round(f1_val, 4),
              "cardinality": round(float(y_pred.sum(1).mean()), 2),
              "total_pos": int(y_pred.sum()), "file_mb": round(mb, 2),
              "elapsed_seconds": round(time.time() - t0, 1)}
    with Path(REL_DIR, "I113_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 88)
    print(f"  I113 完成! 基数均值 {report['cardinality']} | {out} ({mb:.1f} MB)")
    print("=" * 88)


if __name__ == "__main__":
    main()
