"""
A007-copy-gap — 强同源 C011a 与 DIAMOND top-1 拷贝的决策分歧（分析性，CPU）
================================================================
I048 分数混拷贝伤强切片。本脚本看 SCut 之后拷贝还漏了多少真标签、
会加进多少假标签。不写腾讯表。
若 copy_only_tp 明显大于 copy_only_fp，才值得开决策并集。
用法:
  python code/tools/A007-copy-gap.py
输出:
  rel/A007_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))
ck007 = load_mod("ck007", os.path.join("CK", "CK007-pident.py"))
REPORT_PATH = os.path.join(i011.REL_DIR, "A007_report.json")
C011A_ALPHA = 0.50
CNN_W = 0.2
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60


def fuse(deep, sim, y_train, y_val):
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep, sim, K_RRF))
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(s011, p, ALPHA_SK)


def main():
    t0 = time.time()
    print("=" * 60)
    print("A007: C011a vs top-1 copy decision gap on strong homology")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_d, bits_d, pid_v = ck007.parse_aln_pid(tsv, val_ids, train_map)
    hit = idx_d[:, 0] >= 0
    miss = ~hit
    pident = pid_v[:, 0].copy()
    pident[miss] = 0.0
    strong = hit & (pident >= 80)

    sim_bit = i011.agg_labels(idx_d, bits_d, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    cnn42 = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["val"]
    cnn_lock = (1.0 - C011A_ALPHA) * cnn42 + C011A_ALPHA * c005_val
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + CNN_W * cnn_lock
    s_lock = fuse(deep, sim, y_train, y_val)
    pred = (s_lock >= i011.search_t(s_lock, y_val)).astype(np.int8)
    copy = np.zeros_like(pred)
    top1 = idx_d[:, 0]
    ok = top1 >= 0
    copy[ok] = y_train[top1[ok]]

    yb = y_val[strong].astype(bool)
    pb = pred[strong].astype(bool)
    cb = copy[strong].astype(bool)
    copy_only = cb & ~pb
    fuse_only = pb & ~cb
    both = pb & cb
    n_cells = int(yb.size)
    rec = {
        "n_strong": int(strong.sum()),
        "copy_only_cells": int(copy_only.sum()),
        "copy_only_tp": int((copy_only & yb).sum()),
        "copy_only_fp": int((copy_only & ~yb).sum()),
        "fuse_only_cells": int(fuse_only.sum()),
        "fuse_only_tp": int((fuse_only & yb).sum()),
        "fuse_only_fn_if_and": int((fuse_only & yb).sum()),
        "both_tp": int((both & yb).sum()),
        "n_cells": n_cells,
        "copy_only_prec": None,
    }
    if rec["copy_only_cells"]:
        rec["copy_only_prec"] = round(rec["copy_only_tp"] / rec["copy_only_cells"], 4)
    rec["copy_only_tp_rate"] = round(rec["copy_only_tp"] / n_cells, 6)
    rec["copy_only_fp_rate"] = round(rec["copy_only_fp"] / n_cells, 6)
    worth = bool(rec["copy_only_tp"] > rec["copy_only_fp"] and rec["copy_only_tp"] >= 200)
    rec["worth_or"] = worth
    rec["seconds"] = round(time.time() - t0, 1)
    rec["conclusion"] = (
        "拷贝独有格真标签多于假，可开决策并集"
        if worth else "拷贝独有格假标签不少，决策并集价值低"
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"  strong={rec['n_strong']} copy_only tp={rec['copy_only_tp']} "
          f"fp={rec['copy_only_fp']} prec={rec['copy_only_prec']}")
    print(f"  fuse_only tp={rec['fuse_only_tp']}  worth_or={worth}")
    print(f"  报告 {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
