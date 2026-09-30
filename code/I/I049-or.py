"""
I049-or — 强同源行 SCut 与 DIAMOND top-1 拷贝做决策并集，接到 C011a
================================================================
依据:
  - I048：分数混拷贝伤强切片，OOF −0.0015。本实验不改分数，只在决策层
    把拷贝独有格并进去。
  - I016：全局两套 SCut 交并已关（分歧 0.98%）。本实验只强同源×拷贝。
  - A007：拷贝独有格真/假标签比（同回合诊断）。
  - Kittler et al., IEEE TPAMI 1998, doi:10.1109/34.667881
    一致处不动，分歧处才组合。
自己的改法:
  冻 C011a 分数与阈值。强行（pident≥80）pred' = pred ∨ copy。
  弱同源 / miss 不动。门槛半折 OOF +0.001（阈值在拟合折搜，评估折并）。
用法:
  python code/I/I049-or.py
输出:
  rel/I049_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

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

REPORT_PATH = os.path.join(i011.REL_DIR, "I049_report.json")
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


def pack_slices(rec, pred, y, twilight, remote, strong):
    rec["f1_twilight"] = round(i011.macro_f1(pred[twilight], y[twilight]), 6) if twilight.any() else 0.0
    rec["f1_remote"] = round(i011.macro_f1(pred[remote], y[remote]), 6) if remote.any() else 0.0
    rec["f1_strong"] = round(i011.macro_f1(pred[strong], y[strong]), 6) if strong.any() else 0.0
    return rec


def apply_or(pred, copy, strong):
    out = pred.copy()
    out[strong] = np.maximum(pred[strong], copy[strong])
    return out


def main():
    t0 = time.time()
    print("=" * 60)
    print("I049: strong-homology SCut OR top-1 copy")

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
    twilight = hit & (pident < 80) & (pident >= 50)
    remote = hit & (pident < 50)
    strong = hit & (pident >= 80)
    print(f"  strong={int(strong.sum())}")

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

    copy = np.zeros_like(s_lock, dtype=np.int8)
    top1 = idx_d[:, 0]
    ok = top1 >= 0
    copy[ok] = y_train[top1[ok]]

    t = i011.search_t(s_lock, y_val)
    pred0 = (s_lock >= t).astype(np.int8)
    pred_or = apply_or(pred0, copy, strong)
    rec0 = pack_slices(i011.eval_ens(s_lock, y_val, hit, miss, "c011a"),
                       pred0, y_val, twilight, remote, strong)
    rec1 = pack_slices({
        "name": "or_strong",
        "f1_100": round(i011.macro_f1(pred_or, y_val), 6),
        "recall": round(i011.macro_rp(pred_or, y_val)[0], 6),
        "precision": round(i011.macro_rp(pred_or, y_val)[1], 6),
        "auc": rec0["auc"],
        "f1_miss": round(i011.macro_f1(pred_or[miss], y_val[miss]), 6),
        "f1_hit": round(i011.macro_f1(pred_or[hit], y_val[hit]), 6),
        "pred_card": round(float(pred_or.sum(1).mean()), 3),
        "delta_c011a": 0.0,
    }, pred_or, y_val, twilight, remote, strong)
    rec1["delta_c011a"] = round(rec1["f1_100"] - rec0["f1_100"], 6)
    print(f"  C011a F1={rec0['f1_100']:.4f} strong={rec0['f1_strong']:.4f}")
    print(f"  or     F1={rec1['f1_100']:.4f} d={rec1['delta_c011a']:+.4f} "
          f"strong={rec1['f1_strong']:.4f}")

    rng = np.random.RandomState(i011.RANDOM_STATE)
    perm = rng.permutation(len(y_val))
    mid = len(y_val) // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_lock():
        pred = np.zeros_like(pred0)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            th = i011.search_t(s_lock[fit], y_val[fit])
            pred[hold] = (s_lock[hold] >= th).astype(np.int8)
        return pred

    def oof_or():
        pred = np.zeros_like(pred0)
        for i, hold in enumerate(tqdm(folds, desc="    OOF", unit="fold")):
            fit = folds[1 - i]
            th = i011.search_t(s_lock[fit], y_val[fit])
            p = (s_lock[hold] >= th).astype(np.int8)
            pred[hold] = apply_or(p, copy[hold], strong[hold])
        return pred

    p_lock = oof_lock()
    p_or = oof_or()
    oof0 = round(i011.macro_f1(p_lock, y_val), 6)
    oof1 = round(i011.macro_f1(p_or, y_val), 6)
    gain = oof1 - oof0
    useful = bool(gain >= 0.001)
    print(f"  OOF lock={oof0:.4f} or={oof1:.4f} d={gain:+.4f} → "
          f"{'有效' if useful else '无增益'}")
    report = {
        "baseline_c011a": rec0,
        "full_or": rec1,
        "oof_c011a": oof0,
        "oof_or": oof1,
        "gain_oof_vs_c011a": round(gain, 6),
        "useful": useful,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "强同源决策并集 OOF≥+0.001"
            if useful else "强同源决策并集无增益，关闭；不重开 I048/I016"
        ),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}  用时 {report['seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
