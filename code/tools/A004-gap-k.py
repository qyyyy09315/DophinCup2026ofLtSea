"""
A004-gap-k — 行内分数最大间隔估 k 再 RCut（分析性，CPU）
================================================================
I006/A002 用特征回归 k，MAE≥14。本脚本不回归：每条蛋白把 C011a
分数降序，在前 M 名里取相邻最大落差当 k_hat，再 RCut。
对照 Fan & Lin 2007 阈值选择、Ioannou 2010 分数切分。
不写腾讯表。门槛：k MAE<8 且半折 OOF 较 C011a ≥+0.001。
用法:
  python code/tools/A004-gap-k.py
输出:
  rel/A004_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

REPORT_PATH = os.path.join(i011.REL_DIR, "A004_report.json")
C011A_ALPHA = 0.50
CNN_W = 0.2
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
CARD_CUT = 10
M_GRID = (20, 40, 60, 80)


def build_c011a(deep, sim, y_train, y_val):
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


def gap_k(score, m, kmin=1, kmax=80):
    ranked = np.sort(score, axis=1)[:, ::-1]
    gaps = ranked[:, :m] - ranked[:, 1:m + 1]
    k = gaps.argmax(axis=1) + 1
    return np.clip(k, kmin, kmax)


def rcut(score, k):
    order = np.argsort(-score, axis=1)
    pred = np.zeros(score.shape, np.int8)
    idx = np.arange(score.shape[0])
    for j in range(int(k.max())):
        keep = k > j
        pred[idx[keep], order[keep, j]] = 1
    return pred


def pack(pred, y, hit, miss, low, name, extra=None):
    r, p = i011.macro_rp(pred, y)
    rec = {
        "name": name,
        "f1_100": round(i011.macro_f1(pred, y), 6),
        "recall": round(r, 6),
        "precision": round(p, 6),
        "f1_miss": round(i011.macro_f1(pred[miss], y[miss]), 6),
        "f1_hit": round(i011.macro_f1(pred[hit], y[hit]), 6),
        "f1_low": round(i011.macro_f1(pred[low], y[low]), 6),
        "f1_high": round(i011.macro_f1(pred[~low], y[~low]), 6),
        "pred_card": round(float(pred.sum(1).mean()), 3),
        "pred_card_low": round(float(pred[low].sum(1).mean()), 3),
    }
    if extra:
        rec.update(extra)
    return rec


def mae_pack(card, k, low):
    err = np.abs(card - k)
    return {
        "mae": round(float(err.mean()), 3),
        "mae_low": round(float(err[low].mean()), 3),
        "mae_high": round(float(err[~low].mean()), 3),
        "corr": round(float(np.corrcoef(card, k)[0, 1]), 3),
        "k_mean": round(float(k.mean()), 3),
        "k_mean_low": round(float(k[low].mean()), 3),
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("A004: largest-gap k then RCut (CPU, analytical)")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    card = y_val.sum(1).astype(np.int32)
    low = card <= CARD_CUT
    print(f"  low={int(low.sum())} miss={int(miss.sum())} card_mean={card.mean():.2f}")

    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
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
    score = build_c011a(deep, sim, y_train, y_val)

    rng = np.random.RandomState(i011.RANDOM_STATE)
    perm = rng.permutation(len(y_val))
    mid = len(y_val) // 2
    folds = [perm[:mid], perm[mid:]]
    oof_lock = np.zeros_like(score, dtype=np.int8)
    for hold, fit in ((folds[0], folds[1]), (folds[1], folds[0])):
        t_fit = i011.search_t(score[fit], y_val[fit])
        oof_lock[hold] = (score[hold] >= t_fit).astype(np.int8)
    rec_lock = pack(oof_lock, y_val, hit, miss, low, "oof_c011a")
    print(f"  oof_c011a F1={rec_lock['f1_100']:.4f} low={rec_lock['f1_low']:.4f} "
          f"card={rec_lock['pred_card']:.1f}")

    k_stats = {}
    oof_rows = [rec_lock]
    for m in M_GRID:
        k = gap_k(score, m)
        st = mae_pack(card, k, low)
        k_stats[f"m{m}"] = st
        print(f"  gap m={m:2d} MAE={st['mae']:.2f} low={st['mae_low']:.2f} "
              f"corr={st['corr']:.3f} k_low={st['k_mean_low']:.1f}")

        pred_rcut = rcut(score, k)
        rec = pack(pred_rcut, y_val, hit, miss, low, f"full_rcut_m{m}",
                   {"m": m, "kind": "rcut"})
        rec["delta_c011a"] = round(rec["f1_100"] - rec_lock["f1_100"], 6)
        oof_rows.append(rec)
        print(f"    full rcut F1={rec['f1_100']:.4f} d={rec['delta_c011a']:+.4f} "
              f"low={rec['f1_low']:.4f}")

        oof_clip = oof_lock.copy()
        scard = oof_lock.sum(1)
        mask = k < scard
        if mask.any():
            oof_clip[mask] = rcut(score[mask], k[mask])
        rec_c = pack(oof_clip, y_val, hit, miss, low, f"oof_clip_m{m}",
                     {"m": m, "kind": "clip", "n_clip": int(mask.sum())})
        rec_c["delta_c011a"] = round(rec_c["f1_100"] - rec_lock["f1_100"], 6)
        rec_c["delta_high"] = round(rec_c["f1_high"] - rec_lock["f1_high"], 6)
        oof_rows.append(rec_c)
        print(f"    oof clip  F1={rec_c['f1_100']:.4f} d={rec_c['delta_c011a']:+.4f} "
              f"low={rec_c['f1_low']:.4f} high={rec_c['f1_high']:.4f} "
              f"n_clip={int(mask.sum())}")

    cand = [r for r in oof_rows if r["name"].startswith("oof_clip_")]
    best = max(cand, key=lambda r: r["f1_100"]) if cand else rec_lock
    gain = best["f1_100"] - rec_lock["f1_100"]
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best['name']} vs C011a {gain:+.4f} → "
          f"{'有效' if useful else '无增益'}")

    report = {
        "n_low": int(low.sum()),
        "k_stats": k_stats,
        "oof_c011a": rec_lock,
        "best_oof": best,
        "gain_oof_vs_c011a": round(gain, 6),
        "useful": useful,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "最大间隔估 k OOF≥+0.001，可开 I046"
            if useful else "最大间隔估 k 无增益，关闭；不重开 I006 RCut"
        ),
    }
    os.makedirs(i011.REL_DIR, exist_ok=True)
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
