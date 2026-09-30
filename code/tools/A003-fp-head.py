"""
A003-fp-head — 低基数行假阳性是否集中在高频标签（分析性，CPU）
================================================================
I044：真基数截低基数行 +0.027，门控分不开该截谁。
A002：估 k MAE=14.4。本脚本改查列：低基数行多报的是不是头标签。
半折探针：只抬高频标签阈值（精度向），高基数行仍用全局 SCut。
不写腾讯表。若 OOF≥+0.001 再开 I045；否则关闭列侧低基数刀。
用法:
  python code/tools/A003-fp-head.py
输出:
  rel/A003_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "A003_report.json")
C011A_ALPHA = 0.50
CNN_W = 0.2
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
CARD_CUT = 10
HEAD_N = (50, 100, 150)
DELTA = (0.02, 0.05, 0.08, 0.12)


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


def main():
    t0 = time.time()
    print("=" * 60)
    print("A003: low-card FP on head labels (CPU, analytical)")

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
    card = y_val.sum(1)
    low = card <= CARD_CUT
    print(f"  low={int(low.sum())} miss={int(miss.sum())}")

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
    t_g = i011.search_t(score, y_val)
    pred0 = (score >= t_g).astype(np.int8)

    fp = (pred0 == 1) & (y_val == 0)
    extra = np.maximum(pred0.sum(1) - card, 0)
    freq = y_train.sum(0).astype(np.float64)
    order = np.argsort(-freq)
    fp_low = fp[low].sum(0)
    fp_high = fp[~low].sum(0)
    n_extra_low = int(fp[low].sum())
    share = {}
    for k in HEAD_N:
        share[f"head{k}"] = round(float(fp_low[order[:k]].sum() / max(n_extra_low, 1)), 4)
    print(f"  extra_low={n_extra_low}  head50={share['head50']}  "
          f"head100={share['head100']}  head150={share['head150']}")
    print(f"  extra_row_low_mean={float(extra[low].mean()):.2f}  "
          f"fp_low_top50={int(fp_low[order[:50]].sum())}")

    rec0 = pack(pred0, y_val, hit, miss, low, "c011a")
    print(f"  C011a F1={rec0['f1_100']:.4f} low={rec0['f1_low']:.4f} "
          f"pred_card_low={rec0['pred_card_low']:.1f}")

    rng = np.random.RandomState(i011.RANDOM_STATE)
    perm = rng.permutation(len(y_val))
    mid = len(y_val) // 2
    folds = [perm[:mid], perm[mid:]]
    oof_lock = np.zeros_like(pred0)
    oof_cfg = {(h, d): np.zeros_like(pred0) for h in HEAD_N for d in DELTA}
    for hold, fit in ((folds[0], folds[1]), (folds[1], folds[0])):
        t_fit = i011.search_t(score[fit], y_val[fit])
        oof_lock[hold] = (score[hold] >= t_fit).astype(np.int8)
        for h in HEAD_N:
            head = np.zeros(score.shape[1], dtype=bool)
            head[order[:h]] = True
            for d in DELTA:
                t2 = t_fit.copy()
                t2[head] = np.minimum(0.99, t_fit[head] + d)
                oof_cfg[(h, d)][hold] = (score[hold] >= t2).astype(np.int8)

    rec_lock = pack(oof_lock, y_val, hit, miss, low, "oof_c011a")
    print(f"  oof_c011a F1={rec_lock['f1_100']:.4f} low={rec_lock['f1_low']:.4f}")
    oof_rows = [rec_lock]
    for h in HEAD_N:
        for d in DELTA:
            rec = pack(
                oof_cfg[(h, d)], y_val, hit, miss, low,
                f"oof_h{h}_d{d:.2f}", {"head_n": h, "delta": d})
            rec["delta_c011a"] = round(rec["f1_100"] - rec_lock["f1_100"], 6)
            rec["delta_high"] = round(rec["f1_high"] - rec_lock["f1_high"], 6)
            oof_rows.append(rec)
            print(f"  {rec['name']:18s} F1={rec['f1_100']:.4f} d={rec['delta_c011a']:+.4f} "
                  f"low={rec['f1_low']:.4f} high={rec['f1_high']:.4f}")

    cand = [r for r in oof_rows if r["name"] != "oof_c011a"]
    best = max(cand, key=lambda r: r["f1_100"]) if cand else rec_lock
    gain = best["f1_100"] - rec_lock["f1_100"]
    useful = bool(gain >= 0.001 and best["f1_high"] + 1e-12 >= rec_lock["f1_high"] - 0.0005)
    print(f"  OOF 最优 {best['name']} vs C011a {gain:+.4f} → "
          f"{'可开 I045' if useful else '无增益'}")

    report = {
        "n_low": int(low.sum()),
        "n_extra_low": n_extra_low,
        "head_share_fp_low": share,
        "fp_low_sum": int(fp_low.sum()),
        "fp_high_sum": int(fp_high.sum()),
        "baseline_c011a": rec0,
        "oof_c011a": rec_lock,
        "best_oof": best,
        "gain_oof_vs_c011a": round(gain, 6),
        "useful": useful,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "头标签抬阈值 OOF≥+0.001，可开 I045"
            if useful else "低基数假阳性虽偏头标签，全局抬头阈值无增益；不单开 I045"
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
