"""
I016-disvote — CK005a / I011 两套 SCut 只在分歧格子上交或并
================================================================
依据:
  - Kittler et al., IEEE TPAMI 1998, doi:10.1109/34.667881
    分类器组合：和/积/投票。一致处不动，分歧处才组合。
  - Kuncheva, Combining Pattern Classifiers, Wiley 2004, doi:10.1002/0471660264
  - I011 改的是分数再 SCut；没试过「两套已阈值决策」的投票。
自己的改法:
  冻分数。CK005a 与 I011 各自 SCut。一致格子保持。
  分歧处：AND（都正才正）/ OR（任一正）/ 信 I011 / 信 CK005a。
  半折：阈值在拟合折搜，评估折套用后再组合。
  门槛相对 I011 半折 ≥ +0.001。
用法:
  python code/I/I016-disvote.py
输出:
  rel/I016_report.json
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
_spec = importlib.util.spec_from_file_location(
    "i011", os.path.join(PROJECT_ROOT, "code", "I", "I011-rrf.py"))
i011 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i011)

REPORT_PATH = os.path.join(i011.REL_DIR, "I016_report.json")
K, BETA = 30, 0.50


def scut_pred(ens, y, t=None):
    if t is None:
        t = i011.search_t(ens, y)
    return (ens >= t).astype(np.int8), t


def combine(a, b, mode):
    agree = a == b
    out = a.copy()
    if mode == "and":
        out[~agree] = a[~agree] & b[~agree]
    elif mode == "or":
        out[~agree] = a[~agree] | b[~agree]
    elif mode == "trust_i011":
        out[~agree] = b[~agree]
    elif mode == "trust_a":
        out[~agree] = a[~agree]
    return out


def pack(name, pred, y, hit, miss, extra=None):
    rec = {
        "name": name,
        "f1_100": round(i011.macro_f1(pred, y), 6),
        "f1_miss": round(i011.macro_f1(pred[miss], y[miss]), 6),
        "f1_hit": round(i011.macro_f1(pred[hit], y[hit]), 6),
        "pred_card": round(float(pred.sum(1).mean()), 3),
    }
    if extra:
        rec.update(extra)
    return rec


def main():
    t0 = time.time()
    print("=" * 60)
    print("I016: 两套 SCut 分歧投票")

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
    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1 - i011.ALIGN_W) * sim_kmer
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    ens_a = deep + i011.SIM_W * sim
    ens_b = ens_a + BETA * i011.col_norm(i011.rrf_pair(deep, sim, K))

    pred_a, _ = scut_pred(ens_a, y_val)
    pred_b, _ = scut_pred(ens_b, y_val)
    dis = pred_a != pred_b
    print(f"  分歧格子 {int(dis.sum())}/{dis.size} = {dis.mean():.4f}  "
          f"每样本 {dis.sum(1).mean():.2f}")

    rows = [
        pack("ck005a", pred_a, y_val, hit, miss),
        pack("i011", pred_b, y_val, hit, miss),
    ]
    print(f"  ck005a F1={rows[0]['f1_100']:.4f}  i011 F1={rows[1]['f1_100']:.4f}")

    print("[1/2] 全 val 组合...")
    modes = ("and", "or", "trust_i011", "trust_a")
    for mode in modes:
        pred = combine(pred_a, pred_b, mode)
        rec = pack(f"dis_{mode}", pred, y_val, hit, miss, {"mode": mode})
        rec["delta_i011"] = round(rec["f1_100"] - rows[1]["f1_100"], 6)
        rows.append(rec)
        print(f"  {rec['name']:18s} F1={rec['f1_100']:.4f} d={rec['delta_i011']:+.4f} "
              f"card={rec['pred_card']:.1f}")

    print("[2/2] 半折核对...")
    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pair():
        a = np.zeros_like(pred_a)
        b = np.zeros_like(pred_b)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            ta = i011.search_t(ens_a[fit], y_val[fit])
            tb = i011.search_t(ens_b[fit], y_val[fit])
            a[hold] = (ens_a[hold] >= ta).astype(np.int8)
            b[hold] = (ens_b[hold] >= tb).astype(np.int8)
        return a, b

    oa, ob = oof_pair()
    oof_rows = []
    oof_i011 = i011.macro_f1(ob, y_val)
    for name, pred in (
        ("oof_ck005a", oa),
        ("oof_i011", ob),
        ("oof_dis_and", combine(oa, ob, "and")),
        ("oof_dis_or", combine(oa, ob, "or")),
        ("oof_dis_trust_i011", combine(oa, ob, "trust_i011")),
        ("oof_dis_trust_a", combine(oa, ob, "trust_a")),
    ):
        rec = pack(name, pred, y_val, hit, miss)
        rec["delta_i011"] = round(rec["f1_100"] - oof_i011, 6)
        oof_rows.append(rec)
        print(f"  {name:22s} F1={rec['f1_100']:.4f} d={rec['delta_i011']:+.4f}")

    best_oof = max(oof_rows, key=lambda r: r["f1_100"])
    gain = best_oof["f1_100"] - oof_i011
    useful = bool(gain >= 0.001 and best_oof["name"] != "oof_i011")
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I011 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "disagree_frac": round(float(dis.mean()), 6),
        "disagree_per_row": round(float(dis.sum(1).mean()), 3),
        "baseline_i011": rows[1],
        "best_oof": best_oof,
        "gain_oof_vs_i011": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
