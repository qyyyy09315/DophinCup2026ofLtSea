"""
CK020-s2 — DIAMOND 邻居 bitscore×pident 重加权接到 C011a
================================================================
依据:
  - Zhang & Freddolino, Brief Bioinform 2024, doi:10.1093/bib/bbae349
    S2：模板权 = bitscore × 序列一致度，BLAST 上优于纯 bitscore。
  - CK007：按 pident 门槛丢掉邻居，OOF 无增益，关的是集合不是权重。
  - CK016：−log(E) 权 OOF +0.0002，比对权锁 bitscore。
自己的改法:
  冻 deep / k-mer / 邻居名单。只改 DIAMOND 聚合权：
    w = bitscore × (pident/100)^α，α∈{0.5,1.0,2.0}。
  重建 C011a。门槛：半折 OOF 较 C011a ≥ +0.001。
用法:
  python code/CK/CK020-s2.py
输出:
  rel/CK020_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

REPORT_PATH = os.path.join(i011.REL_DIR, "CK020_report.json")
C011A_ALPHA = 0.50
CNN_W = 0.2
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
A_GRID = (0.5, 1.0, 2.0)


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


def pack_extra(rec, pred, y, low):
    rec["f1_low"] = round(i011.macro_f1(pred[low], y[low]), 6)
    rec["f1_high"] = round(i011.macro_f1(pred[~low], y[~low]), 6)
    rec["pred_card_low"] = round(float(pred[low].sum(1).mean()), 3)
    return rec


def main():
    t0 = time.time()
    print("=" * 60)
    print("CK020: bitscore × pident^α into C011a")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v, pid_v = ck007.parse_aln_pid(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    low = y_val.sum(1) <= 10
    print(f"  miss={int(miss.sum())} low={int(low.sum())} "
          f"pident_hit_mean={float(pid_v[hit, 0].mean()):.1f}")

    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim_lock = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer

    cnn42 = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["val"]
    cnn_lock = (1.0 - C011A_ALPHA) * cnn42 + C011A_ALPHA * c005_val
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + CNN_W * cnn_lock

    s_lock = build_c011a(deep, sim_lock, y_train, y_val)
    pred0 = (s_lock >= i011.search_t(s_lock, y_val)).astype(np.int8)
    rec0 = pack_extra(i011.eval_ens(s_lock, y_val, hit, miss, "c011a"),
                      pred0, y_val, low)
    print(f"  C011a F1={rec0['f1_100']:.4f} miss={rec0['f1_miss']:.4f} "
          f"low={rec0['f1_low']:.4f}")

    configs = {"c011a": s_lock}
    rows = [rec0]
    pid01 = np.clip(pid_v / 100.0, 0.0, 1.0)
    for a in A_GRID:
        ww = bits_v * (pid01 ** a)
        sim_s2 = i011.agg_labels(idx_v, ww, y_train)
        sim = i011.ALIGN_W * sim_s2 + (1.0 - i011.ALIGN_W) * sim_kmer
        s = build_c011a(deep, sim, y_train, y_val)
        pred = (s >= i011.search_t(s, y_val)).astype(np.int8)
        rec = pack_extra(
            i011.eval_ens(s, y_val, hit, miss, f"s2_a{a:.1f}"),
            pred, y_val, low)
        rec["alpha"] = a
        rec["delta_c011a"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        configs[rec["name"]] = s
        print(f"  s2 α={a:.1f} F1={rec['f1_100']:.4f} d={rec['delta_c011a']:+.4f} "
              f"miss={rec['f1_miss']:.4f} low={rec['f1_low']:.4f}")

    rng = np.random.RandomState(i011.RANDOM_STATE)
    perm = rng.permutation(len(y_val))
    mid = len(y_val) // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            th = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= th).astype(np.int8)
        return pred

    oof_rows = []
    oof_lock = None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
            "f1_low": round(i011.macro_f1(pred[low], y_val[low]), 6),
            "f1_high": round(i011.macro_f1(pred[~low], y_val[~low]), 6),
        }
        oof_rows.append(rec)
        if name == "c011a":
            oof_lock = rec["f1_100"]
        print(f"  {rec['name']:16s} F1={rec['f1_100']:.4f} miss={rec['f1_miss']:.4f} "
              f"low={rec['f1_low']:.4f}")
    for rec in oof_rows:
        rec["delta_c011a"] = round(rec["f1_100"] - oof_lock, 6)

    cand = [r for r in oof_rows if r["name"] != "oof_c011a"]
    best = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best["f1_100"] - oof_lock
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best['name']} vs C011a {gain:+.4f} → "
          f"{'有效' if useful else '无增益'}")

    report = {
        "baseline_c011a": rec0,
        "oof_c011a": oof_lock,
        "best_oof": best,
        "gain_oof_vs_c011a": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "S2 重加权 OOF≥+0.001，可替换 DIAMOND bitscore 权"
            if useful else "bitscore×pident 无增益，关闭；比对权锁 bitscore"
        ),
    }
    os.makedirs(i011.REL_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
