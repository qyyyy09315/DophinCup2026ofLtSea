"""
CK027-bitgap — bitscore 间隔大则抬 ALIGN_W，接到 C011a
================================================================
依据:
  - EIHR-PROT: Jonathan & Osalodion, Research Square 2026,
    doi:10.21203/rs.3.rs-10341355/v1。用 bitscore / 覆盖 / hit 数学
    同源可靠性门控。本实验不训 PLM，只用 top-1−top-2 间隔。
  - CK009：按 bitscore 绝对值调 ALIGN_W，OOF +0.0003，关闭。
    本实验用间隔，接到 C011a；miss 保持 a=0.4。
  - A009：间隔分布（同回合诊断）。
自己的改法:
  gap=bits1-bits2（无第二命中则 =bits1）。gap≥G 的 hit 行 a=a_hi，
  其余 a=0.4。G∈{50,100}，a_hi∈{0.55,0.70}。门槛半折 OOF +0.001。
用法:
  python code/CK/CK027-bitgap.py
输出:
  rel/CK027_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "CK027_report.json")
C011A_ALPHA = 0.50
CNN_W = 0.2
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
A_LOCK = 0.4
G_GRID = (50.0, 100.0)
A_HI_GRID = (0.55, 0.70)


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


def main():
    t0 = time.time()
    print("=" * 60)
    print("CK027: raise ALIGN_W when bitscore gap is large")

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
    pident = pid_v[:, 0].copy()
    pident[miss] = 0.0
    twilight = hit & (pident < 80.0) & (pident >= 50.0)
    remote = hit & (pident < 50.0)
    strong = hit & (pident >= 80.0)
    two = idx_v[:, 1] >= 0
    gap = np.where(two, bits_v[:, 0] - bits_v[:, 1], bits_v[:, 0])
    print(f"  hit={int(hit.sum())} two={int(two.sum())} "
          f"gap50={int((hit & (gap>=50)).sum())} gap100={int((hit & (gap>=100)).sum())}")

    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim_lock = A_LOCK * sim_bit + (1.0 - A_LOCK) * sim_kmer
    cnn42 = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["val"]
    cnn_lock = (1.0 - C011A_ALPHA) * cnn42 + C011A_ALPHA * c005_val
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + CNN_W * cnn_lock

    s_lock = fuse(deep, sim_lock, y_train, y_val)
    pred0 = (s_lock >= i011.search_t(s_lock, y_val)).astype(np.int8)
    rec0 = pack_slices(i011.eval_ens(s_lock, y_val, hit, miss, "c011a"),
                       pred0, y_val, twilight, remote, strong)
    print(f"  C011a F1={rec0['f1_100']:.4f} tw={rec0['f1_twilight']:.4f} "
          f"remote={rec0['f1_remote']:.4f} miss={rec0['f1_miss']:.4f}")

    configs = {"c011a": s_lock}
    rows = [rec0]
    for gcut in G_GRID:
        gated = hit & (gap >= gcut)
        for a_hi in A_HI_GRID:
            a = np.full(len(gap), A_LOCK, np.float32)
            a[gated] = a_hi
            sim = a[:, None] * sim_bit + (1.0 - a[:, None]) * sim_kmer
            s = fuse(deep, sim, y_train, y_val)
            pred = (s >= i011.search_t(s, y_val)).astype(np.int8)
            name = f"g{gcut:.0f}_a{a_hi:.2f}"
            rec = pack_slices(i011.eval_ens(s, y_val, hit, miss, name),
                              pred, y_val, twilight, remote, strong)
            rec.update({"gcut": gcut, "a_hi": a_hi, "n_gated": int(gated.sum()),
                        "delta_c011a": round(rec["f1_100"] - rec0["f1_100"], 6)})
            rows.append(rec)
            configs[name] = s
            print(f"  {name:12s} F1={rec['f1_100']:.4f} d={rec['delta_c011a']:+.4f} "
                  f"tw={rec['f1_twilight']:.4f} remote={rec['f1_remote']:.4f} "
                  f"n={rec['n_gated']}")

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

    oof_rows, oof_lock = [], None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
            "f1_twilight": round(i011.macro_f1(pred[twilight], y_val[twilight]), 6),
            "f1_remote": round(i011.macro_f1(pred[remote], y_val[remote]), 6),
        }
        oof_rows.append(rec)
        if name == "c011a":
            oof_lock = rec["f1_100"]
        print(f"  {rec['name']:20s} F1={rec['f1_100']:.4f}")
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
            "间隔抬 ALIGN_W OOF≥+0.001"
            if useful else "间隔抬 ALIGN_W 无增益，关闭；ALIGN_W 锁 0.4"
        ),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}  用时 {report['seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
