"""
I045-twilight — 弱同源行压 sim 权重接到 C011a
================================================================
依据:
  - Rost, Protein Engineering 1999, doi:10.1093/protein/12.2.85
    twilight zone ~20–35% identity，功能传递不可靠。
  - DeepGOPlus: Kulmanov & Hoehndorf, Bioinformatics 2021,
    doi:10.1093/bioinformatics/btaa763。强 BLAST 走同源，弱时走序列模型。
  - A001：pident≥80 F1=0.5245；50–80 为 0.3595；<50 hit 为 0.2680。
  - I043 关的是 miss 换 CNN 一元；I005 关的是按长度压 sim。
    本实验按 top-1 pident 压 SIM_W，高一致度不动。
自己的改法:
  冻 deep / 邻居。sim' = g(p)·sim，g=1 当 pident≥T，否则 g∈{0,0.25,0.50}。
  T∈{50,70,80}。miss 的 pident=0。重建 C011a。
  门槛：半折 OOF 较 C011a ≥ +0.001。
用法:
  python code/I/I045-twilight.py
输出:
  rel/I045_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I045_report.json")
C011A_ALPHA = 0.50
CNN_W = 0.2
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
T_GRID = (50.0, 70.0, 80.0)
G_GRID = (0.0, 0.25, 0.50)


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


def pack_slices(rec, pred, y, hit, miss, twilight, remote):
    rec["f1_twilight"] = round(i011.macro_f1(pred[twilight], y[twilight]), 6) if twilight.any() else 0.0
    rec["f1_remote"] = round(i011.macro_f1(pred[remote], y[remote]), 6) if remote.any() else 0.0
    rec["f1_strong"] = round(i011.macro_f1(pred[hit & ~twilight & ~remote], y[hit & ~twilight & ~remote]), 6) if (hit & ~twilight & ~remote).any() else 0.0
    rec["n_twilight"] = int(twilight.sum())
    rec["n_remote"] = int(remote.sum())
    return rec


def main():
    t0 = time.time()
    print("=" * 60)
    print("I045: twilight pident gate on SIM_W")

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
    print(f"  miss={int(miss.sum())} twilight50-80={int(twilight.sum())} "
          f"remote<50={int(remote.sum())} strong={int((hit & (pident >= 80)).sum())}")

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
    rec0 = pack_slices(i011.eval_ens(s_lock, y_val, hit, miss, "c011a"),
                       pred0, y_val, hit, miss, twilight, remote)
    print(f"  C011a F1={rec0['f1_100']:.4f} tw={rec0['f1_twilight']:.4f} "
          f"remote={rec0['f1_remote']:.4f} miss={rec0['f1_miss']:.4f}")

    configs = {"c011a": s_lock}
    rows = [rec0]
    for tcut in T_GRID:
        for g in G_GRID:
            gate = np.where(pident >= tcut, 1.0, g).astype(np.float32)
            sim = sim_lock * gate[:, None]
            s = build_c011a(deep, sim, y_train, y_val)
            pred = (s >= i011.search_t(s, y_val)).astype(np.int8)
            name = f"t{tcut:.0f}_g{g:.2f}"
            rec = pack_slices(
                i011.eval_ens(s, y_val, hit, miss, name),
                pred, y_val, hit, miss, twilight, remote)
            rec["tcut"] = tcut
            rec["g"] = g
            rec["n_gated"] = int((pident < tcut).sum())
            rec["delta_c011a"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[name] = s
            print(f"  {name:12s} F1={rec['f1_100']:.4f} d={rec['delta_c011a']:+.4f} "
                  f"tw={rec['f1_twilight']:.4f} remote={rec['f1_remote']:.4f} "
                  f"gated={rec['n_gated']}")

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
            "f1_twilight": round(i011.macro_f1(pred[twilight], y_val[twilight]), 6) if twilight.any() else 0.0,
            "f1_remote": round(i011.macro_f1(pred[remote], y_val[remote]), 6) if remote.any() else 0.0,
        }
        oof_rows.append(rec)
        if name == "c011a":
            oof_lock = rec["f1_100"]
        print(f"  {rec['name']:20s} F1={rec['f1_100']:.4f} tw={rec['f1_twilight']:.4f} "
              f"remote={rec['f1_remote']:.4f}")
    for rec in oof_rows:
        rec["delta_c011a"] = round(rec["f1_100"] - oof_lock, 6)

    cand = [r for r in oof_rows if r["name"] != "oof_c011a"]
    best = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best["f1_100"] - oof_lock
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best['name']} vs C011a {gain:+.4f} → "
          f"{'有效' if useful else '无增益'}")

    report = {
        "n_twilight": int(twilight.sum()),
        "n_remote": int(remote.sum()),
        "baseline_c011a": rec0,
        "oof_c011a": oof_lock,
        "best_oof": best,
        "gain_oof_vs_c011a": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "twilight 压 sim OOF≥+0.001，可按 pident 调 SIM_W"
            if useful else "按 pident 压 sim 无增益，关闭；不重开 I005/I043"
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
