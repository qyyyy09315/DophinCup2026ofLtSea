"""
CK018-lcr — 低复杂度区 SEG 式过滤 DIAMOND 假同源
================================================================
依据:
  - Wootton & Federhen, Computers & Chemistry 1993,
    doi:10.1016/0097-8485(93)85006-x。局部复杂度统计，SEG 滤
    低复杂度区，避免 BLAST 把 poly-X / 无序区当同源。
  - Forslund & Sonnhammer, Bioinformatics 2009,
    doi:10.1093/bioinformatics/btp446。低复杂度过滤改变同源检测。
  - CK007 按 pident 砍邻居，负。本实验不按一致度砍，按查询
    序列复杂度降权：低复杂度查询的 bitscore 乘 (H/H0)^α。
自己的改法:
  冻名单与加权平均公式。窗口 w=12 的氨基酸香农熵 H，
  取序列中位窗口。w_i = bits · clip(H_i / H_ref, 0.2, 1)^α，
  α∈{1,2}。H_ref 用 train 中位。重建 I019。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/CK/CK018-lcr.py
输出:
  rel/CK018_report.json
"""

import importlib.util
import json
import math
import os
import time
from collections import Counter

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

REPORT_PATH = os.path.join(i011.REL_DIR, "CK018_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
A_GRID = (1.0, 2.0)
G_GRID = (0.15, 0.30)
WIN = 12


def shannon_win(seq, w=WIN):
    s = seq.upper()
    if len(s) < w:
        c = Counter(s)
        n = max(len(s), 1)
        return -sum((v / n) * math.log2(v / n) for v in c.values())
    hs = []
    for i in range(0, len(s) - w + 1, max(w // 2, 1)):
        win = s[i:i + w]
        c = Counter(win)
        hs.append(-sum((v / w) * math.log2(v / w) for v in c.values()))
    return float(np.median(hs)) if hs else 0.0


def build_fusion(deep, sim_bit, sim_kmer, y_train, y_val):
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep, sim, K_RRF))
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t_i011 = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t_i011).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(s011, p, ALPHA_SK)


def main():
    t0 = time.time()
    print("=" * 60)
    print("CK018: 低复杂度降权 DIAMOND")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    seqs = train_df["sequence"].values

    print("[0] 算 val 复杂度 ...")
    H = np.array([shannon_win(s) for s in tqdm(seqs[idx_val], desc="    H", unit="seq")],
                 np.float64)
    href = float(np.median(H))
    print(f"  H med={href:.3f} p10={np.percentile(H,10):.3f} p90={np.percentile(H,90):.3f}")

    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]

    s019 = build_fusion(deep, sim_bit, sim_kmer, y_train, y_val)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    rows = [rec0]
    configs = {"i019": s019}

    print("[1/2] α × γ ...")
    scale0 = np.clip(H / max(href, 1e-6), 0.2, 1.0)
    for a in tqdm(A_GRID, desc="    α", unit="a"):
        sc = np.power(scale0, a).astype(np.float32)
        bits_s = bits_v * sc[:, None]
        sim_s = i011.agg_labels(idx_v, bits_s, y_train)
        corr = float(np.corrcoef(sim_bit[hit].ravel(), sim_s[hit].ravel())[0, 1]) if hit.any() else 0.0
        s_s = build_fusion(deep, sim_s, sim_kmer, y_train, y_val)
        rec_s = i011.eval_ens(s_s, y_val, hit, miss, f"lcr_a{a:.1f}", {"alpha": a})
        rec_s["delta_i019"] = round(rec_s["f1_100"] - rec0["f1_100"], 6)
        rec_s["corr_bit"] = round(corr, 6)
        rows.append(rec_s)
        configs[f"lcr_a{a:.1f}"] = s_s
        print(f"  lcr_a{a:.1f} F1={rec_s['f1_100']:.4f} d={rec_s['delta_i019']:+.4f} "
              f"corr={corr:.3f}")
        sn = i011.col_norm(s_s)
        for g in G_GRID:
            mixed = (1.0 - g) * s019 + g * sn
            name = f"mix_a{a:.1f}_g{g:.2f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"alpha": a, "gamma": g})
            rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[name] = mixed
            print(f"  {name:16s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

    print("[2/2] 半折核对...")
    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            th = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= th).astype(np.int8)
        return pred

    oof_rows = []
    oof_i019 = None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "i019":
            oof_i019 = rec["f1_100"]
        print(f"  oof_{name:16s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    cand = [r for r in oof_rows if r["name"] != "oof_i019"]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline_i019": rec0,
        "H_median": round(href, 6),
        "best_oof": best_oof,
        "gain_oof_vs_i019": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(i011.REL_DIR, exist_ok=True)
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s")


if __name__ == "__main__":
    main()
