"""
CK009-confw — 按 top-1 bitscore 调 ALIGN_W，再重建 I011/I019
================================================================
依据:
  - GOLabeler: You et al., Bioinformatics 2018, doi:10.1093/bioinformatics/bty130
    Naive BLAST 与其他序列组件按同源质量融合，不是全局固定权。
  - ProteInfer: Sanderson et al., eLife 2023, doi:10.7554/eLife.80942
    CNN 与 BLAST 互补；比对弱时更靠序列模型。
  - I005：按长度压 sim，负。本实验按同源置信度调的是 ALIGN_W，不是长度。
  - I010：逐标签 SIM_W，负。本实验是样本级，不是标签级。
  - CK007/CK008：砍邻居 / 改聚合，负。名单与加权平均不动。
  - 现状：miss 的 sim_bit≈0，但仍乘 (1-0.4)，k-mer 被压成 0.6。
自己的改法:
  冻 deep / 邻居 / 加权平均。hit：a_i = a_max · clip(b1/B, 0, 1)；
  miss：a_i=0。sim = a_i·sim_bit + (1-a_i)·sim_kmer，再走 I011/I019。
  B∈{50,100,200}，a_max∈{0.4,0.8,1.0}。
  另混 (1-γ)·I019 + γ·新 I019，γ∈{0.15,0.30}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/CK/CK009-confw.py
输出:
  rel/CK009_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "CK009_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
B_GRID = (50.0, 100.0, 200.0)
A_GRID = (0.4, 0.8, 1.0)
G_GRID = (0.15, 0.30)


def build_fusion(deep, sim, y_train, y_val):
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


def mix_sim(sim_bit, sim_kmer, bits, a_max, b_scale):
    b1 = np.maximum(bits[:, 0], 0.0).astype(np.float32)
    a = np.clip(b1 / b_scale, 0.0, 1.0) * a_max
    a = a[:, None]
    return a * sim_bit + (1.0 - a) * sim_kmer, a[:, 0]


def main():
    t0 = time.time()
    print("=" * 60)
    print("CK009: 同源置信度 ALIGN_W")

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
    b1 = bits_v[:, 0]
    print(f"  hit {int(hit.sum())}/{len(hit)}  top1 bits "
          f"med={float(np.median(b1[hit])):.1f} "
          f"p25={float(np.percentile(b1[hit], 25)):.1f} "
          f"p75={float(np.percentile(b1[hit], 75)):.1f}")

    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]

    sim0 = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    s019 = build_fusion(deep, sim0, y_train, y_val)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    rows = [rec0]
    configs = {"i019": s019}
    stats = {}

    print("[1/2] B × a_max × γ ...")
    for bsc in tqdm(B_GRID, desc="    B", unit="b"):
        for am in A_GRID:
            sim, a = mix_sim(sim_bit, sim_kmer, bits_v, am, bsc)
            stats[f"B{bsc:.0f}_a{am:.1f}"] = {
                "a_hit_mean": round(float(a[hit].mean()), 4),
                "a_miss_mean": round(float(a[miss].mean()) if miss.any() else 0.0, 4),
            }
            s_f = build_fusion(deep, sim, y_train, y_val)
            tag = f"B{bsc:.0f}_a{am:.1f}"
            rec_f = i011.eval_ens(s_f, y_val, hit, miss, f"filt_{tag}",
                                  {"B": bsc, "a_max": am})
            rec_f["delta_i019"] = round(rec_f["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec_f)
            configs[f"filt_{tag}"] = s_f
            print(f"  filt_{tag:16s} F1={rec_f['f1_100']:.4f} "
                  f"d={rec_f['delta_i019']:+.4f} a_hit={stats[tag]['a_hit_mean']}")
            s_n = i011.col_norm(s_f)
            for g in G_GRID:
                mixed = (1.0 - g) * s019 + g * s_n
                name = f"mix_{tag}_g{g:.2f}"
                rec = i011.eval_ens(mixed, y_val, hit, miss, name,
                                    {"B": bsc, "a_max": am, "gamma": g})
                rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
                rows.append(rec)
                configs[name] = mixed
                print(f"  {name:24s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

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
        print(f"  oof_{name:24s} F1={rec['f1_100']:.4f}")
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
        "best_oof": best_oof,
        "gain_oof_vs_i019": round(gain, 6),
        "useful": useful,
        "align_w_stats": stats,
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
