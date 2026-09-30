"""
C011-csraavg — C002b 与 C005 概率平均替换 CNN 路
================================================================
依据:
  - Lakshminarayanan, Pritzel & Blundell, NeurIPS 2017,
    arXiv:1612.01474。独立训练网络平均，多样性来自不同损失盆地。
  - C005 是 CSRA 头（Zhu & Wu, ICCV 2021），独立 F1(0.5)=0.2620
    未超 C002b 0.2714，整路替换进四路只 +0.0007，已关。
  - 本实验不换头、不重训：两路已缓存概率算术平均后接到 I019。
    与 C002b 相关 0.75（C002c/C009/C010 是 0.99），不是同一次前向。
自己的改法:
  cnn' = (1-α)·C002b + α·C005，α∈{0.30,0.50}。重建 I019。
  独立对照 C002b F1(0.5)。融合门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/C/C011-csraavg.py
输出:
  rel/C011_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "C011_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
A_GRID = (0.30, 0.50)
CNN_W = 0.2


def i019_from_cnn(cnn, deep_rest, sim, y_train, y_val):
    deep = deep_rest + CNN_W * cnn
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


def f1_05(p, y):
    return i011.macro_f1((p >= 0.5).astype(np.int8), y)


def main():
    t0 = time.time()
    print("=" * 60)
    print("C011: C002b×C005 概率平均")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    cnn42 = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["val"]
    y5 = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["y_true"]
    if y5.shape != y_val.shape or not np.array_equal(y5, y_val):
        raise RuntimeError("C005 y_true 与当前切分不对齐")

    f1_b = f1_05(cnn42, y_val)
    f1_5 = f1_05(c005, y_val)
    corr = float(np.corrcoef(cnn42.ravel(), c005.ravel())[0, 1])
    print(f"  C002b F1(0.5)={f1_b:.4f}  C005 F1(0.5)={f1_5:.4f}  corr={corr:.3f}")

    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep_rest = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"]

    s019 = i019_from_cnn(cnn42, deep_rest, sim, y_train, y_val)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    rows = [rec0]
    configs = {"i019": s019}
    print("[1/1] α 平均 ...")
    for a in A_GRID:
        mix_cnn = (1.0 - a) * cnn42 + a * c005
        f1a = f1_05(mix_cnn, y_val)
        s = i019_from_cnn(mix_cnn, deep_rest, sim, y_train, y_val)
        name = f"avg_a{a:.2f}"
        rec = i011.eval_ens(s, y_val, hit, miss, name, {"alpha": a, "f1_05": round(f1a, 6)})
        rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        configs[name] = s
        print(f"  {name:12s} F1(0.5)={f1a:.4f} F1(100)={rec['f1_100']:.4f} "
              f"d={rec['delta_i019']:+.4f}")

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
        print(f"  oof_{name:12s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    cand = [r for r in oof_rows if r["name"] != "oof_i019"]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "f1_05_c002b": round(f1_b, 6),
        "f1_05_c005": round(f1_5, 6),
        "corr_c002b_c005": round(corr, 6),
        "baseline_i019": rec0,
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
