"""
I037-pseudo — 高置信 hit 伪标签经 val–val 图回传到 miss
================================================================
依据:
  - Yarowsky, ACL 1995, doi:10.3115/981658.981684
    self-training：高置信预测当伪标签，再扩到未标注。
  - Amini et al., arXiv:2202.12040。self-training 综述。
  - I019 miss F1≈0.075。ECK008 在 val 图上平滑分数，负。
    本实验不平滑分数，只把 hit 的硬标签传到 miss。
  - I006 miss 基数补满负；本实验补的是标签，不是基数。
自己的改法:
  冻 I019。半折 SCut。seed = hit ∧ (行 max≥τ) 的正标签。
  用 ECK008 val–val k6：miss 从邻居 seed 做 sim² 加权。
  ens' = I019；仅 miss 行 ens' += γ · col_norm(pseudo)。
  τ∈{0.6,0.8}，γ∈{0.15,0.30}。门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I037-pseudo.py
输出:
  rel/I037_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I037_report.json")
VAL_NEIGH = os.path.join(i011.REL_DIR, "ECK008_neigh_val_k6.npz")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
T_GRID = (0.60, 0.80)
G_GRID = (0.15, 0.30)
TOP = 5


def build_i019(train_df, y_full, idx_train, idx_val):
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
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
    return i019.mix(s011, p, ALPHA_SK), hit, y_train, y_val


def pseudo_from(score, pred, hit, idx, sim, tau):
    n, l = score.shape
    seed = np.zeros((n, l), np.float32)
    mask = hit & (score.max(1) >= tau)
    seed[mask] = pred[mask].astype(np.float32)
    out = np.zeros((n, l), np.float32)
    for i in range(n):
        if hit[i]:
            continue
        nb = idx[i, :TOP]
        ww = np.maximum(sim[i, :TOP], 1e-7) ** 2
        lab = seed[nb]
        if lab.sum() <= 0:
            continue
        out[i] = (lab * ww[:, None]).sum(0) / ww.sum()
    return out, int(mask.sum())


def main():
    t0 = time.time()
    print("=" * 60)
    print("I037: hit 伪标签回传到 miss")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s019, hit, y_train, y_val = build_i019(train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f} missF1={rec0['f1_miss']:.4f}")

    t_all = i011.search_t(s019, y_val)
    pred_all = (s019 >= t_all).astype(np.int8)
    d = np.load(VAL_NEIGH)
    idx, sim = d["idx"], d["sim"]
    if idx.shape[0] != len(y_val):
        raise RuntimeError(f"ECK008 n={idx.shape[0]} vs val n={len(y_val)}")

    rows = [rec0]
    configs = {"i019": s019}
    stats = {}

    print("[1/2] τ × γ ...")
    for tau in tqdm(T_GRID, desc="    τ", unit="t"):
        pseudo, n_seed = pseudo_from(s019, pred_all, hit, idx, sim, tau)
        nz = int((pseudo.sum(1) > 0).sum())
        stats[f"t{tau:.2f}"] = {"n_seed": n_seed, "n_miss_filled": nz}
        print(f"  τ={tau:.2f} seed={n_seed} miss填 {nz}/{int(miss.sum())}")
        if nz == 0:
            continue
        p_n = i011.col_norm(pseudo)
        rec_r = i011.eval_ens(pseudo, y_val, hit, miss, f"ps_t{tau:.2f}")
        rec_r["delta_i019"] = round(rec_r["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec_r)
        for g in G_GRID:
            mixed = s019.copy()
            mixed[miss] = mixed[miss] + g * p_n[miss]
            name = f"mix_t{tau:.2f}_g{g:.2f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"tau": tau, "gamma": g})
            rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[name] = mixed
            print(f"  {name:20s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f} "
                  f"miss={rec['f1_miss']:.4f}")

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
        print(f"  oof_{name:20s} F1={rec['f1_100']:.4f} miss={rec['f1_miss']:.4f}")
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
        "seed_stats": stats,
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
