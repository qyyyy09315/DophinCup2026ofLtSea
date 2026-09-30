"""
CK014-bitmnz — DIAMOND 邻居共识：bitscore 加权和 × 计数^α
================================================================
依据:
  - Fox & Shaw, TREC-2 1994。CombMNZ：分数和再乘命中数。
  - Lee, SIGIR 1997, doi:10.1145/258525.258587。多证据组合，
    共识项对稀疏证据更稳。
  - CK013：k-mer 上 MNZ 独立 0.375，计数放大常见标签。
    DIAMOND 邻居更稀（top-3），不一定同样炸。
  - CK008：max+λ·count 是加性，负。本实验是乘性。
自己的改法:
  冻 deep / k-mer / 邻居名单。比对路：
    s = Σ bits·y ； c = Σ y ； s' = s · c^α
  列归一后与 k-mer 按 ALIGN_W 混合，重建 I011/I019。
  α∈{0.5,1.0}。另混 (1-γ)·I019 + γ·新 I019，γ∈{0.15,0.30}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/CK/CK014-bitmnz.py
输出:
  rel/CK014_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "CK014_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
TOP_K = 3
A_GRID = (0.5, 1.0)
G_GRID = (0.15, 0.30)


def bit_mnz(idx, bits, y_train, alpha, top_k=TOP_K):
    n = idx.shape[0]
    acc = np.zeros((n, y_train.shape[1]), np.float32)
    cnt = np.zeros_like(acc)
    for j in range(top_k):
        v = idx[:, j] >= 0
        lab = y_train[idx[:, j].clip(min=0)].astype(np.float32)
        lab[~v] = 0.0
        w = np.maximum(bits[:, j:j + 1], 0.0).astype(np.float32)
        acc = acc + lab * w
        cnt = cnt + lab
    return acc * np.power(np.maximum(cnt, 1.0), alpha)


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
    print("CK014: DIAMOND 邻居 CombMNZ")

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
    print(f"  hit {int(hit.sum())}/{len(hit)}")

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
    corr = {}

    print("[1/2] α × γ ...")
    for a in tqdm(A_GRID, desc="    α", unit="a"):
        raw = bit_mnz(idx_v, bits_v, y_train, a)
        sim_q = i011.col_norm(raw)
        c = float(np.corrcoef(sim_q.ravel(), sim_bit.ravel())[0, 1])
        corr[f"a{a:.1f}"] = round(c, 4)
        s_f = build_fusion(deep, sim_q, sim_kmer, y_train, y_val)
        rec_f = i011.eval_ens(s_f, y_val, hit, miss, f"filt_a{a:.1f}", {"alpha": a})
        rec_f["delta_i019"] = round(rec_f["f1_100"] - rec0["f1_100"], 6)
        rec_f["corr_bit"] = corr[f"a{a:.1f}"]
        rows.append(rec_f)
        configs[f"filt_a{a:.1f}"] = s_f
        print(f"  filt_a{a:.1f} F1={rec_f['f1_100']:.4f} d={rec_f['delta_i019']:+.4f} "
              f"corr={c:.3f}")
        s_n = i011.col_norm(s_f)
        for g in G_GRID:
            mixed = (1.0 - g) * s019 + g * s_n
            name = f"mix_a{a:.1f}_g{g:.2f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"alpha": a, "gamma": g})
            rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[name] = mixed
            print(f"  {name:20s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

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
        print(f"  oof_{name:20s} F1={rec['f1_100']:.4f}")
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
        "corr_bit": corr,
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
