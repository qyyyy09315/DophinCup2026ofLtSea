"""
I042-deeprrf — 三个深度头做 RRF，替换线性 deep 混权
================================================================
依据:
  - Cormack, Clarke & Buettcher, SIGIR 2009, doi:10.1145/1571941.1572114
    多路排序 1/(k+rank) 相加，不校准分数。
  - Lakshminarayanan et al., NeurIPS 2017, arXiv:1612.01474
    独立网络平均有效，因落在不同盆地。此处不重训，只把已有
    t001a / t002 / cnn 的名次融合。
  - I011：线性 deep=0.1 t001a+0.2 t002+0.2 cnn，再与 sim 做 RRF。
  - I011a：拆的是 deep/比对/k-mer 三路，不是三个深度头。
自己的改法:
  冻 sim / 邻居。三路概率各自按行排名，
  rrf3 = Σ 1/(k+rank_m+1)，k=30。
  deep' = (1-α)·线性deep + α·col_norm(rrf3)，α∈{0.3,0.6,1.0}。
  再走 I011+I019。门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I042-deeprrf.py
输出:
  rel/I042_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I042_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
A_GRID = (0.30, 0.60, 1.00)
G_GRID = (0.15, 0.30)


def rrf3(a, b, c, k):
    return (1.0 / (k + i011.ranks(a) + 1)
            + 1.0 / (k + i011.ranks(b) + 1)
            + 1.0 / (k + i011.ranks(c) + 1))


def sinkhorn_on(s, y_train, y_val):
    n = s.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t = i011.search_t(s, y_val)
    row_m = np.maximum((s >= t).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s - s.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(s, p, ALPHA_SK)


def i019_from_deep(deep, sim, y_train, y_val):
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep, sim, K_RRF))
    return sinkhorn_on(s011, y_train, y_val)


def main():
    t0 = time.time()
    print("=" * 60)
    print("I042: 三深度头 RRF")

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
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer

    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    a = t1["t001a_val"]
    b = t2["t002"]
    c = vp["cnn"]
    deep = w["t001a"] * a + w["t002"] * b + w["cnn"] * c
    print(f"  corr t001a-t002={np.corrcoef(a.ravel(), b.ravel())[0,1]:.3f} "
          f"t001a-cnn={np.corrcoef(a.ravel(), c.ravel())[0,1]:.3f} "
          f"t002-cnn={np.corrcoef(b.ravel(), c.ravel())[0,1]:.3f}")

    s019 = i019_from_deep(deep, sim, y_train, y_val)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    rrf = i011.col_norm(rrf3(a, b, c, K_RRF))
    rows = [rec0]
    configs = {"i019": s019}

    print("[1/2] α × γ ...")
    for alpha in tqdm(A_GRID, desc="    α", unit="a"):
        deep_n = (1.0 - alpha) * deep + alpha * rrf
        s_n = i019_from_deep(deep_n, sim, y_train, y_val)
        rec_n = i011.eval_ens(s_n, y_val, hit, miss, f"rrf_a{alpha:.2f}",
                              {"alpha": alpha})
        rec_n["delta_i019"] = round(rec_n["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec_n)
        configs[f"rrf_a{alpha:.2f}"] = s_n
        print(f"  rrf_a{alpha:.2f} F1={rec_n['f1_100']:.4f} "
              f"d={rec_n['delta_i019']:+.4f}")
        if alpha >= 0.999:
            continue
        sn = i011.col_norm(s_n)
        for g in G_GRID:
            mixed = (1.0 - g) * s019 + g * sn
            name = f"mix_a{alpha:.2f}_g{g:.2f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, name,
                                {"alpha": alpha, "gamma": g})
            rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[name] = mixed
            print(f"  {name:20s} F1={rec['f1_100']:.4f} "
                  f"d={rec['delta_i019']:+.4f}")

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
        "corr": {
            "t001a_t002": round(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), 6),
            "t001a_cnn": round(float(np.corrcoef(a.ravel(), c.ravel())[0, 1]), 6),
            "t002_cnn": round(float(np.corrcoef(b.ravel(), c.ravel())[0, 1]), 6),
        },
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
