"""
I025-rakel — 经验共现簇上做组内重标度（RAkEL 思想，不重训分类器）
================================================================
依据:
  - RAkEL: Tsoumakas, Katakis & Vlahavas, IEEE TKDE 2011,
    doi:10.1109/tkde.2010.164。把多标签拆成随机 k 子集，子集内学联合。
  - I015：按频次三分位组内排名再 RRF，OOF 差于全局，关闭。
    本实验簇来自共现，不是频次桶；操作是组内 min-max，不是重排名。
  - I021/I022：成对/子空间几乎不改排序。组内重标度只改簇内量纲。
自己的改法:
  冻 I019。train 标签 Jaccard 谱聚类成 C 簇（sklearn SpectralClustering）。
  对每个样本、每个簇，把该簇分数做 min-max，写回后再与原分数凸组合。
  C∈{8,16}，α∈{0.3,0.6,1.0}。对照：随机划分同样 C 簇。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I025-rakel.py
输出:
  rel/I025_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I025_report.json")
K_RRF, BETA = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
C_GRID = (8, 16)
A_GRID = (0.3, 0.6, 1.0)
EPS = 1e-8


def jaccard_aff(y):
    yf = y.astype(np.float64)
    inter = yf.T @ yf
    card = yf.sum(0)
    union = card[:, None] + card[None, :] - inter
    aff = inter / np.maximum(union, EPS)
    np.fill_diagonal(aff, 1.0)
    return aff


def spectral_labels(aff, n_clusters, seed):
    sc = SpectralClustering(
        n_clusters=n_clusters, affinity="precomputed",
        assign_labels="kmeans", random_state=seed)
    return sc.fit_predict(aff)


def cluster_minmax(score, labels):
    out = score.copy()
    for c in np.unique(labels):
        cols = np.where(labels == c)[0]
        if len(cols) < 2:
            continue
        block = score[:, cols]
        lo = block.min(axis=1, keepdims=True)
        hi = block.max(axis=1, keepdims=True)
        out[:, cols] = (block - lo) / np.maximum(hi - lo, 1e-9)
    return out.astype(np.float32)


def build_i019(train_df, y_full, idx_train, idx_val):
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1 - i011.ALIGN_W) * sim_kmer
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    s011 = deep + i011.SIM_W * sim + BETA * i011.col_norm(i011.rrf_pair(deep, sim, K_RRF))
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t_i011 = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t_i011).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    s019 = i019.mix(s011, p, ALPHA_SK)
    return s019, hit, y_train, y_val


def main():
    t0 = time.time()
    print("=" * 60)
    print("I025: 共现簇组内重标度")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s019, hit, y_train, y_val = build_i019(train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")
    rows = [rec0]
    configs = {"i019": s019}

    print("[1/2] 谱聚类簇 + 组内 min-max ...")
    aff = jaccard_aff(y_train)
    rng = np.random.RandomState(i011.RANDOM_STATE)
    for c in tqdm(C_GRID, desc="    C", unit="c"):
        lab = spectral_labels(aff, c, i011.RANDOM_STATE)
        sizes = np.bincount(lab, minlength=c)
        print(f"  C={c} 簇大小 min={sizes.min()} med={np.median(sizes):.0f} max={sizes.max()}")
        scaled = cluster_minmax(s019, lab)
        rand_lab = rng.randint(0, c, size=s019.shape[1])
        scaled_rand = cluster_minmax(s019, rand_lab)
        for a in A_GRID:
            mixed = (1.0 - a) * s019 + a * scaled
            name = f"rak_C{c}_a{a:.1f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"C": c, "alpha": a, "kind": "cooccur"})
            rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[name] = mixed
            print(f"  {name:18s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")
        mixed_r = 0.3 * scaled_rand + 0.7 * s019
        name = f"rand_C{c}_a0.3"
        rec = i011.eval_ens(mixed_r, y_val, hit, miss, name, {"C": c, "alpha": 0.3, "kind": "random"})
        rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        configs[name] = mixed_r
        print(f"  {name:18s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

    print("[2/2] 半折核对...")
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            t = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= t).astype(np.int8)
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
        print(f"  oof_{name:18s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    rak_oof = [r for r in oof_rows if r["name"].startswith("oof_rak_")]
    best_oof = max(rak_oof, key=lambda r: r["f1_100"]) if rak_oof else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline_i019": rec0,
        "best_oof": best_oof,
        "gain_oof_vs_i019": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(i011.REL_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s")


if __name__ == "__main__":
    main()
