"""
I022-spen — 冻结 I011 一元项，低秩标签子空间能量 + 投影梯度推断
================================================================
依据:
  - SPENs: Belanger & McCallum, ICML 2016, arXiv:1511.06350
    能量 E(x,y) 对候选标签可微，预测靠对 y 做梯度下降，而不是 CRF 平均场。
  - PLST: Tai & Lin, Neural Computation 2012, doi:10.1162/neco_a_00320
    多标签落在低维主标签空间；本实验用 idx_train 标签 SVD 当能量的成对项。
  - I021：稀疏 PMI 平均场叠在 I019 上 OOF +0.0003，关闭。
    不在 I019 上再叠成对/共现。本实验把 SPEN 当 Sinkhorn 的替代结构推断，
    一元项锁 I011。
自己的改法:
  冻 I011 分数 ens。V = train 标签中心化 SVD 的前 r 个右奇异向量。
  列归一 s=col_norm(ens)。能量 E(y)= -<s,y> + (λ/2)||(I-VV^T)y||^2。
  T=0：直接把 s 投到子空间。T>0：从 s 出发做投影梯度 T 步，
  y ← clip(y + η(s - λ(I-VV^T)y), 0, 1)。
  混合 ens'=(1-α)·ens + α·col_norm(y)。
  r∈{32,64}，λ∈{0.4,0.8}，T∈{0,8}，α∈{0.3,0.6}，η=0.25。
  门槛：相对 I011 半折 ≥ +0.001；替换本地锁还要相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I022-spen.py
输出:
  rel/I022_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I022_report.json")
K_RRF, BETA = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
R_GRID = (32, 64)
LAM_GRID = (0.4, 0.8)
T_GRID = (0, 8)
A_GRID = (0.3, 0.6)
ETA = 0.25


def label_basis(y, r):
    yc = y.astype(np.float64) - y.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(yc, full_matrices=False)
    return vt[:r].astype(np.float32)


def project(y, v):
    return y @ v.T @ v


def infer(s, v, lam, n_iter, eta=ETA):
    y = np.clip(s, 0.0, 1.0)
    if n_iter <= 0:
        return np.clip(project(y, v), 0.0, 1.0).astype(np.float32)
    for _ in range(n_iter):
        py = y - project(y, v)
        y = np.clip(y + eta * (s - lam * py), 0.0, 1.0)
    return y.astype(np.float32)


def build_i011(train_df, y_full, idx_train, idx_val):
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
    ens = deep + i011.SIM_W * sim + BETA * i011.col_norm(i011.rrf_pair(deep, sim, K_RRF))
    return ens, hit, y_train, y_val


def build_i019(ens, y_train, y_val):
    n = ens.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t_i011 = i011.search_t(ens, y_val)
    row_m = np.maximum((ens >= t_i011).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((ens - ens.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(ens, p, ALPHA_SK)


def main():
    t0 = time.time()
    print("=" * 60)
    print("I022: SPEN 低秩能量推断")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    ens, hit, y_train, y_val = build_i011(train_df, y_full, idx_train, idx_val)
    miss = ~hit
    mixed19 = build_i019(ens, y_train, y_val)

    rec0 = i011.eval_ens(ens, y_val, hit, miss, "i011")
    rec19 = i011.eval_ens(mixed19, y_val, hit, miss, "i019")
    print(f"  I011 F1={rec0['f1_100']:.4f}")
    print(f"  I019 F1={rec19['f1_100']:.4f}")
    rows = [rec0, rec19]
    configs = {"i011": ens, "i019": mixed19}

    s = i011.col_norm(ens)
    print("[1/2] r × λ × T × α ...")
    for r in tqdm(R_GRID, desc="    r", unit="r"):
        v = label_basis(y_train, r)
        for lam in LAM_GRID:
            for n_iter in T_GRID:
                y_hat = infer(s, v, lam, n_iter)
                y_n = i011.col_norm(y_hat)
                for a in A_GRID:
                    mixed = (1.0 - a) * ens + a * y_n
                    name = f"sp_r{r}_l{lam:.1f}_T{n_iter}_a{a:.1f}"
                    rec = i011.eval_ens(mixed, y_val, hit, miss, name,
                                        {"r": r, "lam": lam, "T": n_iter, "alpha": a})
                    rec["delta_i011"] = round(rec["f1_100"] - rec0["f1_100"], 6)
                    rec["delta_i019"] = round(rec["f1_100"] - rec19["f1_100"], 6)
                    rows.append(rec)
                    configs[name] = mixed
                    print(f"  {name:28s} F1={rec['f1_100']:.4f} "
                          f"d011={rec['delta_i011']:+.4f} d019={rec['delta_i019']:+.4f}")

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
            t = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= t).astype(np.int8)
        return pred

    oof_rows = []
    oof_i011 = oof_i019 = None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "i011":
            oof_i011 = rec["f1_100"]
        elif name == "i019":
            oof_i019 = rec["f1_100"]
        print(f"  oof_{name:28s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i011"] = round(rec["f1_100"] - oof_i011, 6)
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    spen_oof = [r for r in oof_rows if r["name"].startswith("oof_sp_")]
    best_oof = max(spen_oof, key=lambda r: r["f1_100"]) if spen_oof else oof_rows[0]
    gain011 = best_oof["f1_100"] - oof_i011
    gain019 = best_oof["f1_100"] - oof_i019
    useful = bool(gain011 >= 0.001)
    replace = bool(gain019 >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I011 {gain011:+.4f} vs I019 {gain019:+.4f} "
          f"→ {'有效' if useful else '无增益'}"
          f"{'，替换 I019' if replace else '，不替换 I019'}")

    report = {
        "baseline_i011": rec0,
        "baseline_i019": rec19,
        "best_oof": best_oof,
        "gain_oof_vs_i011": round(gain011, 6),
        "gain_oof_vs_i019": round(gain019, 6),
        "useful": useful,
        "replace_i019": replace,
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
