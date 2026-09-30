"""
I039-chain — 按频率链做一步残差：常见标签修正稀有标签
================================================================
依据:
  - Read, Pfahringer, Holmes & Frank, Machine Learning 2011,
    doi:10.1007/s10994-011-5256-5。Classifier Chains：按序把已预测
    标签当特征。完整 500 链重训过重。
  - ECK002 决策后硬补 Δ=0；ECK005 共现软传播全局负。
    本实验不补集合、不改 hit，只在 miss 上用高频标签分数
    线性修正低频标签。
自己的改法:
  冻 I019。标签按 train 频率降序。对每个低频标签 j，取更高频的
  条件相关最高的 i，半折拟合 s_j' = s_j + α·(s_i − ē_i) 的 α
  （一维最小二乘，只在 miss 行）。α clip 到 [−0.3,0.3]。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I039-chain.py
输出:
  rel/I039_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I039_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
MIN_PARENT = 20
ALPHA_CLIP = 0.30
G_GRID = (0.15, 0.30, 0.50)


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
    return i019.mix(s011, p, ALPHA_SK), hit, y_train, y_val, pi


def parent_of(y_train, pi):
    order = np.argsort(-pi)
    l = y_train.shape[1]
    parents = np.full(l, -1, np.int32)
    yf = y_train.astype(np.float64)
    co = yf.T @ yf
    freq = yf.sum(0)
    rank = np.empty(l, np.int32)
    rank[order] = np.arange(l)
    for j in tqdm(range(l), desc="    parent", unit="lab"):
        mask = (rank < rank[j]) & (freq >= MIN_PARENT)
        if not mask.any():
            continue
        cond = np.full(l, -np.inf)
        cond[mask] = co[j, mask] / np.maximum(freq[mask], 1.0)
        i = int(np.argmax(cond))
        if cond[i] >= 0.2:
            parents[j] = i
    return parents, order


def fit_alpha(s, y, miss, parents, fit_idx):
    n, l = s.shape
    a = np.zeros(l, np.float32)
    fit_m = fit_idx[miss[fit_idx]]
    if fit_m.size < 20:
        return a
    for j in tqdm(range(l), desc="    alpha", unit="lab", leave=False):
        i = parents[j]
        if i < 0:
            continue
        x = s[fit_m, i] - s[fit_m, i].mean()
        r = y[fit_m, j].astype(np.float64) - s[fit_m, j]
        den = float(np.dot(x, x))
        if den < 1e-8:
            continue
        a[j] = np.clip(float(np.dot(x, r) / den), -ALPHA_CLIP, ALPHA_CLIP)
    return a


def apply_chain(s, miss, parents, a, g=1.0):
    out = s.copy()
    idx = np.where(miss)[0]
    for j in range(s.shape[1]):
        i = parents[j]
        if i < 0 or abs(a[j]) < 1e-8:
            continue
        x = s[idx, i] - s[idx, i].mean()
        out[idx, j] = s[idx, j] + g * a[j] * x
    return out


def main():
    t0 = time.time()
    print("=" * 60)
    print("I039: 频率链残差（miss）")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s019, hit, y_train, y_val, pi = build_i019(
        train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f} missF1={rec0['f1_miss']:.4f}")

    parents, order = parent_of(y_train, pi)
    n_par = int((parents >= 0).sum())
    print(f"  有父标签 {n_par}/500")

    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    rows = [rec0]
    configs = {"i019": s019}

    print("[1/2] 半折拟合 α ...")
    chained = np.zeros_like(s019)
    alphas = []
    for i, hold in enumerate(folds):
        fit = folds[1 - i]
        a = fit_alpha(s019, y_val, miss, parents, fit)
        alphas.append(a)
        chained[hold] = apply_chain(s019, miss, parents, a, g=1.0)[hold]
    a_mean = np.mean(np.abs(alphas[0][parents >= 0])) if n_par else 0.0
    print(f"  |α| mean={a_mean:.4f}")

    rec_c = i011.eval_ens(chained, y_val, hit, miss, "chain")
    rec_c["delta_i019"] = round(rec_c["f1_100"] - rec0["f1_100"], 6)
    rows.append(rec_c)
    print(f"  chain F1={rec_c['f1_100']:.4f} d={rec_c['delta_i019']:+.4f} "
          f"miss={rec_c['f1_miss']:.4f}")

    print("[2/2] γ 混合 ...")
    for g in tqdm(G_GRID, desc="    γ", unit="g"):
        mixed = (1.0 - g) * s019 + g * chained
        mixed[hit] = s019[hit]
        name = f"mix_g{g:.2f}"
        rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"gamma": g})
        rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        configs[name] = mixed
        print(f"  {name:16s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f} "
              f"miss={rec['f1_miss']:.4f}")
    configs["chain"] = chained

    print("[3/3] 半折核对阈值...")
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
        print(f"  oof_{name:16s} F1={rec['f1_100']:.4f} miss={rec['f1_miss']:.4f}")
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
        "n_parent": n_par,
        "abs_alpha_mean": round(float(a_mean), 6),
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
