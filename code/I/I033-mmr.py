"""
I033-mmr — I019 top-K 上按标签 Jaccard 做 MMR 去冗余
================================================================
依据:
  - Carbonell & Goldstein, SIGIR 1998, doi:10.1145/290941.291025
    Maximal Marginal Relevance：λ·相关 − (1-λ)·与已选最大相似。
  - I008/I026：截断减基数，不是去近邻冗余。
  - ECK005/I021：共现加法抬相关标签，方向相反。
    本实验惩罚与已选标签 Jaccard 高的近邻，再排一次名。
自己的改法:
  冻 I019。train 标签 Jaccard。每条蛋白取 top-K，贪心 MMR。
  赋分 (K-pos)/K，其余 0。ens'=(1-γ)·I019 + γ·col_norm(mmr)。
  K∈{30,50}，λ∈{0.5,0.7}，γ∈{0.3}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I033-mmr.py
输出:
  rel/I033_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I033_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
K_GRID = (30, 50)
LAM_GRID = (0.5, 0.7)
GAMMA = 0.3
EPS = 1e-8


def label_jaccard(y):
    yf = y.astype(np.float64)
    inter = yf.T @ yf
    card = yf.sum(0)
    union = card[:, None] + card[None, :] - inter
    jac = inter / np.maximum(union, EPS)
    np.fill_diagonal(jac, 0.0)
    return jac.astype(np.float32)


def mmr_scores(ens, jac, k, lam):
    n, l = ens.shape
    order = np.argpartition(-ens, kth=k - 1, axis=1)[:, :k]
    out = np.zeros((n, l), np.float32)
    for i in tqdm(range(n), desc="    MMR", unit="row"):
        cands = order[i]
        rel = ens[i, cands].astype(np.float32)
        sub = jac[np.ix_(cands, cands)]
        red = np.zeros(k, np.float32)
        taken = np.zeros(k, np.bool_)
        for t in range(k):
            if t == 0:
                pick = int(rel.argmax())
            else:
                v = lam * rel - (1.0 - lam) * red
                v[taken] = -1e9
                pick = int(v.argmax())
            taken[pick] = True
            out[i, cands[pick]] = (k - t) / float(k)
            red = np.maximum(red, sub[pick])
    return out


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
    s019 = i019.mix(s011, p, ALPHA_SK)
    return s019, hit, y_train, y_val


def main():
    t0 = time.time()
    print("=" * 60)
    print("I033: MMR 标签去冗余")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s019, hit, y_train, y_val = build_i019(
        train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    print("[1/2] Jaccard + MMR ...")
    jac = label_jaccard(y_train)
    rows = [rec0]
    configs = {"i019": s019}

    for k in tqdm(K_GRID, desc="    K", unit="k"):
        for lam in LAM_GRID:
            mmr = mmr_scores(s019, jac, k, lam)
            rec_r = i011.eval_ens(mmr, y_val, hit, miss, f"mmr_K{k}_l{lam:.1f}_raw")
            rec_r["delta_i019"] = round(rec_r["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec_r)
            print(f"  mmr K={k} λ={lam:.1f} raw F1={rec_r['f1_100']:.4f}")
            mixed = (1.0 - GAMMA) * s019 + GAMMA * i011.col_norm(mmr)
            name = f"mmr_K{k}_l{lam:.1f}_g{GAMMA:.1f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, name,
                                {"K": k, "lam": lam, "gamma": GAMMA})
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
        print(f"  oof_{name:24s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    mmr_oof = [r for r in oof_rows if r["name"].startswith("oof_mmr_")]
    best_oof = max(mmr_oof, key=lambda r: r["f1_100"]) if mmr_oof else oof_rows[0]
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
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s")


if __name__ == "__main__":
    main()
