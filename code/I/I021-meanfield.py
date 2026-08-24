"""
I021-meanfield — I019 分数上稀疏 PMI 成对势能的平均场迭代
================================================================
依据:
  - Ghamrawi & McCallum, CIKM 2005, doi:10.1145/1099554.1099591
    Collective Multi-Label：标签成对因子，联合推断。
  - Krähenbühl & Koltun, NIPS 2011（全连接 CRF 平均场）
    q ← softmax(一元 + 成对 * q)，迭代几次。
  - ECK005：一次性 score + λ (score @ P) 全局 −0.0005，关闭。
    本实验改成 sigmoid 空间迭代，且叠在 I019 而非 CK005a。
自己的改法:
  冻 I019 分数。W_jk = max(PMI,0)，每行 top-K，对角 0。
  q^0 = sigmoid(ens)。q^{t+1} = sigmoid(ens + λ W q^t)，T 步。
  混合 ens' = (1-α)·ens + α·q^T。
  K∈{5,10}，λ∈{0.05,0.15}，T∈{2,4}，α∈{0.3,0.6}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I021-meanfield.py
输出:
  rel/I021_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I021_report.json")
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
K_GRID = (5, 10)
LAM_GRID = (0.05, 0.15)
T_GRID = (2, 4)
A_GRID = (0.3, 0.6)
EPS = 1e-8


def pmi_sparse(y, k):
    n, l = y.shape
    c = y.sum(0).astype(np.float64) + EPS
    joint = (y.T.astype(np.float64) @ y.astype(np.float64)) + EPS
    pmi = np.log((joint * n) / (c[:, None] * c[None, :]))
    np.fill_diagonal(pmi, 0.0)
    pmi = np.maximum(pmi, 0.0)
    w = np.zeros_like(pmi)
    for i in range(l):
        idx = np.argpartition(-pmi[i], kth=min(k, l - 1))[:k]
        w[i, idx] = pmi[i, idx]
    row = w.sum(1, keepdims=True)
    row[row < EPS] = 1.0
    return (w / row).astype(np.float32)


def meanfield(ens, w, lam, n_iter):
    q = 1.0 / (1.0 + np.exp(-np.clip(ens, -20, 20)))
    for _ in range(n_iter):
        q = 1.0 / (1.0 + np.exp(-np.clip(ens + lam * (q @ w.T), -20, 20)))
    return q.astype(np.float32)


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
    ens = (deep + i011.SIM_W * sim
           + i019.BETA * i011.col_norm(i011.rrf_pair(deep, sim, i019.K_RRF)))
    n = ens.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t_i011 = i011.search_t(ens, y_val)
    row_m = np.maximum((ens >= t_i011).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((ens - ens.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    mixed = i019.mix(ens, p, ALPHA_SK)
    return mixed, hit, y_train, y_val


def main():
    t0 = time.time()
    print("=" * 60)
    print("I021: PMI 平均场迭代")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    mixed, hit, y_train, y_val = build_i019(train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec0 = i011.eval_ens(mixed, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")
    rows = [rec0]
    configs = {"i019": mixed}

    print("[1/2] K × λ × T × α ...")
    for k in tqdm(K_GRID, desc="    K", unit="k"):
        wmat = pmi_sparse(y_train, k)
        for lam in LAM_GRID:
            for n_iter in T_GRID:
                q = meanfield(mixed, wmat, lam, n_iter)
                for a in A_GRID:
                    ens = (1.0 - a) * mixed + a * q
                    name = f"mf_K{k}_l{lam:.2f}_T{n_iter}_a{a:.1f}"
                    rec = i011.eval_ens(ens, y_val, hit, miss, name,
                                        {"K": k, "lam": lam, "T": n_iter, "alpha": a})
                    rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
                    rows.append(rec)
                    configs[name] = ens
                    print(f"  {name:28s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

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
        print(f"  oof_{name:28s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    best_oof = max(oof_rows, key=lambda r: r["f1_100"])
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001 and best_oof["name"] != "oof_i019")
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
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
