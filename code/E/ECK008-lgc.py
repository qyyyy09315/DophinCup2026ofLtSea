"""
ECK008-lgc — val 蛋白 k-mer 图上对 I019 做 LGC 平滑
================================================================
依据:
  - Zhou, Bousquet, Lal, Weston & Schölkopf, NIPS 2003/2004
    Learning with Local and Global Consistency.
    F ← α S F + (1-α) Y，S 为对称归一邻接。
  - ECK003：二跳传播的是邻居标签，corr(2hop,1hop)=0.80，关闭。
    本实验平滑的是查询蛋白自己的 I019 分数，图只看序列。
  - I021–I027：标签轴/分数矩阵结构全关。换实例图。
自己的改法:
  冻 I019。用 HashingVectorizer 6-mer 建 val–val 无向 kNN（k=5，去自环）。
  Y=I019。T 步 LGC 后 ens'=(1-β)·I019 + β·F。
  α∈{0.3,0.6,0.9}，T∈{5,15}，β∈{0.3,0.6}。
  对照：只在 miss 行写入。门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/E/ECK008-lgc.py
输出:
  rel/ECK008_report.json
  rel/ECK008_neigh_val_k6.npz（val–val 邻居缓存）
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
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

REPORT_PATH = os.path.join(i011.REL_DIR, "ECK008_report.json")
CACHE_PATH = os.path.join(i011.REL_DIR, "ECK008_neigh_val_k6.npz")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
TOP_K = 5
N_FEATURES = 2 ** 20
BLOCK = 2048
A_LGC = (0.3, 0.6, 0.9)
T_GRID = (5, 15)
B_GRID = (0.3, 0.6)


def val_knn(seqs, cache=CACHE_PATH):
    if os.path.exists(cache):
        d = np.load(cache)
        return d["idx"], d["sim"]
    vec = HashingVectorizer(
        analyzer="char", ngram_range=(6, 6),
        n_features=N_FEATURES, dtype=np.float32)
    x = normalize(sparse.csr_matrix(vec.transform(seqs)))
    n = x.shape[0]
    idx = np.zeros((n, TOP_K), np.int64)
    sim = np.zeros((n, TOP_K), np.float32)
    for s in tqdm(range(0, n, BLOCK), desc="    val-val kNN", unit="blk"):
        e = min(s + BLOCK, n)
        block = (x[s:e] @ x.T).toarray()
        for i in range(e - s):
            block[i, s + i] = -1.0
        part = np.argpartition(-block, TOP_K, axis=1)[:, :TOP_K]
        idx[s:e] = part
        sim[s:e] = np.take_along_axis(block, part, axis=1)
    np.savez_compressed(cache, idx=idx, sim=sim)
    return idx, sim


def sym_norm(idx, sim, n):
    rows = np.repeat(np.arange(n), TOP_K)
    cols = idx.ravel()
    val = np.maximum(sim.ravel().astype(np.float64), 0.0)
    w = sparse.csr_matrix((val, (rows, cols)), shape=(n, n))
    w = w.maximum(w.T)
    w.setdiag(0)
    w.eliminate_zeros()
    deg = np.array(w.sum(1)).ravel()
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = 1.0 / np.sqrt(deg[nz])
    d = sparse.diags(dinv)
    return d @ w @ d


def lgc(y, s, alpha, n_iter):
    f = y.astype(np.float64)
    y64 = y.astype(np.float64)
    for _ in range(n_iter):
        f = alpha * (s @ f) + (1.0 - alpha) * y64
    return f.astype(np.float32)


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
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(i011.rrf_pair(deep, sim, K_RRF))
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t_i011 = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t_i011).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    s019 = i019.mix(s011, p, ALPHA_SK)
    return s019, hit, y_val


def main():
    t0 = time.time()
    print("=" * 60)
    print("ECK008: val 图 LGC 平滑 I019")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s019, hit, y_val = build_i019(train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    print("[1/3] val–val 6-mer kNN ...")
    val_seqs = train_df["sequence"].values[idx_val]
    knn_idx, knn_sim = val_knn(val_seqs)
    n = len(y_val)
    s_mat = sym_norm(knn_idx, knn_sim, n)
    deg = np.array(s_mat.getnnz(axis=1))
    print(f"  边/点 nnz={s_mat.nnz}  度 min={deg.min()} med={np.median(deg):.0f}")

    rows = [rec0]
    configs = {"i019": s019}

    print("[2/3] α × T × β ...")
    for a in tqdm(A_LGC, desc="    α", unit="a"):
        for n_iter in T_GRID:
            f = lgc(s019, s_mat, a, n_iter)
            for b in B_GRID:
                mixed = (1.0 - b) * s019 + b * f
                name = f"lgc_a{a:.1f}_T{n_iter}_b{b:.1f}"
                rec = i011.eval_ens(mixed, y_val, hit, miss, name,
                                    {"alpha": a, "T": n_iter, "beta": b, "scope": "all"})
                rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
                rows.append(rec)
                configs[name] = mixed
                print(f"  {name:24s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")
            mixed_m = s019.copy()
            mixed_m[miss] = (1.0 - 0.3) * s019[miss] + 0.3 * f[miss]
            name = f"lgc_a{a:.1f}_T{n_iter}_miss"
            rec = i011.eval_ens(mixed_m, y_val, hit, miss, name,
                                {"alpha": a, "T": n_iter, "beta": 0.3, "scope": "miss"})
            rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[name] = mixed_m
            print(f"  {name:24s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

    print("[3/3] 半折核对...")
    rng = np.random.RandomState(i011.RANDOM_STATE)
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

    lgc_oof = [r for r in oof_rows if r["name"].startswith("oof_lgc_")]
    best_oof = max(lgc_oof, key=lambda r: r["f1_100"]) if lgc_oof else oof_rows[0]
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
