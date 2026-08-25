"""
ECK010-esmknn — ESM2 嵌入 kNN 标签传递（GOPredSim）
================================================================
依据:
  - Littmann et al., Sci Rep 2021, doi:10.1038/s41598-020-80786-0
    GOPredSim：蛋白语言模型嵌入空间 kNN，把邻居 GO 传到查询，
    可越过序列同源。本赛标签匿名，但嵌入邻居仍可传 500 维标签。
  - E004/E004a 是线性/注意力头，关。ECK001 是 ProstT5 结构邻居，
    泄漏后关。本实验用已缓存 ESM2 mean 嵌入，train-only 库，不重训。
  - E003 探针：ESM2 余弦均值 0.94（较聚集）。本实验看聚集是否
    仍能给出与 I019 不正交的一元。
自己的改法:
  冻 I019。640 维 ESM2 L2 归一，val 对 idx_train 余弦 top-5，
  sim² 加权传标签。独立 F1 与融合 (1-γ)·I019 + γ·col_norm(esmknn)，
  γ∈{0.15,0.30}。门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/E/ECK010-esmknn.py
输出:
  rel/ECK010_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "ECK010_report.json")
EMB_PATH = os.path.join(i011.REL_DIR, "train_esm2_embeddings.parquet")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
G_GRID = (0.15, 0.30)
TOP = 5
N_EMB = 640


def l2_norm(x, eps=1e-8):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def knn_transfer(q, db, y_db, top=TOP, bs=256):
    n = q.shape[0]
    out = np.zeros((n, y_db.shape[1]), np.float32)
    yf = y_db.astype(np.float32)
    for i0 in tqdm(range(0, n, bs), desc="    knn", unit="b"):
        s = q[i0:i0 + bs] @ db.T
        part = np.argpartition(-s, top, axis=1)[:, :top]
        w = np.maximum(np.take_along_axis(s, part, axis=1), 1e-7) ** 2
        lab = yf[part.ravel()].reshape(s.shape[0], top, -1)
        out[i0:i0 + bs] = (lab * w[:, :, None]).sum(1) / w.sum(1)[:, None]
    return out


def build_i019(deep, sim, y_train, y_val):
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


def main():
    t0 = time.time()
    print("=" * 60)
    print("ECK010: ESM2 嵌入 kNN（GOPredSim）")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    print("[0] ESM2 嵌入 ...")
    emb = pd.read_parquet(EMB_PATH)
    if not (emb["protein_id"].values == train_df["protein_id"].values).all():
        raise RuntimeError("ESM2 嵌入与 train.csv protein_id 不对齐")
    cols = [f"emb_{i}" for i in range(N_EMB)]
    X = l2_norm(emb[cols].to_numpy(np.float32))
    db, q = X[idx_train], X[idx_val]
    print(f"  dim={X.shape[1]} train={db.shape[0]} val={q.shape[0]}")

    esm = knn_transfer(q, db, y_train)
    f1_ind = i011.macro_f1((esm >= 0.5).astype(np.int8), y_val)
    print(f"  独立 F1(0.5)={f1_ind:.4f}")

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
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    s019 = build_i019(deep, sim, y_train, y_val)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    rec_e = i011.eval_ens(esm, y_val, hit, miss, "esmknn")
    rec_e["delta_i019"] = round(rec_e["f1_100"] - rec0["f1_100"], 6)
    print(f"  I019 F1={rec0['f1_100']:.4f}")
    print(f"  esmknn F1={rec_e['f1_100']:.4f} d={rec_e['delta_i019']:+.4f}")

    corr = float(np.corrcoef(s019.ravel(), esm.ravel())[0, 1])
    print(f"  corr(I019,esmknn)={corr:.3f}")

    rows = [rec0, rec_e]
    configs = {"i019": s019}
    en = i011.col_norm(esm)
    print("[2/2] γ 混合 ...")
    for g in tqdm(G_GRID, desc="    γ", unit="g"):
        mixed = (1.0 - g) * s019 + g * en
        name = f"mix_g{g:.2f}"
        rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"gamma": g})
        rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        configs[name] = mixed
        print(f"  {name:12s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

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
        "f1_05_esmknn": round(float(f1_ind), 6),
        "corr_i019_esmknn": round(corr, 6),
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
