"""
CK019-esmrr — DIAMOND 邻居用 ESM2 余弦重加权（软比对）
================================================================
依据:
  - Harrigan et al., BMC Bioinformatics 2024,
    doi:10.1186/s12859-024-05779-6。病毒基因注释把语言模型
    嵌入与软比对合用，不丢序列同源。
  - Littmann et al. GOPredSim（Sci Rep 2021）是全库嵌入 kNN；
    ECK010 已关（独立 0.201，OOF −0.0025）。本实验不扩检索集合，
    只对 DIAMOND 已命中的邻居用嵌入余弦改权。
  - CK007 pident / CK016 E 值 / CK010 qcov / CK018 熵 都关过。
    本实验权 = bitscore · max(cos, 0)^α，名单不变。
自己的改法:
  冻 deep / k-mer / DIAMOND 名单。ESM2 L2 归一后取查询–靶标余弦。
  α∈{1,2}。重建 I019。另混 (1-γ)·I019 + γ·新 I019，γ∈{0.15,0.30}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/CK/CK019-esmrr.py
输出:
  rel/CK019_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "CK019_report.json")
EMB_PATH = os.path.join(i011.REL_DIR, "train_esm2_embeddings.parquet")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
A_GRID = (1.0, 2.0)
G_GRID = (0.15, 0.30)
N_EMB = 640


def l2_norm(x, eps=1e-8):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def pair_cos(q, db, idx):
    """idx: (n, k) into db rows; invalid <0 → 0."""
    n, k = idx.shape
    valid = idx >= 0
    gathered = db[np.clip(idx, 0, None)]
    dots = (gathered * q[:, None, :]).sum(-1)
    dots[~valid] = 0.0
    return dots.astype(np.float32)


def build_fusion(deep, sim_bit, sim_kmer, y_train, y_val):
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
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
    print("CK019: DIAMOND 邻居 ESM2 余弦重加权")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    print("[0] ESM2 + DIAMOND ...")
    emb = pd.read_parquet(EMB_PATH)
    if not (emb["protein_id"].values == train_df["protein_id"].values).all():
        raise RuntimeError("ESM2 嵌入与 train.csv protein_id 不对齐")
    cols = [f"emb_{i}" for i in range(N_EMB)]
    X = l2_norm(emb[cols].to_numpy(np.float32))
    db, q = X[idx_train], X[idx_val]

    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    print(f"  hit {int(hit.sum())}/{len(hit)}")

    cos = pair_cos(q, db, idx_v)
    valid = idx_v >= 0
    c_ok = cos[valid]
    print(f"  cos p10={np.percentile(c_ok, 10):.3f} med={np.median(c_ok):.3f} "
          f"p90={np.percentile(c_ok, 90):.3f}")

    sim_bit0 = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    s019 = build_fusion(deep, sim_bit0, sim_kmer, y_train, y_val)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    rows = [rec0]
    configs = {"i019": s019}
    corr_bits = float(np.corrcoef(bits_v[valid], np.maximum(c_ok, 0.0))[0, 1])
    print(f"  corr(bits, max(cos,0))={corr_bits:.3f}")

    print("[1/2] α 重加权 ...")
    for a in A_GRID:
        ww = bits_v * np.maximum(cos, 0.0) ** a
        sim_bit = i011.agg_labels(idx_v, ww, y_train)
        s_new = build_fusion(deep, sim_bit, sim_kmer, y_train, y_val)
        name = f"rr_a{a:.0f}"
        rec = i011.eval_ens(s_new, y_val, hit, miss, name, {"alpha": a})
        rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rec["corr_bit"] = round(float(np.corrcoef(sim_bit0.ravel(), sim_bit.ravel())[0, 1]), 4)
        rows.append(rec)
        configs[name] = s_new
        print(f"  {name:10s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f} "
              f"corr_sim={rec['corr_bit']:.3f}")

    print("[2/2] γ 混合最优 α ...")
    best_rr = max(
        (r for r in rows if r["name"].startswith("rr_")),
        key=lambda r: r["f1_100"])
    s_rr = configs[best_rr["name"]]
    for g in tqdm(G_GRID, desc="    γ", unit="g"):
        mixed = (1.0 - g) * s019 + g * s_rr
        name = f"mix_{best_rr['name']}_g{g:.2f}"
        rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"gamma": g})
        rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        configs[name] = mixed
        print(f"  {name:22s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

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
        print(f"  oof_{name:22s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    cand = [r for r in oof_rows if r["name"] != "oof_i019"]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "cos_p10": round(float(np.percentile(c_ok, 10)), 4),
        "cos_med": round(float(np.median(c_ok)), 4),
        "cos_p90": round(float(np.percentile(c_ok, 90)), 4),
        "corr_bits_cos": round(corr_bits, 6),
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
