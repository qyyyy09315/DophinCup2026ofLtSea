"""
I028-lightgcn — 蛋白–标签二部图上对 I019 做 LightGCN 邻域聚合
================================================================
依据:
  - LightGCN: He, Deng, Wang, Li, Zhang & Wang, SIGIR 2020,
    doi:10.1145/3397271.3401063。去掉特征变换，只做对称归一邻域聚合，
    多层嵌入再平均。推荐里用户–物品二部图。
  - I019：对分数矩阵做行/列质量匹配，没有邻域。
  - I021：标签–标签 PMI；ECK008：蛋白–蛋白 k-mer。
    本实验走蛋白–标签二部图，两边交替聚合。
自己的改法:
  冻 I019 为边权 R（可按行只留 top-M）。
  R1 = D_p^{-1/2} R D_l^{-1} R^T D_p^{-1/2} R
  一层：蛋白→标签→蛋白。二层再套一次。
  LightGCN 读出 (R+R1)/2 或 (R+R1+R2)/3，再与原分数凸组合。
  M∈{0,50}（0=稠密），T∈{1,2}，β∈{0.3,0.6}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I028-lightgcn.py
输出:
  rel/I028_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I028_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
M_GRID = (0, 50)
T_GRID = (1, 2)
B_GRID = (0.3, 0.6)
EPS = 1e-8


def keep_topm(r, m):
    if m <= 0:
        return np.maximum(r, 0.0)
    out = np.zeros_like(r)
    part = np.argpartition(-r, m, axis=1)[:, :m]
    rows = np.arange(r.shape[0])[:, None]
    out[rows, part] = np.maximum(r[rows, part], 0.0)
    return out


def conv(r):
    """R1 = Dp^{-1/2} R Dl^{-1} R^T Dp^{-1/2} R，全用矩阵乘，不建 N×N。"""
    dp = np.maximum(r.sum(1, keepdims=True), EPS)
    dl = np.maximum(r.sum(0, keepdims=True), EPS)
    inv_sqrt_p = 1.0 / np.sqrt(dp)
    tmp = r * inv_sqrt_p
    mid = (r.T @ tmp) / dl.T
    return (r @ mid) * inv_sqrt_p


def lightgcn(r, n_layer):
    layers = [r]
    cur = r
    for _ in range(n_layer):
        cur = conv(cur)
        layers.append(cur)
    return np.mean(layers, axis=0).astype(np.float32)


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
    print("I028: LightGCN 二部图聚合")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s019, hit, y_val = build_i019(train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")
    rows = [rec0]
    configs = {"i019": s019}

    print("[1/2] M × T × β ...")
    r0 = np.maximum(s019, 0.0)
    for m in tqdm(M_GRID, desc="    M", unit="m"):
        adj = keep_topm(r0, m)
        for n_layer in T_GRID:
            rec_lgc = lightgcn(adj, n_layer)
            rec_n = i011.col_norm(rec_lgc)
            for b in B_GRID:
                mixed = (1.0 - b) * s019 + b * rec_n
                name = f"lgcn_M{m}_T{n_layer}_b{b:.1f}"
                rec = i011.eval_ens(mixed, y_val, hit, miss, name,
                                    {"M": m, "T": n_layer, "beta": b})
                rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
                rows.append(rec)
                configs[name] = mixed
                print(f"  {name:22s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

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
        print(f"  oof_{name:22s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    lgc_oof = [r for r in oof_rows if r["name"].startswith("oof_lgcn_")]
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
