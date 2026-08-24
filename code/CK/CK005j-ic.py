"""
CK005j-ic — 按邻居稀有标签质量重加权 k-mer 传递，混进 I019
================================================================
依据:
  - Resnik, JAIR 1999, doi:10.1613/jair.514
    IC(c)=−log P(c)。稀有概念信息量大。
  - CAFA: Radivojac et al., Nature Methods 2013,
    doi:10.1038/nmeth.2340。评估用信息量加权，稀有 GO 更值钱。
  - 标签匿名无本体，用经验频率 π_j 当 P(c)。
  - CK005i：TF-IDF 改的是 k-mer 种子权，corr 0.85，关闭。
    I029/I030：列缩放 / 改阈值目标被 SCut 吞。
    本实验改邻居权重，不是列乘 IC。
自己的改法:
  冻 I019。ic_j=−log(π_j)。对每个 k-mer 邻居：
    rare = y_neigh · ic，w = sim² · (1+λ·rare/mean(rare))。
  再 sim² 加权传标签。λ∈{0.5,1.0,2.0}。
  ens'=(1-γ)·I019 + γ·col_norm(ic_sim)，γ∈{0.15,0.30}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/CK/CK005j-ic.py
输出:
  rel/CK005j_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "CK005j_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
LAM_GRID = (0.5, 1.0, 2.0)
G_GRID = (0.15, 0.30)
EPS = 1e-8


def ic_kmer(y_train, ic, lam):
    sim = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        prefix = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(i011.REL_DIR, f"{prefix}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_train[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        s = d["sim_val"][:, :5].astype(np.float32)
        rare = (lab * ic[None, None, :]).sum(-1)
        scale = float(rare.mean()) + EPS
        ww = np.maximum(s, 1e-7) ** 2
        ww = ww * (1.0 + lam * rare / scale)
        sim = sim + wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return sim.astype(np.float32)


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
    return s019, hit, y_train, y_val, sim_kmer, pi


def main():
    t0 = time.time()
    print("=" * 60)
    print("CK005j: IC 邻居重加权")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s019, hit, y_train, y_val, sim_kmer, pi = build_i019(
        train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    ic = (-np.log(np.clip(pi, EPS, 1.0))).astype(np.float32)
    print(f"  IC min={ic.min():.3f} med={np.median(ic):.3f} max={ic.max():.3f}")

    rows = [rec0]
    configs = {"i019": s019}
    overlap = {}

    print("[1/2] λ × γ ...")
    for lam in tqdm(LAM_GRID, desc="    λ", unit="l"):
        ic_sim = ic_kmer(y_train, ic, lam)
        rec_r = i011.eval_ens(ic_sim, y_val, hit, miss, f"ic_l{lam:.1f}")
        rec_r["delta_i019"] = round(rec_r["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec_r)
        corr = float(np.corrcoef(ic_sim.ravel(), sim_kmer.ravel())[0, 1])
        overlap[f"l{lam:.1f}"] = round(corr, 4)
        print(f"  ic λ={lam:.1f} 独立 F1={rec_r['f1_100']:.4f} corr_kmer={corr:.3f}")
        ic_n = i011.col_norm(ic_sim)
        for g in G_GRID:
            mixed = (1.0 - g) * s019 + g * ic_n
            name = f"mix_l{lam:.1f}_g{g:.2f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, name,
                                {"lam": lam, "gamma": g})
            rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[name] = mixed
            print(f"  {name:20s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

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
        print(f"  oof_{name:20s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    mix_oof = [r for r in oof_rows if r["name"].startswith("oof_mix_")]
    best_oof = max(mix_oof, key=lambda r: r["f1_100"]) if mix_oof else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline_i019": rec0,
        "best_oof": best_oof,
        "gain_oof_vs_i019": round(gain, 6),
        "useful": useful,
        "corr_kmer": overlap,
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
