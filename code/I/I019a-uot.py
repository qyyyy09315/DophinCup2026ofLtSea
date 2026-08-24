"""
I019a-uot — I019 派生：不平衡 Sinkhorn（行列质量可松）
================================================================
依据:
  - Chizat, Peyré, Schmitzer, Vialard, Math. Comp. 2018, doi:10.1090/mcom/3303
    不平衡 OT：KL 惩罚代替硬边际，缩放迭代
    u ← (a / Kv)^{ε/(ε+λ)}。
  - I019 平衡 Sinkhorn τ=0.20 α=0.60 OOF 较 I011 +0.0011。
    硬对齐 π 与 SCut 基数可能过约束（蛋白真实 k 不是精确 π·N）。
自己的改法:
  冻 I019 的 τ=0.20、ρ=1.0、α=0.60、核与混合。只把 Sinkhorn 改成
  不平衡：γ=ε/(ε+λ)∈{0.5,0.7,0.85,1.0}（1.0=平衡=I019）。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I019a-uot.py
输出:
  rel/I019a_report.json
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
_spec_i = importlib.util.spec_from_file_location(
    "i011", os.path.join(PROJECT_ROOT, "code", "I", "I011-rrf.py"))
i011 = importlib.util.module_from_spec(_spec_i)
_spec_i.loader.exec_module(i011)
_spec_s = importlib.util.spec_from_file_location(
    "i019", os.path.join(PROJECT_ROOT, "code", "I", "I019-sinkhorn.py"))
i019 = importlib.util.module_from_spec(_spec_s)
_spec_s.loader.exec_module(i019)

REPORT_PATH = os.path.join(i011.REL_DIR, "I019a_report.json")
TAU, RHO, ALPHA = 0.20, 1.0, 0.60
GAMMA = (0.50, 0.70, 0.85, 1.00)
N_ITER = 40


def sinkhorn_uot(kernel, row_m, col_m, gamma, n_iter=N_ITER, eps=1e-12):
    """gamma=1 平衡；gamma<1 不平衡。"""
    u = np.ones(kernel.shape[0], np.float64)
    v = np.ones(kernel.shape[1], np.float64)
    k = kernel.astype(np.float64)
    for _ in range(n_iter):
        u = (row_m / (k @ v + eps)) ** gamma
        v = (col_m / (k.T @ u + eps)) ** gamma
    return (u[:, None] * k * v[None, :]).astype(np.float32)


def main():
    t0 = time.time()
    print("=" * 60)
    print("I019a: 不平衡 Sinkhorn")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
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
    t_scut = i011.search_t(ens, y_val)
    row_m = np.maximum((ens >= t_scut).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((ens - ens.max(axis=1, keepdims=True)) / TAU)

    rec0 = i011.eval_ens(ens, y_val, hit, miss, "i011")
    p_bal = i019.sinkhorn(ker, row_m, col_m)
    ens_i019 = i019.mix(ens, p_bal, ALPHA)
    rec1 = i011.eval_ens(ens_i019, y_val, hit, miss, "i019_bal")
    print(f"  I011 {rec0['f1_100']:.4f}  I019 {rec1['f1_100']:.4f}")
    rows = [rec0, rec1]
    configs = {"i011": ens, "i019_bal": ens_i019}

    print("[1/2] γ 网格...")
    for g in GAMMA:
        p = sinkhorn_uot(ker, row_m, col_m, g)
        mixed = i019.mix(ens, p, ALPHA)
        name = f"uot_g{g:.2f}"
        rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"gamma": g})
        rec["delta_i019"] = round(rec["f1_100"] - rec1["f1_100"], 6)
        rows.append(rec)
        configs[name] = mixed
        print(f"  {name:14s} F1={rec['f1_100']:.4f} d_i019={rec['delta_i019']:+.4f}")

    print("[2/2] 半折核对...")
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
        if name == "i019_bal":
            oof_i019 = rec["f1_100"]
        print(f"  oof_{name:14s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    best_oof = max(oof_rows, key=lambda r: r["f1_100"])
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001 and best_oof["name"] != "oof_i019_bal")
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline_i019": rec1,
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
