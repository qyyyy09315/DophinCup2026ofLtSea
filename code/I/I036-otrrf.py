"""
I036-otrrf — I011 一元分数与 Sinkhorn 传输矩阵做 RRF，不再分数插值
================================================================
依据:
  - Cormack, Clarke & Buettcher, SIGIR 2009, doi:10.1145/1571941.1572114
    RRF 融两路排序，不校准分数。I011 因此过门槛。
  - Hsu & Taksa, Information Retrieval 2005, doi:10.1007/s10791-005-6994-4
    秩组合常优于分数组合，尤其两路量纲不同时。
  - I019：ens'=(1-α)·I011 + α·col_norm(P)，是分数插值。
    I031 把 Sinkhorn 放到融合前，负。本实验仍在 I011 之后，
    只改 I011 与 P 的组合方式。
  - I035：I008×I011 系统级 RRF，两套相关太高。I011 与 P 不是
    同一套分数的截断，P 是双边约束后的传输。
自己的改法:
  冻 I011。按 I019 锁参出 P（τ=0.20 ρ=1.0）。
  ens' = I011 + β·col_norm(RRF(I011, P))，k=30，β∈{0.20,0.50,0.75}。
  对照 I019 分数插值。另混 (1-γ)·I019 + γ·ens'，γ∈{0.15,0.30}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I036-otrrf.py
输出:
  rel/I036_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I036_report.json")
K_RRF, BETA_I011 = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
BETA_GRID = (0.20, 0.50, 0.75)
G_GRID = (0.15, 0.30)


def build_i011_p(train_df, y_full, idx_train, idx_val):
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
    s011 = deep + i011.SIM_W * sim + BETA_I011 * i011.col_norm(
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
    return s011, p, s019, hit, y_train, y_val


def main():
    t0 = time.time()
    print("=" * 60)
    print("I036: I011 × Sinkhorn P 秩融合")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s011, p, s019, hit, y_train, y_val = build_i011_p(
        train_df, y_full, idx_train, idx_val)
    miss = ~hit
    print(f"  hit {int(hit.sum())}/{len(hit)}")

    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    rec011 = i011.eval_ens(s011, y_val, hit, miss, "i011")
    rec_p = i011.eval_ens(p, y_val, hit, miss, "P", scale=True)
    corr = float(np.corrcoef(s011.ravel(), p.ravel())[0, 1])
    print(f"  I019 F1={rec0['f1_100']:.4f}  I011 F1={rec011['f1_100']:.4f}  "
          f"P F1={rec_p['f1_100']:.4f}  corr(I011,P)={corr:.3f}")

    rows = [rec0, rec011, rec_p]
    configs = {"i019": s019, "i011": s011}

    print("[1/2] β × γ ...")
    rrf = i011.rrf_pair(s011, p, K_RRF)
    rrf_n = i011.col_norm(rrf)
    for b in tqdm(BETA_GRID, desc="    β", unit="b"):
        ens = s011 + b * rrf_n
        rec = i011.eval_ens(ens, y_val, hit, miss, f"rrf_b{b:.2f}", {"beta": b})
        rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        configs[f"rrf_b{b:.2f}"] = ens
        print(f"  rrf_b{b:.2f} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")
        s_n = i011.col_norm(ens)
        for g in G_GRID:
            mixed = (1.0 - g) * s019 + g * s_n
            name = f"mix_b{b:.2f}_g{g:.2f}"
            rec_m = i011.eval_ens(mixed, y_val, hit, miss, name, {"beta": b, "gamma": g})
            rec_m["delta_i019"] = round(rec_m["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec_m)
            configs[name] = mixed
            print(f"  {name:20s} F1={rec_m['f1_100']:.4f} d={rec_m['delta_i019']:+.4f}")

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
        print(f"  oof_{name:20s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    cand = [r for r in oof_rows if r["name"] not in ("oof_i019", "oof_i011")]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline_i019": rec0,
        "i011": rec011,
        "P": rec_p,
        "corr_i011_p": round(corr, 4),
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
