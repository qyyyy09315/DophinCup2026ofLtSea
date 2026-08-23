"""
I011a-rrf3 — I011 派生：deep / 比对 / k-mer 三路 RRF
================================================================
依据:
  - RRF: Cormack et al., SIGIR 2009, doi:10.1145/1571941.1572114
  - I011：两路 deep vs 合成 sim，k=30 β=0.50 全 val +0.0044 / OOF +0.0055
自己的改法:
  合成 sim 把比对和 k-mer 捆成一路。拆开成三条名次再 RRF，
  弱同源时 k-mer 不再被 DIAMOND 名次盖住。混合仍是 CK005a + β·列归一 RRF。
  k ∈ {10,30,60}，β ∈ {0.20,0.50,0.75}。门槛相对 I011 全 val 0.4443，
  诚实口径看半折相对 I011 同协议。
用法:
  python code/I/I011a-rrf3.py
  python code/I/I011a-rrf3.py --gen-sub
输出:
  rel/I011a_report.json
  submit/submission_I011a_ensemble.csv（--gen-sub）
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    "i011", os.path.join(PROJECT_ROOT, "code", "I", "I011-rrf.py"))
i011 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i011)

DATA_DIR = i011.DATA_DIR
REL_DIR = i011.REL_DIR
SUB_DIR = i011.SUB_DIR
CK005A_TMP = i011.CK005A_TMP
TCK002_CONFIG = i011.TCK002_CONFIG
REPORT_PATH = os.path.join(REL_DIR, "I011a_report.json")

ALIGN_W = i011.ALIGN_W
SIM_W = i011.SIM_W
RANDOM_STATE = i011.RANDOM_STATE
K_GRID = (10, 30, 60)
BETA_GRID = (0.20, 0.50, 0.75)
I011_F1 = 0.444345


def rrf3(deep, bit, kmer, k):
    return (1.0 / (k + i011.ranks(deep) + 1)
            + 1.0 / (k + i011.ranks(bit) + 1)
            + 1.0 / (k + i011.ranks(kmer) + 1))


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print("I011a: 三路 RRF（deep / 比对 / k-mer）")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = ALIGN_W * sim_bit + (1 - ALIGN_W) * sim_kmer
    vp = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    base_ens = deep + SIM_W * sim
    rrf2 = i011.rrf_pair(deep, sim, 30)
    i011_ens = base_ens + 0.50 * i011.col_norm(rrf2)

    rows = []
    rec = i011.eval_ens(base_ens, y_val, hit, miss, "ck005a")
    rows.append(rec)
    rec = i011.eval_ens(i011_ens, y_val, hit, miss, "i011_k30_b0.50")
    rec["delta_i011"] = 0.0
    rows.append(rec)
    print(f"  ck005a {rows[0]['f1_100']:.4f}  i011 {rows[1]['f1_100']:.4f}")

    print("[1/2] 三路混合网格...")
    hybrids = {}
    for k in tqdm(K_GRID, desc="    k", unit="k"):
        rrf = rrf3(deep, sim_bit, sim_kmer, k)
        rrf_s = i011.col_norm(rrf)
        for b in BETA_GRID:
            ens = base_ens + b * rrf_s
            rec = i011.eval_ens(ens, y_val, hit, miss, f"hyb3_k{k}_b{b:.2f}",
                                {"k": k, "beta": b})
            rec["delta"] = round(rec["f1_100"] - rows[0]["f1_100"], 6)
            rec["delta_i011"] = round(rec["f1_100"] - rows[1]["f1_100"], 6)
            rows.append(rec)
            hybrids[(k, b)] = ens
            print(f"  {rec['name']:24s} F1={rec['f1_100']:.4f} "
                  f"d_a={rec['delta']:+.4f} d_i011={rec['delta_i011']:+.4f}")

    print("[2/2] 半折核对...")
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(ens):
        pred = np.zeros_like(ens, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            t = i011.search_t(ens[fit], y_val[fit])
            pred[hold] = (ens[hold] >= t).astype(np.int8)
        return pred

    oof_rows = []
    for name, ens in [("oof_ck005a", base_ens), ("oof_i011", i011_ens)]:
        pred = oof_pred(ens)
        rec = {"name": name,
               "f1_100": round(i011.macro_f1(pred, y_val), 6),
               "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6)}
        oof_rows.append(rec)
        print(f"  {name:24s} F1={rec['f1_100']:.4f}")
    oof_i011 = oof_rows[1]["f1_100"]

    for k in K_GRID:
        rrf = rrf3(deep, sim_bit, sim_kmer, k)
        rrf_s = i011.col_norm(rrf)
        for b in BETA_GRID:
            ens = base_ens + b * rrf_s
            pred = oof_pred(ens)
            rec = {
                "name": f"oof_hyb3_k{k}_b{b:.2f}",
                "k": k, "beta": b,
                "f1_100": round(i011.macro_f1(pred, y_val), 6),
                "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
            }
            rec["delta_i011"] = round(rec["f1_100"] - oof_i011, 6)
            oof_rows.append(rec)
            print(f"  {rec['name']:24s} F1={rec['f1_100']:.4f} d_i011={rec['delta_i011']:+.4f}")

    ranked = [r for r in rows if r["name"].startswith("hyb3_")]
    best = max(ranked, key=lambda r: r["f1_100"])
    best_oof = max((r for r in oof_rows if r["name"].startswith("oof_hyb3_")),
                   key=lambda r: r["f1_100"])
    gain_i011 = best["f1_100"] - rows[1]["f1_100"]
    gain_oof = best_oof["f1_100"] - oof_i011
    useful = bool(gain_oof >= 0.001)
    print(f"  全 val 最优 {best['name']} F1={best['f1_100']:.4f} vs I011 {gain_i011:+.4f}")
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} vs I011 {gain_oof:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "baseline_ck005a": rows[0],
        "baseline_i011": rows[1],
        "best": best,
        "best_oof": best_oof,
        "gain_vs_i011": round(gain_i011, 6),
        "gain_oof_vs_i011": round(gain_oof, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")

    if gen_sub:
        k_s, b_s = int(best["k"]), float(best["beta"])
        test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        tsv_test = os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
        full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
        idx_t, bits_t = i011.parse_aln(tsv_test, test_df["protein_id"].values, full_map)
        sim_bit_t = i011.agg_labels(idx_t, bits_t, y_full)
        sim_kmer_t = 0.0
        for kk, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
            if wt <= 0:
                continue
            d = np.load(os.path.join(
                REL_DIR, "TCK002a_neigh_full.npz" if kk == 6
                else f"CK004_neigh_k{kk}_full.npz"))
            n2 = d["idx_test"].shape[0]
            lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
            s = d["sim_test"][:, :5].astype(np.float32)
            ww = np.maximum(s, 1e-7) ** 2
            sim_kmer_t = sim_kmer_t + wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
        sim_t = ALIGN_W * sim_bit_t + (1.0 - ALIGN_W) * sim_kmer_t
        tp = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))
        t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
        t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))
        deep_t = w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"] + w["cnn"] * tp["cnn"]
        base_t = deep_t + SIM_W * sim_t
        rrf_t = rrf3(deep_t, sim_bit_t, sim_kmer_t, k_s)
        ens_t = base_t + b_s * i011.col_norm(rrf_t)
        t_sub = i011.search_t(hybrids[(k_s, b_s)], y_val)
        y_pred = (ens_t >= t_sub).astype(np.int8)
        os.makedirs(SUB_DIR, exist_ok=True)
        sub = pd.concat([test_df[["protein_id"]],
                         pd.DataFrame(y_pred, columns=label_cols)], axis=1)
        sub_path = os.path.join(SUB_DIR, "submission_I011a_ensemble.csv")
        sub.to_csv(sub_path, index=False)
        report["submission"] = sub_path
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  提交: {sub_path}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
