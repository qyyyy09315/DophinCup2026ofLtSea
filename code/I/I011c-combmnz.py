"""
I011c-combmnz — I011 派生：CombSUM / CombMNZ 对照 RRF
================================================================
依据:
  - CombMNZ: Fox & Shaw, TREC-2 1994, doi:10.6028/nist.sp.500-215.adhoc-vt
    多路分数先归一再求和，再乘「命中路数」。
  - I011：RRF 1/(k+rank) 全 val +0.0044 / OOF +0.0055
  - I011a/b：拆路、分 hit/miss 都未超 I011
自己的改法:
  每条蛋白把 500 标签按 deep、sim 排名换成百分位 p=(500-rank)/500。
  CombSUM = p_deep + p_sim。
  CombMNZ = CombSUM × (1[rank_deep<M] + 1[rank_sim<M])，M∈{50,100,200}。
  混合仍是 CK005a + β·列归一，β∈{0.20,0.50}，全样本写回。
  门槛相对 I011 半折 ≥ +0.001。
用法:
  python code/I/I011c-combmnz.py
输出:
  rel/I011c_report.json
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
_spec = importlib.util.spec_from_file_location(
    "i011", os.path.join(PROJECT_ROOT, "code", "I", "I011-rrf.py"))
i011 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i011)

REPORT_PATH = os.path.join(i011.REL_DIR, "I011c_report.json")
N = i011.N_LABELS
M_GRID = (50, 100, 200)
BETA_GRID = (0.20, 0.50)


def pct(score):
    return (N - i011.ranks(score)) / float(N)


def combsum(deep, sim):
    return pct(deep) + pct(sim)


def combmnz(deep, sim, m):
    rd, rs = i011.ranks(deep), i011.ranks(sim)
    hits = (rd < m).astype(np.float32) + (rs < m).astype(np.float32)
    return combsum(deep, sim) * np.maximum(hits, 1.0)


def main():
    t0 = time.time()
    print("=" * 60)
    print("I011c: CombSUM / CombMNZ 对照 RRF")

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
    base = deep + i011.SIM_W * sim
    i011_ens = base + 0.50 * i011.col_norm(i011.rrf_pair(deep, sim, 30))

    rows = []
    rec = i011.eval_ens(base, y_val, hit, miss, "ck005a")
    rows.append(rec)
    rec = i011.eval_ens(i011_ens, y_val, hit, miss, "i011_k30_b0.50")
    rows.append(rec)
    print(f"  ck005a {rows[0]['f1_100']:.4f}  i011 {rows[1]['f1_100']:.4f}")

    print("[1/2] CombSUM / CombMNZ 网格...")
    configs = {"i011": i011_ens}
    for b in BETA_GRID:
        ens = base + b * i011.col_norm(combsum(deep, sim))
        rec = i011.eval_ens(ens, y_val, hit, miss, f"sum_b{b:.2f}", {"beta": b})
        rec["delta_i011"] = round(rec["f1_100"] - rows[1]["f1_100"], 6)
        rows.append(rec)
        configs[rec["name"]] = ens
        print(f"  {rec['name']:20s} F1={rec['f1_100']:.4f} d_i011={rec['delta_i011']:+.4f}")
    for m in tqdm(M_GRID, desc="    M", unit="m"):
        cm = combmnz(deep, sim, m)
        for b in BETA_GRID:
            ens = base + b * i011.col_norm(cm)
            rec = i011.eval_ens(ens, y_val, hit, miss, f"mnz_M{m}_b{b:.2f}",
                                {"M": m, "beta": b})
            rec["delta_i011"] = round(rec["f1_100"] - rows[1]["f1_100"], 6)
            rows.append(rec)
            configs[rec["name"]] = ens
            print(f"  {rec['name']:20s} F1={rec['f1_100']:.4f} d_i011={rec['delta_i011']:+.4f}")

    print("[2/2] 半折核对...")
    rng = np.random.RandomState(i011.RANDOM_STATE)
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
    oof_i011 = None
    for name, ens in configs.items():
        pred = oof_pred(ens)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "i011":
            oof_i011 = rec["f1_100"]
        print(f"  oof_{name:18s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i011"] = round(rec["f1_100"] - oof_i011, 6)

    best_oof = max(oof_rows, key=lambda r: r["f1_100"])
    gain = best_oof["f1_100"] - oof_i011
    useful = bool(gain >= 0.001 and best_oof["name"] != "oof_i011")
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I011 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline_i011": rows[1],
        "best_oof": best_oof,
        "gain_oof_vs_i011": round(gain, 6),
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
