"""
I011d-qdep — I011 派生：按实例调 β（查询相关融合）
================================================================
依据:
  - Vogt & Cottrell, Information Retrieval 1999, doi:10.1023/a:1009980820262
    多路分数线性组合，权随查询变：系统越不一致，融合越有用。
  - I011：全局 β=0.50，k=30，全 val +0.0044 / OOF +0.0055
  - I011a/b/c：拆路、分 hit/miss、CombMNZ 均未超；公式锁 1/(k+rank)
自己的改法:
  冻 k=30、RRF 公式、全样本写回。每条蛋白算 deep 与 sim 的 top-M 标签重合
  Jaccard。重合低（两路打架）用高 β，重合高用低 β。
  M∈{20,50}，β_lo/β_hi ∈ {0.20,0.50,0.75} 且 lo≤hi。
  门槛相对 I011 半折 ≥ +0.001。
用法:
  python code/I/I011d-qdep.py
输出:
  rel/I011d_report.json
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
_spec = importlib.util.spec_from_file_location(
    "i011", os.path.join(PROJECT_ROOT, "code", "I", "I011-rrf.py"))
i011 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i011)

REPORT_PATH = os.path.join(i011.REL_DIR, "I011d_report.json")
K = 30
M_GRID = (20, 50)
BETA_GRID = (0.20, 0.50, 0.75)


def top_jaccard(deep, sim, m):
    rd, rs = i011.ranks(deep), i011.ranks(sim)
    n = deep.shape[0]
    jac = np.empty(n, np.float32)
    for i in range(n):
        a = set(np.flatnonzero(rd[i] < m))
        b = set(np.flatnonzero(rs[i] < m))
        u = len(a | b)
        jac[i] = (len(a & b) / u) if u else 0.0
    return jac


def mix_beta(base, rrf_s, beta):
    return base + beta[:, None] * rrf_s


def main():
    t0 = time.time()
    print("=" * 60)
    print("I011d: 按实例 Jaccard 调 β")

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
    rrf_s = i011.col_norm(i011.rrf_pair(deep, sim, K))
    i011_ens = base + 0.50 * rrf_s

    rows = []
    rec = i011.eval_ens(base, y_val, hit, miss, "ck005a")
    rows.append(rec)
    rec = i011.eval_ens(i011_ens, y_val, hit, miss, "i011_k30_b0.50")
    rows.append(rec)
    print(f"  ck005a {rows[0]['f1_100']:.4f}  i011 {rows[1]['f1_100']:.4f}")

    print("[1/2] Jaccard 二分 β 网格...")
    configs = {"i011": i011_ens}
    for m in M_GRID:
        jac = top_jaccard(deep, sim, m)
        med = float(np.median(jac))
        low = jac < med
        print(f"  M={m} Jaccard median={med:.3f} 低重合 {int(low.sum())}/{len(jac)}")
        for blo in BETA_GRID:
            for bhi in BETA_GRID:
                if bhi < blo:
                    continue
                beta = np.where(low, bhi, blo).astype(np.float32)
                ens = mix_beta(base, rrf_s, beta)
                name = f"q_M{m}_lo{blo:.2f}_hi{bhi:.2f}"
                rec = i011.eval_ens(ens, y_val, hit, miss, name,
                                    {"M": m, "beta_lo": blo, "beta_hi": bhi,
                                     "jac_med": round(med, 4)})
                rec["delta_i011"] = round(rec["f1_100"] - rows[1]["f1_100"], 6)
                rows.append(rec)
                configs[name] = ens
                print(f"  {name:28s} F1={rec['f1_100']:.4f} d_i011={rec['delta_i011']:+.4f}")

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
    for name, ens in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(ens)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "i011":
            oof_i011 = rec["f1_100"]
        print(f"  oof_{name:28s} F1={rec['f1_100']:.4f}")
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
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
