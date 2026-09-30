"""
I017-prod — Kittler 积/和规则：CK005a 与 I011 分数层组合后再 SCut
================================================================
依据:
  - Kittler et al., IEEE TPAMI 1998, doi:10.1109/34.667881
    后验的积规则（product）与和规则（sum）。I016 是决策层投票，
    两套 SCut 几乎重合；分数层组合仍可能改排序。
  - I011 = CK005a + 0.50·列归一 RRF，已是加性；积规则不是同一单调变换。
自己的改法:
  冻两路分数。prod = √(a·b)；sum = (a+b)/2；max；min。
  各自 100 档 SCut。半折阈值。门槛相对 I011 半折 ≥ +0.001。
用法:
  python code/I/I017-prod.py
输出:
  rel/I017_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I017_report.json")
K, BETA = 30, 0.50


def main():
    t0 = time.time()
    print("=" * 60)
    print("I017: 分数层积/和规则")

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
    ens_a = deep + i011.SIM_W * sim
    ens_b = ens_a + BETA * i011.col_norm(i011.rrf_pair(deep, sim, K))
    a = np.clip(ens_a, 1e-8, None)
    b = np.clip(ens_b, 1e-8, None)

    configs = {
        "ck005a": ens_a,
        "i011": ens_b,
        "prod": np.sqrt(a * b),
        "sum": 0.5 * (ens_a + ens_b),
        "max": np.maximum(ens_a, ens_b),
        "min": np.minimum(ens_a, ens_b),
    }

    rows = []
    print("[1/2] 全 val ...")
    for name, ens in configs.items():
        rec = i011.eval_ens(ens, y_val, hit, miss, name)
        rec["delta_i011"] = round(rec["f1_100"] - 0.444345, 6)
        rows.append(rec)
        print(f"  {name:8s} F1={rec['f1_100']:.4f} d={rec['delta_i011']:+.4f} "
              f"miss={rec['f1_miss']:.4f}")

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
        print(f"  oof_{name:8s} F1={rec['f1_100']:.4f}")
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
