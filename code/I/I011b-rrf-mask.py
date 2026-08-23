"""
I011b-rrf-mask — I011 派生：RRF 只作用在 hit 或 miss
================================================================
依据:
  - I011：全样本 k=30 β=0.50，全 val +0.0044 / OOF +0.0055
  - I011a：拆比对/k-mer 未过 +0.001，合成 sim 不拆
自己的改法:
  冻 k=30 β=0.50。混合分数只写回 miss / 只写回 hit / 全写（=I011）。
  弱同源更吃名次、强同源已有 bitscore，分着试。半折同协议。
  门槛相对 I011 OOF ≥ +0.001。
用法:
  python code/I/I011b-rrf-mask.py
输出:
  rel/I011b_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    "i011", os.path.join(PROJECT_ROOT, "code", "I", "I011-rrf.py"))
i011 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(i011)

K, BETA = 30, 0.50
REPORT_PATH = os.path.join(i011.REL_DIR, "I011b_report.json")


def apply_mask(base, hyb, mask):
    out = base.copy()
    out[mask] = hyb[mask]
    return out


def main():
    t0 = time.time()
    print("=" * 60)
    print("I011b: RRF 只写 hit / 只写 miss")

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
    hyb = base + BETA * i011.col_norm(i011.rrf_pair(deep, sim, K))

    rows = []
    for name, ens in [
        ("ck005a", base),
        ("i011_all", hyb),
        ("rrf_hit_only", apply_mask(base, hyb, hit)),
        ("rrf_miss_only", apply_mask(base, hyb, miss)),
    ]:
        rec = i011.eval_ens(ens, y_val, hit, miss, name)
        rec["delta_i011"] = round(rec["f1_100"] - 0.444345, 6) if name != "ck005a" else 0.0
        rows.append(rec)
        print(f"  {name:16s} F1={rec['f1_100']:.4f} miss={rec['f1_miss']:.4f} "
              f"hit={rec['f1_hit']:.4f}")

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
    configs = {
        "oof_ck005a": base,
        "oof_i011_all": hyb,
        "oof_hit_only": apply_mask(base, hyb, hit),
        "oof_miss_only": apply_mask(base, hyb, miss),
    }
    oof_i011 = None
    for name, ens in configs.items():
        pred = oof_pred(ens)
        rec = {
            "name": name,
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
            "f1_hit": round(i011.macro_f1(pred[hit], y_val[hit]), 6),
        }
        oof_rows.append(rec)
        if name == "oof_i011_all":
            oof_i011 = rec["f1_100"]
        print(f"  {name:16s} F1={rec['f1_100']:.4f} miss={rec['f1_miss']:.4f}")

    for rec in oof_rows:
        rec["delta_i011"] = round(rec["f1_100"] - oof_i011, 6)

    best_oof = max((r for r in oof_rows if r["name"] != "oof_ck005a"),
                   key=lambda r: r["f1_100"])
    gain = best_oof["f1_100"] - oof_i011
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I011 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "grid": rows,
        "oof_grid": oof_rows,
        "best_oof": best_oof,
        "gain_oof_vs_i011": round(gain, 6),
        "useful": useful,
        "seconds": round(time.time() - t0, 1),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
