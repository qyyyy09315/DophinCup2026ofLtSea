"""
A009-bitgap — DIAMOND top-1 vs top-2 bitscore 间隔（分析性，CPU）
================================================================
EIHR-PROT 用 bitscore / 覆盖 / hit 数做同源可靠性。本脚本只看间隔：
gap = bits1-bits2（无第二命中则 =bits1）。不写腾讯表。
若强同源间隔中位明显高于远程，才值得按间隔调 ALIGN_W。
用法:
  python code/tools/A009-bitgap.py
输出:
  rel/A009_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
ck007 = load_mod("ck007", os.path.join("CK", "CK007-pident.py"))
REPORT_PATH = os.path.join(i011.REL_DIR, "A009_report.json")


def pct(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return None
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return [round(float(v), 1) for v in q]


def slice_rec(name, mask, gap, two):
    if mask.sum() == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(mask.sum()),
        "n_two": int(two[mask].sum()),
        "gap_q": pct(gap[mask]),
        "gap_median": round(float(np.median(gap[mask])), 1),
        "frac_gap50": round(float((gap[mask] >= 50).mean()), 4),
        "frac_gap100": round(float((gap[mask] >= 100).mean()), 4),
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("A009: DIAMOND bitscore gap")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx, bits, pid = ck007.parse_aln_pid(tsv, val_ids, train_map)
    hit = idx[:, 0] >= 0
    miss = ~hit
    pident = pid[:, 0].copy()
    pident[miss] = 0.0
    two = idx[:, 1] >= 0
    gap = np.where(two, bits[:, 0] - bits[:, 1], bits[:, 0])
    twilight = hit & (pident < 80) & (pident >= 50)
    remote = hit & (pident < 50)
    strong = hit & (pident >= 80)

    slices = []
    for name, mask in (
        ("all", np.ones(len(idx_val), bool)),
        ("hit", hit),
        ("miss", miss),
        ("strong>=80", strong),
        ("twilight50-80", twilight),
        ("remote<50", remote),
        ("two", two),
    ):
        rec = slice_rec(name, mask, gap, two)
        slices.append(rec)
        print(f"  {name:16s} n={rec['n']:5d} med={rec.get('gap_median')} "
              f"g50={rec.get('frac_gap50')} g100={rec.get('frac_gap100')}")

    by = {r["name"]: r for r in slices}
    worth = bool(
        by["strong>=80"]["gap_median"] >= 50
        and by["remote<50"]["gap_median"] + 20 <= by["strong>=80"]["gap_median"]
    )
    report = {
        "slices": slices,
        "worth_gap": worth,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "强同源间隔明显高于远程，可按间隔调 ALIGN_W"
            if worth else "间隔区分不够，按间隔调权价值低"
        ),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}  worth_gap={worth}")
    print("=" * 60)


if __name__ == "__main__":
    main()
