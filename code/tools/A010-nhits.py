"""
A010-nhits — DIAMOND 命中数按 pident 切片（分析性，CPU）
================================================================
EIHR-PROT 用 hit 数做同源可靠性。TSV 最多收 5 个邻居。本脚本看
n_hits 是否随 pident 分层。不写腾讯表。
若强同源 n≥4 的比例明显高于远程，才值得按命中数调 ALIGN_W。
用法:
  python code/tools/A010-nhits.py
输出:
  rel/A010_report.json
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
REPORT_PATH = os.path.join(i011.REL_DIR, "A010_report.json")


def rec(name, mask, n_hits):
    if mask.sum() == 0:
        return {"name": name, "n": 0}
    h = n_hits[mask]
    return {
        "name": name,
        "n": int(mask.sum()),
        "mean": round(float(h.mean()), 3),
        "median": round(float(np.median(h)), 3),
        "frac_ge3": round(float((h >= 3).mean()), 4),
        "frac_ge4": round(float((h >= 4).mean()), 4),
        "frac_ge5": round(float((h >= 5).mean()), 4),
        "hist": {str(k): int((h == k).sum()) for k in range(6)},
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("A010: DIAMOND hit count by pident")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx, _, pid = ck007.parse_aln_pid(tsv, val_ids, train_map)
    hit = idx[:, 0] >= 0
    miss = ~hit
    pident = pid[:, 0].copy()
    pident[miss] = 0.0
    n_hits = (idx >= 0).sum(1)
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
    ):
        r = rec(name, mask, n_hits)
        slices.append(r)
        print(f"  {name:16s} n={r['n']:5d} mean={r.get('mean')} "
              f"ge4={r.get('frac_ge4')} ge5={r.get('frac_ge5')}")

    by = {r["name"]: r for r in slices}
    worth = bool(
        by["strong>=80"]["frac_ge4"] - by["remote<50"]["frac_ge4"] >= 0.10
        and by["strong>=80"]["frac_ge4"] >= 0.50
    )
    report = {
        "slices": slices,
        "worth_nhits": worth,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "强同源命中数明显高于远程，可按 n_hits 调 ALIGN_W"
            if worth else "命中数分层不够，按 n_hits 调权价值低"
        ),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}  worth_nhits={worth}")
    print("=" * 60)


if __name__ == "__main__":
    main()
