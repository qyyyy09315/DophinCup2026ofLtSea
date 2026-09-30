"""
A005-overlap — DIAMOND 与 k-mer 邻居重叠按 pident 切片（分析性，CPU）
================================================================
CK005f：hit 上 DIAMOND top-1 ∈ k6 top-5 为 0.785。本脚本按
pident≥80 / 50–80 / <50 / miss 切开，看弱同源是否还有独有邻居。
不写腾讯表。若 twilight Jaccard 明显低于 strong，才值得开并集检索。
用法:
  python code/tools/A005-overlap.py
输出:
  rel/A005_report.json
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
ck007 = load_mod("ck007", os.path.join("CK", "CK007-pident.py"))
REPORT_PATH = os.path.join(i011.REL_DIR, "A005_report.json")


def set_rows(idx, k):
    out = []
    for i in range(idx.shape[0]):
        out.append(set(int(j) for j in idx[i, :k] if j >= 0))
    return out


def stats(mask, dsets, ksets, d1, k5, name):
    if mask.sum() == 0:
        return {"name": name, "n": 0}
    jac, top1_in, uniq_k, uniq_d = [], [], [], []
    for i in np.flatnonzero(mask):
        a, b = dsets[i], ksets[i]
        u = len(a | b)
        jac.append((len(a & b) / u) if u else 0.0)
        uniq_k.append(len(b - a))
        uniq_d.append(len(a - b))
        top1_in.append(1.0 if d1[i] >= 0 and d1[i] in k5[i] else 0.0)
    return {
        "name": name,
        "n": int(mask.sum()),
        "jaccard_d3_k5": round(float(np.mean(jac)), 4),
        "d1_in_k5": round(float(np.mean(top1_in)), 4),
        "kmer_only_mean": round(float(np.mean(uniq_k)), 3),
        "diamond_only_mean": round(float(np.mean(uniq_d)), 3),
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("A005: DIAMOND vs k-mer overlap by pident")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_d, bits_d, pid_v = ck007.parse_aln_pid(tsv, val_ids, train_map)
    hit = idx_d[:, 0] >= 0
    miss = ~hit
    pident = pid_v[:, 0].copy()
    pident[miss] = 0.0
    slices = {
        "all": np.ones(len(hit), bool),
        "hit": hit,
        "miss": miss,
        "strong>=80": hit & (pident >= 80),
        "twilight50-80": hit & (pident < 80) & (pident >= 50),
        "remote<50": hit & (pident < 50),
    }
    d6 = np.load(os.path.join(i011.REL_DIR, "CK002_neigh_k6.npz"))
    idx_k = d6["idx_val"].astype(np.int64)
    dsets = set_rows(idx_d, 3)
    ksets = set_rows(idx_k, 5)
    k5 = [set(int(j) for j in idx_k[i, :5] if j >= 0) for i in range(len(hit))]
    d1 = idx_d[:, 0]
    rows = []
    for name, mask in slices.items():
        rec = stats(mask, dsets, ksets, d1, k5, name)
        rows.append(rec)
        print(f"  {name:16s} n={rec.get('n', 0):5d} jac={rec.get('jaccard_d3_k5', 0):.3f} "
              f"d1_in_k5={rec.get('d1_in_k5', 0):.3f} kmer_only={rec.get('kmer_only_mean', 0):.2f}")
    tw = next(r for r in rows if r["name"] == "twilight50-80")
    st = next(r for r in rows if r["name"] == "strong>=80")
    worth = bool(tw.get("jaccard_d3_k5", 1) + 0.05 < st.get("jaccard_d3_k5", 0)
                 and tw.get("kmer_only_mean", 0) >= 1.0)
    report = {
        "slices": rows,
        "worth_union": worth,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "弱同源 k-mer 独有邻居明显多于强同源，可开并集检索"
            if worth else "切片重叠与强同源接近，并集检索预期重复 CK005f"
        ),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}  worth_union={worth}")
    print("=" * 60)


if __name__ == "__main__":
    main()
