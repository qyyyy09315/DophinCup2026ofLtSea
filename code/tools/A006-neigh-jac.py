"""
A006-neigh-jac — 重叠 / 独有邻居相对查询的标签 Jaccard（分析性，CPU）
================================================================
CK022：弱同源确有 k-mer 独有邻居，并入后 remote 掉。本脚本看这些邻居
的标签是否贴近查询。不写腾讯表。
若独有 k-mer Jaccard 明显低于重叠邻居，才值得开交集过滤。
用法:
  python code/tools/A006-neigh-jac.py
输出:
  rel/A006_report.json
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
REPORT_PATH = os.path.join(i011.REL_DIR, "A006_report.json")


def jac(a, b):
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 1.0


def mean_jac(ids, y_train, yq):
    if not ids:
        return None
    return float(np.mean([jac(y_train[j].astype(bool), yq) for j in ids]))


def stats(mask, dsets, ksets, y_train, y_val, name):
    if mask.sum() == 0:
        return {"name": name, "n": 0}
    ov, ku, du = [], [], []
    for i in np.flatnonzero(mask):
        a, b = dsets[i], ksets[i]
        yq = y_val[i].astype(bool)
        o = a & b
        k_only = b - a
        d_only = a - b
        jo = mean_jac(o, y_train, yq)
        jk = mean_jac(k_only, y_train, yq)
        jd = mean_jac(d_only, y_train, yq)
        if jo is not None:
            ov.append(jo)
        if jk is not None:
            ku.append(jk)
        if jd is not None:
            du.append(jd)
    rec = {
        "name": name,
        "n": int(mask.sum()),
        "overlap_jac": round(float(np.mean(ov)), 4) if ov else None,
        "kmer_only_jac": round(float(np.mean(ku)), 4) if ku else None,
        "diamond_only_jac": round(float(np.mean(du)), 4) if du else None,
        "n_overlap": len(ov),
        "n_kmer_only": len(ku),
        "n_diamond_only": len(du),
    }
    return rec


def main():
    t0 = time.time()
    print("=" * 60)
    print("A006: neighbor label Jaccard overlap vs unique")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_d, _, pid_v = ck007.parse_aln_pid(tsv, val_ids, train_map)
    hit = idx_d[:, 0] >= 0
    miss = ~hit
    pident = pid_v[:, 0].copy()
    pident[miss] = 0.0
    twilight = hit & (pident < 80) & (pident >= 50)
    remote = hit & (pident < 50)
    strong = hit & (pident >= 80)

    d6 = np.load(os.path.join(i011.REL_DIR, "CK002_neigh_k6.npz"))
    dsets = [set(int(j) for j in idx_d[i, :3] if j >= 0) for i in range(len(idx_val))]
    ksets = [set(int(j) for j in d6["idx_val"][i, :5] if j >= 0) for i in range(len(idx_val))]

    slices = []
    for name, mask in (
        ("all", np.ones(len(idx_val), bool)),
        ("hit", hit),
        ("miss", miss),
        ("strong>=80", strong),
        ("twilight50-80", twilight),
        ("remote<50", remote),
    ):
        rec = stats(mask, dsets, ksets, y_train, y_val, name)
        slices.append(rec)
        print(f"  {name:16s} n={rec['n']:5d} ov={rec.get('overlap_jac')} "
              f"konly={rec.get('kmer_only_jac')} donly={rec.get('diamond_only_jac')}")

    by = {r["name"]: r for r in slices}
    tw, rem, st = by["twilight50-80"], by["remote<50"], by["strong>=80"]
    gap_tw = None
    gap_re = None
    if tw.get("overlap_jac") is not None and tw.get("kmer_only_jac") is not None:
        gap_tw = tw["overlap_jac"] - tw["kmer_only_jac"]
    if rem.get("overlap_jac") is not None and rem.get("kmer_only_jac") is not None:
        gap_re = rem["overlap_jac"] - rem["kmer_only_jac"]
    worth = bool((gap_tw is not None and gap_tw >= 0.05)
                 or (gap_re is not None and gap_re >= 0.05))
    report = {
        "slices": slices,
        "gap_twilight": round(gap_tw, 4) if gap_tw is not None else None,
        "gap_remote": round(gap_re, 4) if gap_re is not None else None,
        "worth_inter": worth,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "独有 k-mer 标签差于重叠，可开交集过滤"
            if worth else "独有 k-mer 标签不差于重叠，交集过滤价值低"
        ),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}  worth_inter={worth}")
    print("=" * 60)


if __name__ == "__main__":
    main()
