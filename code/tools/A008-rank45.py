"""
A008-rank45 — DIAMOND 第 4–5 名相对查询的标签 Jaccard（分析性，CPU）
================================================================
CK005c 全局 bit_top5 只 +0.0005。本脚本按 pident 切片看第 4–5 名
标签是否还贴近查询。不写腾讯表。
若 remote 第 4–5 名 Jaccard ≥0.18 且不低于 top-3 太多，才值得开弱 hit top-5。
用法:
  python code/tools/A008-rank45.py
输出:
  rel/A008_report.json
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
REPORT_PATH = os.path.join(i011.REL_DIR, "A008_report.json")


def jac(a, b):
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 1.0


def mean_ranks(mask, idx, y_train, y_val, ranks):
    vals = []
    n_has = 0
    for i in np.flatnonzero(mask):
        yq = y_val[i].astype(bool)
        js = []
        for r in ranks:
            t = int(idx[i, r])
            if t < 0:
                continue
            js.append(jac(y_train[t].astype(bool), yq))
        if js:
            vals.append(float(np.mean(js)))
            n_has += 1
    if not vals:
        return None, 0
    return round(float(np.mean(vals)), 4), n_has


def stats(mask, idx, y_train, y_val, name):
    if mask.sum() == 0:
        return {"name": name, "n": 0}
    j1, n1 = mean_ranks(mask, idx, y_train, y_val, (0,))
    j13, n13 = mean_ranks(mask, idx, y_train, y_val, (0, 1, 2))
    j45, n45 = mean_ranks(mask, idx, y_train, y_val, (3, 4))
    rec = {
        "name": name,
        "n": int(mask.sum()),
        "jac_r1": j1,
        "jac_r1_3": j13,
        "jac_r4_5": j45,
        "n_r1": n1,
        "n_r1_3": n13,
        "n_r4_5": n45,
        "frac_r4": round(n45 / int(mask.sum()), 4) if mask.sum() else 0.0,
    }
    return rec


def main():
    t0 = time.time()
    print("=" * 60)
    print("A008: DIAMOND rank 4-5 label Jaccard")

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

    slices = []
    for name, mask in (
        ("all", np.ones(len(idx_val), bool)),
        ("hit", hit),
        ("strong>=80", strong),
        ("twilight50-80", twilight),
        ("remote<50", remote),
    ):
        rec = stats(mask, idx_d, y_train, y_val, name)
        slices.append(rec)
        print(f"  {name:16s} n={rec['n']:5d} r1={rec.get('jac_r1')} "
              f"r13={rec.get('jac_r1_3')} r45={rec.get('jac_r4_5')} "
              f"frac4={rec.get('frac_r4')}")

    by = {r["name"]: r for r in slices}
    rem = by["remote<50"]
    tw = by["twilight50-80"]
    worth = bool(
        (rem.get("jac_r4_5") is not None and rem["jac_r4_5"] >= 0.18
         and rem.get("frac_r4", 0) >= 0.30)
        or (tw.get("jac_r4_5") is not None and tw["jac_r4_5"] >= 0.22
            and tw.get("frac_r4", 0) >= 0.40)
    )
    report = {
        "slices": slices,
        "worth_top5": worth,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "弱同源第 4–5 名仍有标签信号，可开 top-5"
            if worth else "第 4–5 名标签差或覆盖低，top-5 价值低"
        ),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}  worth_top5={worth}")
    print("=" * 60)


if __name__ == "__main__":
    main()
