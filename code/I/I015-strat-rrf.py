"""
I015-strat-rrf — 头尾标签组内排名后再 RRF
================================================================
依据:
  - Liu & Croft, SIGIR 2004, doi:10.1145/1008992.1009026
    先分簇再在簇内检索，避免全局排序被大簇主导。
  - I010：尾标签更信 sim、头更信 CNN（三分位权 1.125 / 0.5）。
  - I011 全局排 500 标签，尾标签名次被头标签挤到很后。
自己的改法:
  训练频次三分位：头/中/尾。每组内对 deep、sim 各自排名（0=组内最高），
  再按 I011 的 k=30 做组内 RRF，拼回 500 维后列归一。
  对照：只分层 sim / 只分层 deep / 两组都分层。
  混合仍 CK005a + β=0.50。门槛相对 I011 半折 ≥ +0.001。
用法:
  python code/I/I015-strat-rrf.py
输出:
  rel/I015_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I015_report.json")
K_RRF, BETA = 30, 0.50
N = i011.N_LABELS


def freq_bins(pos):
    active = pos > 0
    q1, q2 = np.quantile(pos[active], [1 / 3, 2 / 3])
    bins = np.zeros(N, np.int32)
    bins[pos <= q1] = 0
    bins[(pos > q1) & (pos < q2)] = 1
    bins[pos >= q2] = 2
    bins[~active] = 1
    return bins, float(q1), float(q2)


def ranks_in_groups(score, bins):
    """组内排名，0=该组最高；其他组位置填一个大数不参与。"""
    out = np.full(score.shape, 10_000, dtype=np.int32)
    for b in (0, 1, 2):
        cols = np.where(bins == b)[0]
        if cols.size == 0:
            continue
        sub = score[:, cols]
        r = np.argsort(np.argsort(-sub, axis=1, kind="mergesort"), axis=1)
        out[:, cols] = r
    return out


def rrf_from_ranks(rd, rs, k):
    return 1.0 / (k + rd + 1) + 1.0 / (k + rs + 1)


def main():
    t0 = time.time()
    print("=" * 60)
    print("I015: 头尾组内排名 RRF")

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
    rd_g = i011.ranks(deep)
    rs_g = i011.ranks(sim)
    i011_ens = base + BETA * i011.col_norm(rrf_from_ranks(rd_g, rs_g, K_RRF))

    bins, q1, q2 = freq_bins(y_train.sum(0).astype(np.float64))
    print(f"  q1={q1:.0f} q2={q2:.0f}  尾 {(bins==0).sum()} 中 {(bins==1).sum()} 头 {(bins==2).sum()}")
    rd_s = ranks_in_groups(deep, bins)
    rs_s = ranks_in_groups(sim, bins)

    rows = []
    rec = i011.eval_ens(base, y_val, hit, miss, "ck005a")
    rows.append(rec)
    rec = i011.eval_ens(i011_ens, y_val, hit, miss, "i011")
    rows.append(rec)
    print(f"  ck005a {rows[0]['f1_100']:.4f}  i011 {rows[1]['f1_100']:.4f}")

    print("[1/2] 分层组合...")
    configs = {"i011": i011_ens}
    variants = {
        "strat_both": rrf_from_ranks(rd_s, rs_s, K_RRF),
        "strat_sim": rrf_from_ranks(rd_g, rs_s, K_RRF),
        "strat_deep": rrf_from_ranks(rd_s, rs_g, K_RRF),
    }
    for name, rrf in variants.items():
        ens = base + BETA * i011.col_norm(rrf)
        rec = i011.eval_ens(ens, y_val, hit, miss, name)
        rec["delta_i011"] = round(rec["f1_100"] - rows[1]["f1_100"], 6)
        rows.append(rec)
        configs[name] = ens
        print(f"  {name:14s} F1={rec['f1_100']:.4f} d_i011={rec['delta_i011']:+.4f}")

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
        print(f"  oof_{name:14s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i011"] = round(rec["f1_100"] - oof_i011, 6)

    best_oof = max(oof_rows, key=lambda r: r["f1_100"])
    gain = best_oof["f1_100"] - oof_i011
    useful = bool(gain >= 0.001 and best_oof["name"] != "oof_i011")
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I011 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "q1": q1, "q2": q2,
        "n_bin": [int((bins == b).sum()) for b in (0, 1, 2)],
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
