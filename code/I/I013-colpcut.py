"""
I013-colpcut — 按列 PCut：每标签按训练先验截断，再与 SCut 相交
================================================================
依据:
  - Yang Y, SIGIR 2001, doi:10.1145/383952.383975
    三种截断：SCut（逐标签阈值）、RCut（按行取 k）、PCut（按列取比例）。
  - I007 名义 PCut，实际是**行内分位数**，不是 Yang 的列比例。
  - I006/I008 行基数截断已关。SCut 已是 CK005a/I011 默认。
自己的改法:
  冻 I011 分数（k=30 β=0.50）。训练正例率 π_j，每列保留分数最高的
  round(π_j · N · ρ) 个。ρ∈{0.8,1.0,1.2,1.5}。
  与 SCut 相交（两边都同意才正）；空列回退 SCut。
  半折：π 与截断人数用拟合折估，评估折套用。门槛相对 I011 半折 ≥ +0.001。
用法:
  python code/I/I013-colpcut.py
输出:
  rel/I013_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I013_report.json")
K, BETA = 30, 0.50
RHO = (0.8, 1.0, 1.2, 1.5)
N_LAB = i011.N_LABELS


def col_keep(ens, pi, n_rows, rho):
    """每列保留 top round(π·n·ρ)，至少 1。返回 bool (n, L)。"""
    keep = np.zeros(ens.shape, dtype=bool)
    for j in range(N_LAB):
        k = max(1, int(round(float(pi[j]) * n_rows * rho)))
        k = min(k, n_rows)
        order = np.argpartition(-ens[:, j], kth=k - 1)[:k]
        keep[order, j] = True
    return keep


def build_i011(deep, sim):
    return deep + i011.SIM_W * sim + BETA * i011.col_norm(i011.rrf_pair(deep, sim, K))


def main():
    t0 = time.time()
    print("=" * 60)
    print("I013: 按列 PCut × SCut 相交")

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
    ens = build_i011(deep, sim)
    n = len(y_val)
    pi_tr = y_train.mean(0)

    t_scut = i011.search_t(ens, y_val)
    scut = ens >= t_scut
    rec0 = i011.eval_ens(ens, y_val, hit, miss, "i011_scut")
    print(f"  I011+SCut F1={rec0['f1_100']:.4f} card={rec0['pred_card']:.1f}")

    rows = [rec0]
    print("[1/2] 全 val 网格...")
    for rho in RHO:
        keep = col_keep(ens, pi_tr, n, rho)
        pred = (scut & keep).astype(np.int8)
        empty = pred.sum(0) == 0
        pred[:, empty] = scut[:, empty]
        rec = {
            "name": f"inter_rho{rho:.1f}",
            "rho": rho,
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
            "f1_hit": round(i011.macro_f1(pred[hit], y_val[hit]), 6),
            "pred_card": round(float(pred.sum(1).mean()), 3),
            "n_empty_fallback": int(empty.sum()),
        }
        rec["delta"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        print(f"  {rec['name']:18s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f} "
              f"card={rec['pred_card']:.1f} fallback={rec['n_empty_fallback']}")

        pred_r = keep.astype(np.int8)
        rec = {
            "name": f"replace_rho{rho:.1f}",
            "rho": rho,
            "f1_100": round(i011.macro_f1(pred_r, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred_r[miss], y_val[miss]), 6),
            "pred_card": round(float(pred_r.sum(1).mean()), 3),
        }
        rec["delta"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        print(f"  {rec['name']:18s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f}")

    print("[2/2] 半折核对...")
    rng = np.random.RandomState(i011.RANDOM_STATE)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_scut(ens_use):
        pred = np.zeros(ens_use.shape, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            t = i011.search_t(ens_use[fit], y_val[fit])
            pred[hold] = (ens_use[hold] >= t).astype(np.int8)
        return pred

    oof_base_pred = oof_scut(ens)
    oof_base = i011.macro_f1(oof_base_pred, y_val)
    print(f"  oof_i011_scut F1={oof_base:.4f}")
    oof_rows = [{"name": "oof_i011_scut", "f1_100": round(oof_base, 6), "delta": 0.0}]

    for rho in tqdm(RHO, desc="    ρ", unit="r"):
        pred = np.zeros(ens.shape, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            t = i011.search_t(ens[fit], y_val[fit])
            sc = ens[hold] >= t
            pi = y_val[fit].mean(0)
            keep = col_keep(ens[hold], pi, len(hold), rho)
            p = sc & keep
            empty = p.sum(0) == 0
            p[:, empty] = sc[:, empty]
            pred[hold] = p.astype(np.int8)
        rec = {
            "name": f"oof_inter_rho{rho:.1f}",
            "rho": rho,
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        rec["delta"] = round(rec["f1_100"] - oof_base, 6)
        oof_rows.append(rec)
        print(f"  {rec['name']:22s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f}")

    best_oof = max(oof_rows, key=lambda r: r["f1_100"])
    gain = best_oof["f1_100"] - oof_base
    useful = bool(gain >= 0.001 and best_oof["name"] != "oof_i011_scut")
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"Δ={gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline": rec0,
        "best_oof": best_oof,
        "gain_oof": round(gain, 6),
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
