"""
I034-rowtau — I011 上按行自适应 Sinkhorn 温度
================================================================
依据:
  - Cuturi, NeurIPS 2013, arXiv:1306.0895（Sinkhorn）
  - Zelnik-Manor & Perona, NIPS 2004
    Self-Tuning Spectral Clustering：局部尺度 σ_i=到第 k 近邻的距离。
    本实验把局部尺度用到行温度，不是蛋白图。
  - I019：全局 τ=0.20。I031：融合前各路匹配，关闭。
    本实验仍在 I011 融合之后，只让每条蛋白的核温度随分数峰度变。
自己的改法:
  冻 I011。gap = s_max − s_(k)（k=10）。
  τ_i = clip(τ0 · median(gap) / gap_i, τ_lo, τ_hi)。
  峰锐（gap 大）→ τ 更小；峰平 → τ 更大。
  其余与 I019 相同：行质量=SCut 基数，列质量=π·N，α=0.60。
  τ0∈{0.20}，(τ_lo,τ_hi)∈{(0.10,0.40),(0.15,0.30)}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I034-rowtau.py
输出:
  rel/I034_report.json
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


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))

REPORT_PATH = os.path.join(i011.REL_DIR, "I034_report.json")
K_RRF, BETA_RRF = 30, 0.50
RHO, ALPHA = 1.0, 0.60
TAU0 = 0.20
K_GAP = 10
BANDS = ((0.10, 0.40), (0.15, 0.30))


def row_tau(ens, tau0, lo, hi, k=K_GAP):
    mx = ens.max(axis=1)
    part = np.partition(ens, -k, axis=1)
    kth = part[:, -k]
    gap = np.maximum(mx - kth, 1e-6)
    med = float(np.median(gap))
    tau = tau0 * (med / gap)
    return np.clip(tau, lo, hi).astype(np.float64)


def sinkhorn_rowtau(ens, tau, row_m, col_m, n_iter=40, eps=1e-12):
    shift = ens - ens.max(axis=1, keepdims=True)
    ker = np.exp(shift / tau[:, None]).astype(np.float64)
    return i019.sinkhorn(ker, row_m, col_m, n_iter=n_iter, eps=eps)


def build_i011(train_df, y_full, idx_train, idx_val):
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1 - i011.ALIGN_W) * sim_kmer
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep, sim, K_RRF))
    return s011, hit, y_train, y_val


def apply_i019(ens, y_train, y_val, tau_vec=None, tau_const=TAU0):
    n = ens.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t = i011.search_t(ens, y_val)
    row_m = np.maximum((ens >= t).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    if tau_vec is None:
        ker = np.exp((ens - ens.max(axis=1, keepdims=True)) / tau_const)
        p = i019.sinkhorn(ker, row_m, col_m)
    else:
        p = sinkhorn_rowtau(ens, tau_vec, row_m, col_m)
    return i019.mix(ens, p, ALPHA)


def main():
    t0 = time.time()
    print("=" * 60)
    print("I034: 行自适应 Sinkhorn 温度")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s011, hit, y_train, y_val = build_i011(
        train_df, y_full, idx_train, idx_val)
    miss = ~hit
    s019 = apply_i019(s011, y_train, y_val)
    rec011 = i011.eval_ens(s011, y_val, hit, miss, "i011")
    rec019 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I011 F1={rec011['f1_100']:.4f}")
    print(f"  I019 F1={rec019['f1_100']:.4f}")

    rows = [rec011, rec019]
    configs = {"i011": s011, "i019": s019}
    tau_stats = {}

    print("[1/2] (τ_lo, τ_hi) ...")
    for lo, hi in tqdm(BANDS, desc="    band", unit="b"):
        tau = row_tau(s011, TAU0, lo, hi)
        tau_stats[f"{lo:.2f}-{hi:.2f}"] = {
            "min": round(float(tau.min()), 4),
            "median": round(float(np.median(tau)), 4),
            "max": round(float(tau.max()), 4),
            "std": round(float(tau.std()), 4),
        }
        mixed = apply_i019(s011, y_train, y_val, tau_vec=tau)
        name = f"rt_{lo:.2f}_{hi:.2f}"
        rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"lo": lo, "hi": hi})
        rec["delta_i011"] = round(rec["f1_100"] - rec011["f1_100"], 6)
        rec["delta_i019"] = round(rec["f1_100"] - rec019["f1_100"], 6)
        rows.append(rec)
        configs[name] = mixed
        print(f"  {name:16s} F1={rec['f1_100']:.4f} "
              f"d011={rec['delta_i011']:+.4f} d019={rec['delta_i019']:+.4f} "
              f"τ med={np.median(tau):.3f}")

    print("[2/2] 半折核对...")
    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            t = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= t).astype(np.int8)
        return pred

    oof_rows = []
    oof_i011 = oof_i019 = None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "i011":
            oof_i011 = rec["f1_100"]
        elif name == "i019":
            oof_i019 = rec["f1_100"]
        print(f"  oof_{name:16s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i011"] = round(rec["f1_100"] - oof_i011, 6)
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    rt_oof = [r for r in oof_rows if r["name"].startswith("oof_rt_")]
    best_oof = max(rt_oof, key=lambda r: r["f1_100"]) if rt_oof else oof_rows[0]
    gain011 = best_oof["f1_100"] - oof_i011
    gain019 = best_oof["f1_100"] - oof_i019
    useful = bool(gain019 >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I011 {gain011:+.4f} vs I019 {gain019:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "baseline_i011": rec011,
        "baseline_i019": rec019,
        "best_oof": best_oof,
        "gain_oof_vs_i011": round(gain011, 6),
        "gain_oof_vs_i019": round(gain019, 6),
        "useful": useful,
        "tau_stats": tau_stats,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(i011.REL_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s")


if __name__ == "__main__":
    main()
