"""
I056-pathtemp — 融合前对 deep / sim 分路温度，接到 C011a
================================================================
依据:
  - Temperature scaling: Guo C, Pleiss G, Sun Y, Weinberger KQ.
    On Calibration of Modern Neural Networks. ICML 2017.
    arXiv:1706.04599。过自信网络用单参数 T 校准。
  - 本赛独立 sigmoid 多标签：p' = σ(logit(p) / T)，不是 softmax。
  - I019 之后再温度 + 逐标签 SCut 是列单调恒等，不开。
    本实验把温度放在融合前，让 RRF / Sinkhorn 看到不同质量。
  - I031：分路 Sinkhorn 再 I011，已关。本实验不改行列质量，只改幅度。
  - I004 / I038：残差头、行内幂，已关。
自己的改法:
  冻 ALIGN_W、SIM_W、邻居、CNN 权。
  deep' = σ(logit(deep)/T_d)，sim' = σ(logit(sim)/T_s)。
  再走 I011 + I019（k=30 β=0.50 τ=0.20 α=0.60）。
  先单路扫 T∈{0.5,0.75,1.25,1.5}（另一路 T=1），
  再交叉两路各自全 val 最优。半折 OOF 评。门槛 +0.001。
用法:
  python code/I/I056-pathtemp.py
输出:
  rel/I056_report.json
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


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))
ck007 = load_mod("ck007", os.path.join("CK", "CK007-pident.py"))

REPORT_PATH = os.path.join(i011.REL_DIR, "I056_report.json")
C011A_ALPHA = 0.50
CNN_W = 0.2
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
T_GRID = (0.50, 0.75, 1.25, 1.50)
EPS = 1e-6


def temp_scale(p, t):
    if abs(t - 1.0) < 1e-12:
        return p
    x = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    z = np.log(x) - np.log(1.0 - x)
    out = 1.0 / (1.0 + np.exp(-z / t))
    return out.astype(np.float32)


def fuse(deep, sim, y_train, y_val):
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep, sim, K_RRF))
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(s011, p, ALPHA_SK)


def pack_slices(rec, pred, y, twilight, remote, strong):
    rec["f1_twilight"] = round(i011.macro_f1(pred[twilight], y[twilight]), 6) if twilight.any() else 0.0
    rec["f1_remote"] = round(i011.macro_f1(pred[remote], y[remote]), 6) if remote.any() else 0.0
    rec["f1_strong"] = round(i011.macro_f1(pred[strong], y[strong]), 6) if strong.any() else 0.0
    return rec


def main():
    t0 = time.time()
    print("=" * 60)
    print("I056: path-wise temperature before I011+I019")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v, pid_v = ck007.parse_aln_pid(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    pident = pid_v[:, 0].copy()
    pident[miss] = 0.0
    twilight = hit & (pident < 80) & (pident >= 50)
    remote = hit & (pident < 50)
    strong = hit & (pident >= 80)
    print(f"  miss={int(miss.sum())} tw={int(twilight.sum())} remote={int(remote.sum())}")

    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim_lock = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    cnn42 = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["val"]
    cnn_lock = (1.0 - C011A_ALPHA) * cnn42 + C011A_ALPHA * c005_val
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep_lock = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + CNN_W * cnn_lock

    s_lock = fuse(deep_lock, sim_lock, y_train, y_val)
    pred0 = (s_lock >= i011.search_t(s_lock, y_val)).astype(np.int8)
    rec0 = pack_slices(i011.eval_ens(s_lock, y_val, hit, miss, "c011a"),
                       pred0, y_val, twilight, remote, strong)
    rec0.update({"td": 1.0, "ts": 1.0, "delta_c011a": 0.0})
    print(f"  C011a F1={rec0['f1_100']:.4f} miss={rec0['f1_miss']:.4f}")

    configs = {"c011a": s_lock}
    rows = [rec0]

    def add_cfg(name, td, ts, deep, sim):
        s = fuse(deep, sim, y_train, y_val)
        pred = (s >= i011.search_t(s, y_val)).astype(np.int8)
        rec = pack_slices(i011.eval_ens(s, y_val, hit, miss, name),
                          pred, y_val, twilight, remote, strong)
        rec.update({
            "td": td, "ts": ts,
            "delta_c011a": round(rec["f1_100"] - rec0["f1_100"], 6),
        })
        rows.append(rec)
        configs[name] = s
        print(f"  {name:16s} F1={rec['f1_100']:.4f} d={rec['delta_c011a']:+.4f} "
              f"miss={rec['f1_miss']:.4f}")
        return rec

    for td in T_GRID:
        add_cfg(f"td{td:.2f}", td, 1.0, temp_scale(deep_lock, td), sim_lock)
    for ts in T_GRID:
        add_cfg(f"ts{ts:.2f}", 1.0, ts, deep_lock, temp_scale(sim_lock, ts))

    deep_cands = [r for r in rows if r["name"].startswith("td")]
    sim_cands = [r for r in rows if r["name"].startswith("ts")]
    best_td = max(deep_cands, key=lambda r: r["f1_100"])["td"] if deep_cands else 1.0
    best_ts = max(sim_cands, key=lambda r: r["f1_100"])["ts"] if sim_cands else 1.0
    if abs(best_td - 1.0) > 1e-12 or abs(best_ts - 1.0) > 1e-12:
        add_cfg(f"td{best_td:.2f}_ts{best_ts:.2f}", best_td, best_ts,
                temp_scale(deep_lock, best_td), temp_scale(sim_lock, best_ts))
    else:
        print("  单路最优仍 T=1，跳过交叉")

    rng = np.random.RandomState(i011.RANDOM_STATE)
    perm = rng.permutation(len(y_val))
    mid = len(y_val) // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            th = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= th).astype(np.int8)
        return pred

    oof_rows, oof_lock = [], None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "c011a":
            oof_lock = rec["f1_100"]
        print(f"  {rec['name']:24s} F1={rec['f1_100']:.4f} miss={rec['f1_miss']:.4f}")
    for rec in oof_rows:
        rec["delta_c011a"] = round(rec["f1_100"] - oof_lock, 6)
    cand = [r for r in oof_rows if r["name"] != "oof_c011a"]
    best = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best["f1_100"] - oof_lock
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best['name']} vs C011a {gain:+.4f} → "
          f"{'有效' if useful else '无增益'}")
    report = {
        "baseline_c011a": rec0,
        "oof_c011a": oof_lock,
        "best_oof": best,
        "gain_oof_vs_c011a": round(gain, 6),
        "useful": useful,
        "best_td_fullval": best_td,
        "best_ts_fullval": best_ts,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "分路温度 OOF≥+0.001"
            if useful else "分路温度无增益，关闭；不重开 I004/I031/I038"
        ),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}  用时 {report['seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
