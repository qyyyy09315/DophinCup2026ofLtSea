"""
I040-zsum — 行内 z-score / min-max 规范 deep 与 sim 再相加
================================================================
依据:
  - Montague & Aslam, CIKM 2001, doi:10.1145/502653.502657
    元搜索相加前先 z-score 或 min-max，消量纲。
  - I011 线性项是 raw deep + 0.75·sim，RRF 项才列归一。
    I017 是融合后系统积/和；I038 是 I019 行内幂；I031 是路级
    Sinkhorn。本实验只规范两路一元分数再 CombSUM。
自己的改法:
  冻邻居。每行对 deep、sim 做 z-score 或 min-max，等权相加，
  再叠 β=0.50 的 I011 RRF，重建 I019。另混 (1-γ)·I019 + γ·新分数，
  γ∈{0.15,0.30,1.0}。门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I040-zsum.py
输出:
  rel/I040_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I040_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
G_GRID = (0.15, 0.30, 1.0)
NORMS = ("zrow", "mmrow")


def zrow(x):
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-8)


def mmrow(x):
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, 1e-8)


def sinkhorn_on(s, y_train, y_val):
    n = s.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t = i011.search_t(s, y_val)
    row_m = np.maximum((s >= t).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s - s.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(s, p, ALPHA_SK)


def main():
    t0 = time.time()
    print("=" * 60)
    print("I040: 行内 z-score CombSUM")

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
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    rrf = i011.col_norm(i011.rrf_pair(deep, sim, K_RRF))
    s011 = deep + i011.SIM_W * sim + BETA_RRF * rrf
    s019 = sinkhorn_on(s011, y_train, y_val)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    rows = [rec0]
    configs = {"i019": s019}
    fn = {"zrow": zrow, "mmrow": mmrow}

    print("[1/2] 规范 × γ ...")
    for name in tqdm(NORMS, desc="    norm", unit="n"):
        nd, ns = fn[name](deep), fn[name](sim)
        s_lin = nd + ns
        s_new = sinkhorn_on(s_lin + BETA_RRF * rrf, y_train, y_val)
        rec_n = i011.eval_ens(s_new, y_val, hit, miss, name)
        rec_n["delta_i019"] = round(rec_n["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec_n)
        configs[name] = s_new
        print(f"  {name:8s} F1={rec_n['f1_100']:.4f} d={rec_n['delta_i019']:+.4f}")
        s_n = i011.col_norm(s_new)
        for g in G_GRID:
            if g >= 0.999:
                continue
            mixed = (1.0 - g) * s019 + g * s_n
            mix_name = f"mix_{name}_g{g:.2f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, mix_name,
                                {"norm": name, "gamma": g})
            rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[mix_name] = mixed
            print(f"  {mix_name:20s} F1={rec['f1_100']:.4f} "
                  f"d={rec['delta_i019']:+.4f}")

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
            th = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= th).astype(np.int8)
        return pred

    oof_rows = []
    oof_i019 = None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "i019":
            oof_i019 = rec["f1_100"]
        print(f"  oof_{name:20s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    cand = [r for r in oof_rows if r["name"] != "oof_i019"]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline_i019": rec0,
        "best_oof": best_oof,
        "gain_oof_vs_i019": round(gain, 6),
        "useful": useful,
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
