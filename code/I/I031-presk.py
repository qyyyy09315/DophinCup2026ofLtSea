"""
I031-presk — deep/sim 各自 Sinkhorn 后再按 I011 融合
================================================================
依据:
  - Cuturi, NeurIPS 2013, arXiv:1306.0895（Sinkhorn）
  - Montague & Aslam, CIKM 2001, doi:10.1145/502585.502657
    元搜索应先把各路分数规范化再融合，而不是融合后再规范。
  - I019：对 I011 融合后的分数做双边匹配，有效。
    本实验把匹配提前到融合前：deep、sim 各自对齐行列质量，
    再走 I011 的量纲加 + RRF。
自己的改法:
  冻 deep/sim。对每一路：核 exp((s-rowmax)/τ)，行质量=该路 SCut 基数，
  列质量=π_j·N。s'=(1-α)·s + α·col_norm(P)。
  然后 ens = deep' + 0.75·sim' + 0.50·col_norm(RRF(deep',sim',k=30))。
  τ=0.20，α∈{0.3,0.6}，作用范围 {both, deep, sim}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I031-presk.py
输出:
  rel/I031_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I031_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU, RHO = 0.20, 1.0
A_GRID = (0.3, 0.6)
SCOPES = ("both", "deep", "sim")


def sk_one(score, y_val, pi, alpha, tau=TAU, rho=RHO):
    n = score.shape[0]
    t = i011.search_t(score, y_val)
    row_m = np.maximum((score >= t).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * rho, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((score - score.max(axis=1, keepdims=True)) / tau)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(score, p, alpha)


def fuse(deep, sim):
    return deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep, sim, K_RRF))


def build_parts(train_df, y_full, idx_train, idx_val):
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
    return deep, sim, hit, y_train, y_val


def post_i019(ens, y_train, y_val):
    n = ens.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t = i011.search_t(ens, y_val)
    row_m = np.maximum((ens >= t).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((ens - ens.max(axis=1, keepdims=True)) / TAU)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(ens, p, 0.60)


def main():
    t0 = time.time()
    print("=" * 60)
    print("I031: 路级 Sinkhorn 再 RRF")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    deep, sim, hit, y_train, y_val = build_parts(
        train_df, y_full, idx_train, idx_val)
    miss = ~hit
    pi = y_train.mean(0).astype(np.float64)

    ens011 = fuse(deep, sim)
    ens019 = post_i019(ens011, y_train, y_val)
    rec011 = i011.eval_ens(ens011, y_val, hit, miss, "i011")
    rec019 = i011.eval_ens(ens019, y_val, hit, miss, "i019")
    print(f"  I011 F1={rec011['f1_100']:.4f}")
    print(f"  I019 F1={rec019['f1_100']:.4f}")
    rows = [rec011, rec019]
    configs = {"i011": ens011, "i019": ens019}

    print("[1/2] scope × α ...")
    for scope in tqdm(SCOPES, desc="    scope", unit="s"):
        for a in A_GRID:
            d = deep
            s = sim
            if scope in ("both", "deep"):
                d = sk_one(deep, y_val, pi, a)
            if scope in ("both", "sim"):
                s = sk_one(sim, y_val, pi, a)
            ens = fuse(d, s)
            name = f"pre_{scope}_a{a:.1f}"
            rec = i011.eval_ens(ens, y_val, hit, miss, name,
                                {"scope": scope, "alpha": a})
            rec["delta_i011"] = round(rec["f1_100"] - rec011["f1_100"], 6)
            rec["delta_i019"] = round(rec["f1_100"] - rec019["f1_100"], 6)
            rows.append(rec)
            configs[name] = ens
            print(f"  {name:20s} F1={rec['f1_100']:.4f} "
                  f"d011={rec['delta_i011']:+.4f} d019={rec['delta_i019']:+.4f}")

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
        print(f"  oof_{name:20s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i011"] = round(rec["f1_100"] - oof_i011, 6)
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    pre_oof = [r for r in oof_rows if r["name"].startswith("oof_pre_")]
    best_oof = max(pre_oof, key=lambda r: r["f1_100"]) if pre_oof else oof_rows[0]
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
