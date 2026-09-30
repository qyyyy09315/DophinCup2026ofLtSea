"""
CK008-annscore — BLAST2GO 式 annotation score 聚合 DIAMOND 邻居
================================================================
依据:
  - Conesa et al., Bioinformatics 2005, doi:10.1093/bioinformatics/bti610
    Blast2GO：候选 GO 的 annotation score =
    max(hit similarity) × GOw + (#hits 带该 GO)。
  - Götz et al., NAR 2008, doi:10.1093/nar/gkn176
    高通量 Blast2GO 仍用该打分，不把相似度平均掉。
  - CK005 / I011：bitscore 加权平均，行归一化后同源强度被抹掉。
    CK007：按 pident 砍邻居，负。本实验不砍邻居，改聚合。
自己的改法:
  冻 deep / k-mer / 邻居名单。比对路三种聚合：
    sum：Σ bits·y（保留质量）
    max：max bits·y（Naive BLAST / GOLabeler 同源）
    as ：max + λ·count（Blast2GO）
  列 min-max 后再与 k-mer 按 ALIGN_W 混合，重建 I011/I019。
  λ∈{20,50}。另混 (1-γ)·I019 + γ·新 I019，γ∈{0.15,0.30}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/CK/CK008-annscore.py
输出:
  rel/CK008_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "CK008_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
TOP_K = 3
LAM_GRID = (20.0, 50.0)
G_GRID = (0.15, 0.30)


def neigh_pack(idx, bits, y_train, top_k=TOP_K):
    labs, wts, valid = [], [], []
    for j in range(top_k):
        v = idx[:, j] >= 0
        lab = y_train[idx[:, j].clip(min=0)].astype(np.float32)
        lab[~v] = 0.0
        labs.append(lab)
        wts.append(np.maximum(bits[:, j:j + 1], 0.0).astype(np.float32))
        valid.append(v)
    return labs, wts, valid


def agg_sum(labs, wts):
    out = 0.0
    for lab, w in zip(labs, wts):
        out = out + lab * w
    return np.asarray(out, np.float32)


def agg_max(labs, wts):
    out = np.zeros_like(labs[0], np.float32)
    for lab, w in zip(labs, wts):
        np.maximum(out, lab * w, out=out)
    return out


def agg_count(labs):
    out = 0.0
    for lab in labs:
        out = out + lab
    return np.asarray(out, np.float32)


def build_fusion(deep, sim_bit, sim_kmer, y_train, y_val):
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep, sim, K_RRF))
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t_i011 = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t_i011).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(s011, p, ALPHA_SK)


def main():
    t0 = time.time()
    print("=" * 60)
    print("CK008: BLAST2GO 式 annotation score")

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
    print(f"  hit {int(hit.sum())}/{len(hit)}")

    sim_bit_avg = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]

    s019 = build_fusion(deep, sim_bit_avg, sim_kmer, y_train, y_val)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    labs, wts, _ = neigh_pack(idx_v, bits_v, y_train)
    raw = {
        "sum": agg_sum(labs, wts),
        "max": agg_max(labs, wts),
    }
    cnt = agg_count(labs)
    for lam in LAM_GRID:
        raw[f"as_l{lam:.0f}"] = raw["max"] + lam * cnt

    rows = [rec0]
    configs = {"i019": s019}
    corr = {}

    print("[1/2] 聚合 × γ ...")
    rebuilt = {}
    for name, score in tqdm(raw.items(), desc="    agg", unit="m"):
        sim_bit = i011.col_norm(score)
        c = float(np.corrcoef(sim_bit.ravel(), sim_bit_avg.ravel())[0, 1])
        corr[name] = round(c, 4)
        s_f = build_fusion(deep, sim_bit, sim_kmer, y_train, y_val)
        rebuilt[name] = s_f
        rec_f = i011.eval_ens(s_f, y_val, hit, miss, f"filt_{name}")
        rec_f["delta_i019"] = round(rec_f["f1_100"] - rec0["f1_100"], 6)
        rec_f["corr_avg"] = corr[name]
        rows.append(rec_f)
        configs[f"filt_{name}"] = s_f
        print(f"  filt_{name:10s} F1={rec_f['f1_100']:.4f} d={rec_f['delta_i019']:+.4f} "
              f"corr={c:.3f}")
        s_n = i011.col_norm(s_f)
        for g in G_GRID:
            mixed = (1.0 - g) * s019 + g * s_n
            mix_name = f"mix_{name}_g{g:.2f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, mix_name, {"gamma": g})
            rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[mix_name] = mixed
            print(f"  {mix_name:22s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

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
        print(f"  oof_{name:22s} F1={rec['f1_100']:.4f}")
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
        "corr_avg": corr,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(i011.REL_DIR, exist_ok=True)
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  用时 {report['seconds']}s")


if __name__ == "__main__":
    main()
