"""
I032-probfuse — 按名次段估 P(相关) 再融合 deep/sim
================================================================
依据:
  - ProbFuse: Lillis, Toolan, Collier & Dunnion, SIGIR 2006,
    doi:10.1145/1148170.1148197。把每路排序切成段，用训练估
    P(相关|段,系统)，融合时把各路该段概率相加。
  - I011：RRF 1/(k+rank) 不估段相关，有效。
  - I011d：按 top-M Jaccard 调 β，最优仍恒定 0.50，关闭。
    本实验估的是「名次段」的精度，不是查询级 β。
自己的改法:
  冻 deep/sim。段长 B∈{10,25,50}。半折：拟合折上
  P_r(b)=该路 rank∈[bB,(b+1)B) 的标签正例率；评估折
  s_pf = P_deep(seg)+P_sim(seg)。
  混合 ens'=(1-γ)·base + γ·col_norm(s_pf)，base∈{I011,I019}，
  γ∈{0.3,0.5}。门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/I/I032-probfuse.py
输出:
  rel/I032_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "I032_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
B_GRID = (10, 25, 50)
G_GRID = (0.3, 0.5)
N_LABELS = i011.N_LABELS


def n_bins(bin_size):
    return int(np.ceil(N_LABELS / bin_size))


def segment_p(ranks, y, bin_size):
    nb = n_bins(bin_size)
    num = np.zeros(nb, np.float64)
    den = np.zeros(nb, np.float64)
    seg = np.minimum(ranks // bin_size, nb - 1)
    for b in range(nb):
        m = seg == b
        den[b] = m.sum()
        if den[b] > 0:
            num[b] = y[m].sum()
    p = np.zeros(nb, np.float64)
    nz = den > 0
    p[nz] = num[nz] / den[nz]
    if (~nz).any() and nz.any():
        p[~nz] = p[nz].mean()
    return p.astype(np.float32)


def apply_pf(rd, rs, p_d, p_s, bin_size):
    nb = len(p_d)
    sd = np.minimum(rd // bin_size, nb - 1)
    ss = np.minimum(rs // bin_size, nb - 1)
    return (p_d[sd] + p_s[ss]).astype(np.float32)


def oof_pf(deep, sim, y, folds, bin_size):
    out = np.zeros_like(deep, np.float32)
    rd_all, rs_all = i011.ranks(deep), i011.ranks(sim)
    for i, hold in enumerate(folds):
        fit = folds[1 - i]
        p_d = segment_p(rd_all[fit], y[fit], bin_size)
        p_s = segment_p(rs_all[fit], y[fit], bin_size)
        out[hold] = apply_pf(rd_all[hold], rs_all[hold], p_d, p_s, bin_size)
    return out


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
    s019 = i019.mix(s011, p, ALPHA_SK)
    return deep, sim, s011, s019, hit, y_val


def main():
    t0 = time.time()
    print("=" * 60)
    print("I032: ProbFuse 段相关融合")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    deep, sim, s011, s019, hit, y_val = build_parts(
        train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec011 = i011.eval_ens(s011, y_val, hit, miss, "i011")
    rec019 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I011 F1={rec011['f1_100']:.4f}")
    print(f"  I019 F1={rec019['f1_100']:.4f}")

    rng = np.random.RandomState(i011.RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    rows = [rec011, rec019]
    configs = {"i011": s011, "i019": s019}

    print("[1/2] B × γ（半折估段概率）...")
    for bsz in tqdm(B_GRID, desc="    B", unit="b"):
        pf = oof_pf(deep, sim, y_val, folds, bsz)
        rec_raw = i011.eval_ens(pf, y_val, hit, miss, f"pf_B{bsz}_raw", scale=True)
        rec_raw["delta_i019"] = round(rec_raw["f1_100"] - rec019["f1_100"], 6)
        rows.append(rec_raw)
        configs[f"pf_B{bsz}_raw"] = pf
        print(f"  pf_B{bsz}_raw           F1={rec_raw['f1_100']:.4f}")
        pf_n = i011.col_norm(pf)
        for g in G_GRID:
            for base_name, base in (("i011", s011), ("i019", s019)):
                mixed = (1.0 - g) * base + g * pf_n
                name = f"pf_B{bsz}_{base_name}_g{g:.1f}"
                rec = i011.eval_ens(mixed, y_val, hit, miss, name,
                                    {"B": bsz, "gamma": g, "base": base_name})
                rec["delta_i011"] = round(rec["f1_100"] - rec011["f1_100"], 6)
                rec["delta_i019"] = round(rec["f1_100"] - rec019["f1_100"], 6)
                rows.append(rec)
                configs[name] = mixed
                print(f"  {name:28s} F1={rec['f1_100']:.4f} "
                      f"d011={rec['delta_i011']:+.4f} d019={rec['delta_i019']:+.4f}")

    print("[2/2] 半折阈值...")

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
        print(f"  oof_{name:28s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i011"] = round(rec["f1_100"] - oof_i011, 6)
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    pf_oof = [r for r in oof_rows if "pf_" in r["name"]]
    best_oof = max(pf_oof, key=lambda r: r["f1_100"]) if pf_oof else oof_rows[0]
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
