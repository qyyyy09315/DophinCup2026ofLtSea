"""
CK007-pident — 按序列一致度门槛过滤 DIAMOND 邻居，再重建 I011/I019
================================================================
依据:
  - Tian & Skolnick, JMB 2003, doi:10.1016/j.jmb.2003.08.057
    酶功能随 pairwise identity 掉：~40% 已不稳，≥70% 才较可靠。
  - Rost, Protein Engineering 1999, doi:10.1093/protein/12.2.85
    twilight zone ~20–35% identity，结构/功能传递不可靠。
  - CK005：pident 只当过加权（pident²），融合端被 bitscore 吸收。
    本实验改邻居集合，不是改权重。
  - I011.parse_aln 丢掉第 5 列；TSV 里一直有 pident。
自己的改法:
  冻 deep / k-mer。val→idx_train 的 DIAMOND 邻居按 pident≥T 过滤，
  空行回退 top-1 bitscore。重建 sim 与 I011/I019。
  T∈{40,50,60,70}。另混 (1-γ)·I019 + γ·过滤后 I019，γ∈{0.15,0.30}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/CK/CK007-pident.py
输出:
  rel/CK007_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "CK007_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
T_GRID = (40.0, 50.0, 60.0, 70.0)
G_GRID = (0.15, 0.30)
MAX_ACCEPT = 5
EVALUE = 0.001


def parse_aln_pid(tsv, query_ids, id_map, evalue=EVALUE):
    idx = np.full((len(query_ids), MAX_ACCEPT), -1, np.int64)
    bits = np.zeros((len(query_ids), MAX_ACCEPT), np.float32)
    pident = np.zeros((len(query_ids), MAX_ACCEPT), np.float32)
    qpos = {q: i for i, q in enumerate(query_ids)}
    with open(tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            q, t = p[0], p[1]
            if q not in qpos or t not in id_map:
                continue
            i = qpos[q]
            try:
                bs, ev, pid = float(p[2]), float(p[3]), float(p[4])
            except ValueError:
                continue
            if ev > evalue:
                continue
            j = 0
            while j < MAX_ACCEPT and bits[i, j] > bs:
                j += 1
            if j < MAX_ACCEPT:
                idx[i, j:] = np.roll(idx[i, j:], 1)
                bits[i, j:] = np.roll(bits[i, j:], 1)
                pident[i, j:] = np.roll(pident[i, j:], 1)
                idx[i, j] = id_map[t]
                bits[i, j] = bs
                pident[i, j] = pid
    return idx, bits, pident


def filter_pid(idx, bits, pid, thresh):
    n, k = idx.shape
    out_i = np.full_like(idx, -1)
    out_b = np.zeros_like(bits)
    n_keep = np.zeros(n, np.int32)
    fallback = 0
    for i in range(n):
        slot = 0
        for j in range(k):
            if idx[i, j] < 0:
                continue
            if pid[i, j] >= thresh:
                out_i[i, slot] = idx[i, j]
                out_b[i, slot] = bits[i, j]
                slot += 1
        if slot == 0 and idx[i, 0] >= 0:
            out_i[i, 0] = idx[i, 0]
            out_b[i, 0] = bits[i, 0]
            slot = 1
            fallback += 1
        n_keep[i] = slot
    return out_i, out_b, n_keep, fallback


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
    return i019.mix(s011, p, ALPHA_SK), sim


def main():
    t0 = time.time()
    print("=" * 60)
    print("CK007: pident 门槛过滤 DIAMOND 邻居")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v, pid_v = parse_aln_pid(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    valid = idx_v >= 0
    print(f"  hit {int(hit.sum())}/{len(hit)}  pident "
          f"med={float(np.median(pid_v[valid])):.1f} "
          f"p25={float(np.percentile(pid_v[valid], 25)):.1f} "
          f"p75={float(np.percentile(pid_v[valid], 75)):.1f}")

    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]

    s019, _ = build_fusion(deep, sim_bit, sim_kmer, y_train, y_val)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    rows = [rec0]
    configs = {"i019": s019}
    keep_stats = {}

    print("[1/2] T × γ ...")
    for t in tqdm(T_GRID, desc="    T", unit="t"):
        fi, fb, n_keep, fb_n = filter_pid(idx_v, bits_v, pid_v, t)
        kept = int((n_keep[hit] > 0).sum())
        mean_k = float(n_keep[hit].mean()) if hit.any() else 0.0
        keep_stats[f"T{t:.0f}"] = {
            "mean_keep_hit": round(mean_k, 3),
            "fallback": int(fb_n),
            "still_hit": kept,
        }
        print(f"  T={t:.0f}  hit均邻居 {mean_k:.2f}  回退 {fb_n}  仍hit {kept}")
        sim_bit_f = i011.agg_labels(fi, fb, y_train)
        s_f, _ = build_fusion(deep, sim_bit_f, sim_kmer, y_train, y_val)
        rec_f = i011.eval_ens(s_f, y_val, hit, miss, f"filt_T{t:.0f}", {"T": t})
        rec_f["delta_i019"] = round(rec_f["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec_f)
        configs[f"filt_T{t:.0f}"] = s_f
        print(f"  filt_T{t:.0f} F1={rec_f['f1_100']:.4f} d={rec_f['delta_i019']:+.4f}")
        s_n = i011.col_norm(s_f)
        for g in G_GRID:
            mixed = (1.0 - g) * s019 + g * s_n
            name = f"mix_T{t:.0f}_g{g:.2f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"T": t, "gamma": g})
            rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[name] = mixed
            print(f"  {name:20s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

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
        "keep_stats": keep_stats,
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
