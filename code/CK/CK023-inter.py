"""
CK023-inter — 弱同源行 k-mer 只用与 DIAMOND 的交集邻居，接到 C011a
================================================================
依据:
  - CK022：弱行并入 k-mer 独有邻居，OOF −0.0009，remote 掉。
    本实验走反操作：弱行 k-mer 只保留 DIAMOND top-3 ∩ k6 top-5。
  - CK005f 全局邻居 RRF 已关；CK006e 共享近邻过滤已关。本实验按 pident
    切片改 k-mer 集合，不改 ALIGN_W / SIM_W。
  - A006：独有 vs 重叠邻居标签 Jaccard（同回合诊断）。
自己的改法:
  冻 deep、ALIGN_W、SIM_W。弱行（pident<T）用交集邻居 sim² 传标签，
  sim' = (1-γ)·sim + γ·(0.4·DIAMOND + 0.6·inter)。
  T∈{50,80}，γ∈{0.30,1.00}。强同源不动。门槛半折 OOF +0.001。
用法:
  python code/CK/CK023-inter.py
输出:
  rel/CK023_report.json
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
ck007 = load_mod("ck007", os.path.join("CK", "CK007-pident.py"))

REPORT_PATH = os.path.join(i011.REL_DIR, "CK023_report.json")
C011A_ALPHA = 0.50
CNN_W = 0.2
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
T_GRID = (50.0, 80.0)
G_GRID = (0.30, 1.00)


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


def inter_kmer(idx_d, idx_k, sim_k, y_train, top=5):
    n, nlab = idx_d.shape[0], y_train.shape[1]
    extra = np.zeros((n, nlab), np.float32)
    n_kept = np.zeros(n, np.int32)
    for i in range(n):
        dset = set(int(j) for j in idx_d[i, :3] if j >= 0)
        acc = np.zeros(nlab, np.float64)
        wsum = 0.0
        c = 0
        for j in range(top):
            t = int(idx_k[i, j])
            if t < 0 or t not in dset:
                continue
            w = float(max(sim_k[i, j], 1e-7)) ** 2
            acc += w * y_train[t]
            wsum += w
            c += 1
        if wsum > 0:
            extra[i] = (acc / wsum).astype(np.float32)
        n_kept[i] = c
    return extra, n_kept


def pack_slices(rec, pred, y, twilight, remote, strong):
    rec["f1_twilight"] = round(i011.macro_f1(pred[twilight], y[twilight]), 6) if twilight.any() else 0.0
    rec["f1_remote"] = round(i011.macro_f1(pred[remote], y[remote]), 6) if remote.any() else 0.0
    rec["f1_strong"] = round(i011.macro_f1(pred[strong], y[strong]), 6) if strong.any() else 0.0
    return rec


def main():
    t0 = time.time()
    print("=" * 60)
    print("CK023: weak-homology k-mer ∩ DIAMOND")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_d, bits_d, pid_v = ck007.parse_aln_pid(tsv, val_ids, train_map)
    hit = idx_d[:, 0] >= 0
    miss = ~hit
    pident = pid_v[:, 0].copy()
    pident[miss] = 0.0
    twilight = hit & (pident < 80) & (pident >= 50)
    remote = hit & (pident < 50)
    strong = hit & (pident >= 80)

    d6 = np.load(os.path.join(i011.REL_DIR, "CK002_neigh_k6.npz"))
    inter, n_kept = inter_kmer(idx_d, d6["idx_val"], d6["sim_val"], y_train)
    print(f"  inter mean kept: all={n_kept.mean():.2f} tw={n_kept[twilight].mean():.2f} "
          f"remote={n_kept[remote].mean():.2f} strong={n_kept[strong].mean():.2f}")

    sim_bit = i011.agg_labels(idx_d, bits_d, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim_lock = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    sim_inter = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * inter
    empty = n_kept == 0
    sim_inter[empty] = sim_lock[empty]
    cnn42 = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(i011.REL_DIR, "C005_val_probs.npz"))["val"]
    cnn_lock = (1.0 - C011A_ALPHA) * cnn42 + C011A_ALPHA * c005_val
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + CNN_W * cnn_lock

    s_lock = fuse(deep, sim_lock, y_train, y_val)
    pred0 = (s_lock >= i011.search_t(s_lock, y_val)).astype(np.int8)
    rec0 = pack_slices(i011.eval_ens(s_lock, y_val, hit, miss, "c011a"),
                       pred0, y_val, twilight, remote, strong)
    print(f"  C011a F1={rec0['f1_100']:.4f} tw={rec0['f1_twilight']:.4f} "
          f"remote={rec0['f1_remote']:.4f}")

    configs = {"c011a": s_lock}
    rows = [rec0]
    for tcut in T_GRID:
        weak = pident < tcut
        for g in G_GRID:
            sim = sim_lock.copy()
            sim[weak] = (1.0 - g) * sim_lock[weak] + g * sim_inter[weak]
            s = fuse(deep, sim, y_train, y_val)
            pred = (s >= i011.search_t(s, y_val)).astype(np.int8)
            name = f"t{tcut:.0f}_g{g:.2f}"
            rec = pack_slices(i011.eval_ens(s, y_val, hit, miss, name),
                              pred, y_val, twilight, remote, strong)
            rec.update({"tcut": tcut, "gamma": g, "n_weak": int(weak.sum()),
                        "delta_c011a": round(rec["f1_100"] - rec0["f1_100"], 6)})
            rows.append(rec)
            configs[name] = s
            print(f"  {name:12s} F1={rec['f1_100']:.4f} d={rec['delta_c011a']:+.4f} "
                  f"tw={rec['f1_twilight']:.4f} remote={rec['f1_remote']:.4f}")

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
            "f1_twilight": round(i011.macro_f1(pred[twilight], y_val[twilight]), 6),
            "f1_remote": round(i011.macro_f1(pred[remote], y_val[remote]), 6),
        }
        oof_rows.append(rec)
        if name == "c011a":
            oof_lock = rec["f1_100"]
        print(f"  {rec['name']:20s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_c011a"] = round(rec["f1_100"] - oof_lock, 6)
    cand = [r for r in oof_rows if r["name"] != "oof_c011a"]
    best = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best["f1_100"] - oof_lock
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best['name']} vs C011a {gain:+.4f} → "
          f"{'有效' if useful else '无增益'}")
    report = {
        "inter_mean_kept": {
            "all": round(float(n_kept.mean()), 3),
            "twilight": round(float(n_kept[twilight].mean()), 3),
            "remote": round(float(n_kept[remote].mean()), 3),
            "strong": round(float(n_kept[strong].mean()), 3),
        },
        "baseline_c011a": rec0,
        "oof_c011a": oof_lock,
        "best_oof": best,
        "gain_oof_vs_c011a": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
        "conclusion": (
            "弱同源交集 k-mer OOF≥+0.001"
            if useful else "弱同源交集 k-mer 无增益，关闭；不重开 CK022/CK006e"
        ),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}  用时 {report['seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
