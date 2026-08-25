"""
CK006d-hub — k-mer 邻居按入度 hubness 降权，混进 I019
================================================================
依据:
  - Radovanović, Nanopoulos & Ivanović, ICML 2009,
    doi:10.1145/1553374.1553485；JMLR 2010「Hubs in Space」。
    高维 kNN 里少数点成为许多查询的近邻（hub），会污染标签传递。
  - CK006c：按邻居携带的稀有标签 IC 加权，corr 0.99，关闭。
    本实验降权的是「被多少 query 选中」，不是标签稀有度。
  - CK006b：TF-IDF 改的是 k-mer 种子权。这里改的是邻居样本权。
自己的改法:
  冻 I019。对每个 k 的 val→train top-5 名单，N_k[j]=该 train 点被选次数。
  w = sim² / (N_k[j]^α)，再加权传标签。α∈{0.5,1.0}。
  ens'=(1-γ)·I019 + γ·col_norm(hub_sim)，γ∈{0.15,0.30}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/CK/CK006d-hub.py
输出:
  rel/CK006d_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "CK006d_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
A_GRID = (0.5, 1.0)
G_GRID = (0.15, 0.30)
TOP = 5
K_W = ((3, 0.0), (4, 0.1), (5, 0.2), (6, 0.4), (7, 0.2), (8, 0.1))


def hub_kmer(y_train, n_train, alpha):
    sim = 0.0
    stats = {}
    for k, wt in K_W:
        if wt <= 0:
            continue
        prefix = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(i011.REL_DIR, f"{prefix}_neigh_k{k}.npz"))
        idx = d["idx_val"][:, :TOP]
        s = d["sim_val"][:, :TOP].astype(np.float32)
        n = idx.shape[0]
        nk = np.bincount(idx.ravel(), minlength=n_train).astype(np.float32)
        stats[k] = {
            "max": int(nk.max()),
            "mean": round(float(nk[nk > 0].mean()) if (nk > 0).any() else 0.0, 2),
            "hubs": int((nk >= 10).sum()),
        }
        lab = y_train[idx.ravel()].reshape(n, TOP, -1).astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        ww = ww / np.maximum(nk[idx] ** alpha, 1.0)
        sim = sim + wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return sim.astype(np.float32), stats


def build_i019(train_df, y_full, idx_train, idx_val):
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
    return s019, hit, y_train, y_val, sim_kmer


def main():
    t0 = time.time()
    print("=" * 60)
    print("CK006d: hubness 邻居降权")

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=i011.RANDOM_STATE)
    s019, hit, y_train, y_val, sim_kmer = build_i019(
        train_df, y_full, idx_train, idx_val)
    miss = ~hit
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    rows = [rec0]
    configs = {"i019": s019}
    hub_stats = {}
    overlap = {}

    print("[1/2] α × γ ...")
    for a in tqdm(A_GRID, desc="    α", unit="a"):
        hub_sim, stats = hub_kmer(y_train, len(y_train), a)
        hub_stats[f"a{a:.1f}"] = stats
        rec_r = i011.eval_ens(hub_sim, y_val, hit, miss, f"hub_a{a:.1f}")
        rec_r["delta_i019"] = round(rec_r["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec_r)
        corr = float(np.corrcoef(hub_sim.ravel(), sim_kmer.ravel())[0, 1])
        overlap[f"a{a:.1f}"] = round(corr, 4)
        print(f"  hub α={a:.1f} 独立 F1={rec_r['f1_100']:.4f} corr_kmer={corr:.3f} "
              f"k6 hub≥10={stats.get(6, {}).get('hubs', '?')}")
        hub_n = i011.col_norm(hub_sim)
        for g in G_GRID:
            mixed = (1.0 - g) * s019 + g * hub_n
            name = f"mix_a{a:.1f}_g{g:.2f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, name,
                                {"alpha": a, "gamma": g})
            rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[name] = mixed
            print(f"  {name:18s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

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
        print(f"  oof_{name:18s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    mix_oof = [r for r in oof_rows if r["name"].startswith("oof_mix_")]
    best_oof = max(mix_oof, key=lambda r: r["f1_100"]) if mix_oof else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} "
          f"vs I019 {gain:+.4f} → {'有效' if useful else '无增益'}")

    report = {
        "baseline_i019": rec0,
        "best_oof": best_oof,
        "gain_oof_vs_i019": round(gain, 6),
        "useful": useful,
        "hub_stats": hub_stats,
        "corr_kmer": overlap,
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
