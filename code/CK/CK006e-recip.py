"""
CK006e-recip — k-mer 邻居按共享近邻过滤后再传标签，混进 I019
================================================================
依据:
  - Qin et al., CVPR 2011, doi:10.1109/cvpr.2011.5995373
    k-reciprocal nearest neighbors：双方互为近邻才保留。
  - Zhong et al., CVPR 2017, doi:10.1109/cvpr.2017.389
    k-reciprocal encoding 做人再识别重排。
  - CK006d：入度降权 corr 0.995，不换邻居集合，关闭。
    val 不在 train 图里，无法直接互近邻。改成共享近邻：
    保留 t 当且仅当 |N(q) ∩ N(t)| ≥ m。
自己的改法:
  冻 I019。k=6：val→train top-5（CK002）× train→train top-5（ECK003）。
  m∈{1,2}。过滤后 sim² 加权；若全空则回退 top-1。
  ens'=(1-γ)·I019 + γ·col_norm(recip)，γ∈{0.15,0.30}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/CK/CK006e-recip.py
输出:
  rel/CK006e_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "CK006e_report.json")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
M_GRID = (1, 2)
G_GRID = (0.15, 0.30)
TOP = 5


def shared_filter(idx_q, sim_q, idx_tr, m):
    n, k = idx_q.shape
    keep_idx = np.full((n, k), -1, np.int64)
    keep_sim = np.zeros((n, k), np.float32)
    n_keep = np.zeros(n, np.int32)
    fallback = 0
    for i in tqdm(range(n), desc="    过滤", unit="q"):
        nq = set(idx_q[i].tolist())
        slot = 0
        for j in range(k):
            t = int(idx_q[i, j])
            nt = set(idx_tr[t].tolist())
            if len(nq & nt) >= m:
                keep_idx[i, slot] = t
                keep_sim[i, slot] = sim_q[i, j]
                slot += 1
        if slot == 0:
            keep_idx[i, 0] = idx_q[i, 0]
            keep_sim[i, 0] = sim_q[i, 0]
            slot = 1
            fallback += 1
        n_keep[i] = slot
    return keep_idx, keep_sim, n_keep, fallback


def transfer(idx, sim, y_train):
    n, k = idx.shape
    out = np.zeros((n, y_train.shape[1]), np.float32)
    for i in range(n):
        valid = idx[i] >= 0
        if not valid.any():
            continue
        lab = y_train[idx[i, valid]].astype(np.float32)
        ww = np.maximum(sim[i, valid], 1e-7) ** 2
        out[i] = (lab * ww[:, None]).sum(0) / ww.sum()
    return out


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
    print("CK006e: 共享近邻过滤")

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

    ck6 = np.load(os.path.join(i011.REL_DIR, "CK002_neigh_k6.npz"))
    tr6 = np.load(os.path.join(i011.REL_DIR, "ECK003_neigh_train_k6.npz"))
    idx_q, sim_q = ck6["idx_val"][:, :TOP], ck6["sim_val"][:, :TOP]
    idx_tr = tr6["idx"][:, :TOP]
    if len(idx_tr) != len(y_train):
        raise RuntimeError(f"train 邻居 {len(idx_tr)} ≠ y_train {len(y_train)}")

    rows = [rec0]
    configs = {"i019": s019}
    filt_stats = {}
    overlap = {}

    print("[1/2] m × γ ...")
    for m in tqdm(M_GRID, desc="    m", unit="m"):
        k_idx, k_sim, n_keep, n_fb = shared_filter(idx_q, sim_q, idx_tr, m)
        kept = float(n_keep.mean())
        same_top1 = float((k_idx[:, 0] == idx_q[:, 0]).mean())
        filt_stats[f"m{m}"] = {
            "mean_keep": round(kept, 3),
            "fallback": int(n_fb),
            "same_top1": round(same_top1, 4),
        }
        recip = transfer(k_idx, k_sim, y_train)
        rec_r = i011.eval_ens(recip, y_val, hit, miss, f"recip_m{m}")
        rec_r["delta_i019"] = round(rec_r["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec_r)
        corr = float(np.corrcoef(recip.ravel(), sim_kmer.ravel())[0, 1])
        overlap[f"m{m}"] = round(corr, 4)
        print(f"  m={m} keep={kept:.2f} fallback={n_fb} top1同={same_top1:.3f} "
              f"独立 F1={rec_r['f1_100']:.4f} corr={corr:.3f}")
        rec_n = i011.col_norm(recip)
        for g in G_GRID:
            mixed = (1.0 - g) * s019 + g * rec_n
            name = f"mix_m{m}_g{g:.2f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"m": m, "gamma": g})
            rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
            rows.append(rec)
            configs[name] = mixed
            print(f"  {name:16s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f}")

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
        print(f"  oof_{name:16s} F1={rec['f1_100']:.4f}")
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
        "filter": filt_stats,
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
