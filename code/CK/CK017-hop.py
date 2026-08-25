"""
CK017-hop — 用 train 图把 DIAMOND 邻居扩一跳（改检索集合）
================================================================
依据:
  - Altschul et al., NAR 1997, doi:10.1093/nar/25.17.3389
    PSI-BLAST：用首轮命中重建 profile 再搜，扩同源集合。
  - Rocchio / 伪相关反馈：把首轮命中当相关，扩查询。
    Lv & Zhai, SIGIR 2010, doi:10.1145/1835449.1835546。
  - ECK003：二跳扩的是 k-mer 分数，corr 0.80，负。
  - CK015：在 train 图上扩散标签再取出，负。
    本实验不改分数形态、不扩散标签，只把 DIAMOND 邻居名单
    用 train–train k6 扩一跳，再按 bitscore 加权平均。
自己的改法:
  冻 deep / k-mer。val DIAMOND top-3。对每个命中 n，把
  ECK003 train 图上 n 的邻居 n' 以 w=bits(n)·λ·sim(n,n') 并入。
  按权取 top-5 再 agg。λ∈{0.3,0.6}。重建 I011/I019。
  另混 (1-γ)·I019 + γ·新 I019，γ∈{0.15,0.30}。
  门槛相对 I019 半折 ≥ +0.001。
用法:
  python code/CK/CK017-hop.py
输出:
  rel/CK017_report.json
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

REPORT_PATH = os.path.join(i011.REL_DIR, "CK017_report.json")
TRAIN_NEIGH = os.path.join(i011.REL_DIR, "ECK003_neigh_train_k6.npz")
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
LAM_GRID = (0.30, 0.60)
G_GRID = (0.15, 0.30)
TOP_OUT = 5
TOP_IN = 3


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


def expand(idx, bits, tr_idx, tr_sim, lam):
    n = idx.shape[0]
    out_i = np.full((n, TOP_OUT), -1, np.int64)
    out_w = np.zeros((n, TOP_OUT), np.float32)
    n_extra = 0
    n_new = 0
    for i in tqdm(range(n), desc="    expand", unit="q", leave=False):
        cand = {}
        seeds = set()
        for j in range(TOP_IN):
            n0 = int(idx[i, j])
            if n0 < 0:
                continue
            w0 = float(bits[i, j])
            if w0 <= 0:
                continue
            seeds.add(n0)
            cand[n0] = max(cand.get(n0, 0.0), w0)
            for t in range(tr_idx.shape[1]):
                n1 = int(tr_idx[n0, t])
                if n1 < 0:
                    continue
                w1 = w0 * lam * float(tr_sim[n0, t])
                if w1 <= 0:
                    continue
                cand[n1] = max(cand.get(n1, 0.0), w1)
                n_extra += 1
        if not cand:
            continue
        items = sorted(cand.items(), key=lambda x: -x[1])[:TOP_OUT]
        for j, (nid, w) in enumerate(items):
            out_i[i, j] = nid
            out_w[i, j] = w
            if nid not in seeds:
                n_new += 1
    return out_i, out_w, n_extra, n_new


def set_jaccard(a, b):
    ja = []
    for i in range(a.shape[0]):
        sa = set(int(x) for x in a[i] if x >= 0)
        sb = set(int(x) for x in b[i] if x >= 0)
        if not sa and not sb:
            continue
        ja.append(len(sa & sb) / max(len(sa | sb), 1))
    return float(np.mean(ja)) if ja else 0.0


def main():
    t0 = time.time()
    print("=" * 60)
    print("CK017: DIAMOND 邻居扩一跳")

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

    graph = np.load(TRAIN_NEIGH)
    tr_idx, tr_sim = graph["idx"], graph["sim"]
    print(f"  train 图 {tr_idx.shape}")

    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    vp = np.load(os.path.join(i011.REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(i011.REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(i011.REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]

    s019 = build_fusion(deep, sim_bit, sim_kmer, y_train, y_val)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f}")

    rows = [rec0]
    configs = {"i019": s019}
    stats = {}

    print("[1/2] λ × γ ...")
    for lam in tqdm(LAM_GRID, desc="    λ", unit="l"):
        ei, ew, n_ex, n_new = expand(idx_v, bits_v, tr_idx, tr_sim, lam)
        jac = set_jaccard(idx_v[:, :TOP_IN], ei)
        stats[f"lam{lam:.2f}"] = {
            "jaccard": round(jac, 4),
            "n_extra_edges": int(n_ex),
            "n_new_slots": int(n_new),
        }
        print(f"  λ={lam:.2f} Jaccard={jac:.3f} 新槽 {n_new}")
        sim_e = i011.agg_labels(ei, ew, y_train, top_k=TOP_OUT)
        corr = float(np.corrcoef(sim_bit[hit].ravel(), sim_e[hit].ravel())[0, 1]) if hit.any() else 0.0
        s_e = build_fusion(deep, sim_e, sim_kmer, y_train, y_val)
        rec_e = i011.eval_ens(s_e, y_val, hit, miss, f"hop_l{lam:.2f}", {"lam": lam})
        rec_e["delta_i019"] = round(rec_e["f1_100"] - rec0["f1_100"], 6)
        rec_e["corr_bit"] = round(corr, 6)
        rec_e["jaccard"] = round(jac, 6)
        rows.append(rec_e)
        configs[f"hop_l{lam:.2f}"] = s_e
        print(f"  hop_l{lam:.2f} F1={rec_e['f1_100']:.4f} d={rec_e['delta_i019']:+.4f} "
              f"corr={corr:.3f}")
        s_n = i011.col_norm(s_e)
        for g in G_GRID:
            mixed = (1.0 - g) * s019 + g * s_n
            name = f"mix_l{lam:.2f}_g{g:.2f}"
            rec = i011.eval_ens(mixed, y_val, hit, miss, name, {"lam": lam, "gamma": g})
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
        "expand_stats": stats,
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
