"""
I011-rrf — 实例内 Reciprocal Rank Fusion（deep vs sim）
================================================================
依据:
  - Reciprocal Rank Fusion: Cormack, Clarke & Buettcher, SIGIR 2009,
    doi:10.1145/1571941.1572114。两路排序按 1/(k+rank) 相加，不校准分数。
  - I010：逐标签 SIM_W 全 val 上界 +0.0043，半折 −0.0001。量纲混权过拟合。
自己的改法:
  每条蛋白把 500 标签按 deep、按 sim 各排一次名，RRF 融合。
  对照：纯 RRF（丢掉幅度）；混合 CK005a 分数 + β·RRF（保留幅度）。
  k ∈ {10,30,60}，β ∈ {0.05,0.10,0.20,0.50}。主指标全局 F1，门槛 +0.001。
用法:
  python code/I/I011-rrf.py
  python code/I/I011-rrf.py --gen-sub
输出:
  rel/I011_report.json
  submit/submission_I011_ensemble.csv（--gen-sub）
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
REPORT_PATH = os.path.join(REL_DIR, "I011_report.json")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
ALIGN_W = 0.4
SIM_W = 0.75
TOP_K = 3
MAX_ACCEPT = 5
RANDOM_STATE = 42
K_GRID = (10, 30, 60)
BETA_GRID = (0.05, 0.10, 0.20, 0.50)


def parse_aln(tsv, query_ids, id_map, evalue=0.001):
    idx = np.full((len(query_ids), MAX_ACCEPT), -1, np.int64)
    bits = np.zeros((len(query_ids), MAX_ACCEPT), np.float32)
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
                bs, ev = float(p[2]), float(p[3])
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
                idx[i, j] = id_map[t]
                bits[i, j] = bs
    return idx, bits


def agg_labels(idx, w, y_space, top_k=TOP_K):
    n = idx.shape[0]
    valid = idx[:, :top_k] >= 0
    lab = np.zeros((n, top_k, y_space.shape[1]), np.float32)
    for j in range(top_k):
        lab[:, j] = y_space[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :top_k], 1e-7)
    ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)


def build_kmer_val(y_train):
    sim = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        prefix = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{prefix}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_train[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        s = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        sim += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return sim


def ranks(score):
    """0 = 该行最高分。稳定：同分按列序。"""
    return np.argsort(np.argsort(-score, axis=1, kind="mergesort"), axis=1)


def rrf_pair(deep, sim, k):
    return 1.0 / (k + ranks(deep) + 1) + 1.0 / (k + ranks(sim) + 1)


def col_norm(x):
    lo = x.min(axis=0)
    hi = x.max(axis=0)
    return (x - lo) / np.maximum(hi - lo, 1e-9)


def search_t(ens, y):
    cands = np.linspace(0.01, 0.99, 100)
    best = np.full(N_LABELS, 0.5, dtype=np.float64)
    for i in range(N_LABELS):
        yt = y[:, i]
        if yt.sum() == 0:
            continue
        pr = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0)
        fp = (pr & ~ytm).sum(0)
        fn = (~pr & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best[i] = cands[f1t.argmax()]
    return best


def search_t_minmax(ens, y):
    """RRF 分数很小，按列 min-max 后再搜 100 档。"""
    lo = ens.min(axis=0)
    hi = ens.max(axis=0)
    scaled = (ens - lo) / np.maximum(hi - lo, 1e-9)
    return search_t(scaled, y), scaled


def macro_f1(pred, y):
    f1s = []
    for i in range(y.shape[1]):
        yt = y[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i].astype(bool)
        tp = int((p & yt).sum())
        fp = int((p & ~yt).sum())
        fn = int((~p & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def macro_rp(pred, y):
    rs, ps = [], []
    for i in range(y.shape[1]):
        yt = y[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i].astype(bool)
        tp = int((p & yt).sum())
        fp = int((p & ~yt).sum())
        fn = int((~p & yt).sum())
        rs.append(tp / (tp + fn) if tp + fn else 0.0)
        ps.append(tp / (tp + fp) if tp + fp else 0.0)
    return float(np.mean(rs)), float(np.mean(ps))


def macro_auc(y, p):
    aucs = []
    for i in range(y.shape[1]):
        yt = y[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, p[:, i]))
    return float(np.mean(aucs))


def eval_ens(ens, y, hit, miss, name, extra=None, scale=False):
    if scale:
        t, ens_use = search_t_minmax(ens, y)
    else:
        t = search_t(ens, y)
        ens_use = ens
    pred = (ens_use >= t).astype(np.int8)
    r, p = macro_rp(pred, y)
    rec = {
        "name": name,
        "f1_100": round(macro_f1(pred, y), 6),
        "recall": round(r, 6),
        "precision": round(p, 6),
        "auc": round(macro_auc(y, ens_use), 6),
        "f1_miss": round(macro_f1(pred[miss], y[miss]), 6) if miss.any() else 0.0,
        "f1_hit": round(macro_f1(pred[hit], y[hit]), 6) if hit.any() else 0.0,
        "pred_card": round(float(pred.sum(1).mean()), 3),
    }
    if extra:
        rec.update(extra)
    return rec


def main():
    t0 = time.time()
    print("=" * 60)
    print("I011: 实例内 RRF（deep vs sim）")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    if not os.path.exists(tsv):
        raise FileNotFoundError("需要 CK005a very-sensitive val tsv")
    idx_v, bits_v = parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    print(f"  hit {int(hit.sum())}/{len(hit)} miss {int(miss.sum())}")

    sim_bit = agg_labels(idx_v, bits_v, y_train)
    sim_kmer = build_kmer_val(y_train)
    sim = ALIGN_W * sim_bit + (1 - ALIGN_W) * sim_kmer
    vp = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    base_ens = deep + SIM_W * sim

    rows = []
    print("[1/3] CK005a 基线...")
    rec = eval_ens(base_ens, y_val, hit, miss, "ck005a")
    rows.append(rec)
    print(f"  {rec['name']:24s} F1={rec['f1_100']:.4f} miss={rec['f1_miss']:.4f}")

    print("[2/4] 纯 RRF...")
    for k in tqdm(K_GRID, desc="    k", unit="k"):
        rrf = rrf_pair(deep, sim, k)
        rec = eval_ens(rrf, y_val, hit, miss, f"rrf_k{k}", {"k": k}, scale=True)
        rec["delta"] = round(rec["f1_100"] - rows[0]["f1_100"], 6)
        rows.append(rec)
        print(f"  {rec['name']:24s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f}")

    print("[3/4] CK005a + β·RRF...")
    hybrids = {}
    for k in tqdm(K_GRID, desc="    k", unit="k"):
        rrf = rrf_pair(deep, sim, k)
        rrf_s = col_norm(rrf)
        for b in BETA_GRID:
            ens = base_ens + b * rrf_s
            rec = eval_ens(ens, y_val, hit, miss, f"hyb_k{k}_b{b:.2f}", {"k": k, "beta": b})
            rec["delta"] = round(rec["f1_100"] - rows[0]["f1_100"], 6)
            rows.append(rec)
            hybrids[(k, b)] = ens
            print(f"  {rec['name']:24s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f}")

    print("[4/4] 半折核对（阈值在拟合折搜，评估折套用）...")
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_val)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(ens):
        pred = np.zeros_like(ens, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            t = search_t(ens[fit], y_val[fit])
            pred[hold] = (ens[hold] >= t).astype(np.int8)
        return pred

    oof_base_pred = oof_pred(base_ens)
    oof_base = {
        "name": "oof_ck005a",
        "f1_100": round(macro_f1(oof_base_pred, y_val), 6),
        "f1_miss": round(macro_f1(oof_base_pred[miss], y_val[miss]), 6),
        "f1_hit": round(macro_f1(oof_base_pred[hit], y_val[hit]), 6),
    }
    print(f"  {oof_base['name']:26s} F1={oof_base['f1_100']:.4f}")
    oof_rows = [oof_base]
    for k in K_GRID:
        rrf = rrf_pair(deep, sim, k)
        rrf_s = col_norm(rrf)
        for b in BETA_GRID:
            ens = base_ens + b * rrf_s
            pred = oof_pred(ens)
            rec = {
                "name": f"oof_hyb_k{k}_b{b:.2f}",
                "k": k, "beta": b,
                "f1_100": round(macro_f1(pred, y_val), 6),
                "f1_miss": round(macro_f1(pred[miss], y_val[miss]), 6),
                "f1_hit": round(macro_f1(pred[hit], y_val[hit]), 6),
            }
            rec["delta"] = round(rec["f1_100"] - oof_base["f1_100"], 6)
            oof_rows.append(rec)
            print(f"  {rec['name']:26s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f}")

    base = rows[0]["f1_100"]
    ranked = [r for r in rows if r["name"].startswith("hyb_")]
    best = max(ranked, key=lambda r: r["f1_100"])
    best_oof = max((r for r in oof_rows if r["name"] != "oof_ck005a"),
                   key=lambda r: r["f1_100"])
    gain = best["f1_100"] - base
    gain_oof = best_oof["f1_100"] - oof_base["f1_100"]
    useful = bool(gain_oof >= 0.001)
    print(f"  全 val 最优 {best['name']} F1={best['f1_100']:.4f} Δ={gain:+.4f}")
    print(f"  OOF 最优 {best_oof['name']} F1={best_oof['f1_100']:.4f} Δ={gain_oof:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    gen_sub = "--gen-sub" in sys.argv
    report = {
        "baseline": rows[0],
        "best": best,
        "best_oof": best_oof,
        "gain_all_f1": round(gain, 6),
        "gain_oof_f1": round(gain_oof, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")

    if gen_sub:
        print("[5/5] 按全 val 最优 (k,β) 出提交...")
        k_s, b_s = int(best["k"]), float(best["beta"])
        test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        tsv_test = os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
        if not os.path.exists(tsv_test):
            raise FileNotFoundError("需要 CK005a very-sensitive test tsv")
        full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
        idx_t, bits_t = parse_aln(tsv_test, test_df["protein_id"].values, full_map)
        sim_bit_t = agg_labels(idx_t, bits_t, y_full)
        sim_kmer_t = 0.0
        for kk, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
            if wt <= 0:
                continue
            d = np.load(os.path.join(
                REL_DIR, "TCK002a_neigh_full.npz" if kk == 6
                else f"CK004_neigh_k{kk}_full.npz"))
            n2 = d["idx_test"].shape[0]
            lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
            s = d["sim_test"][:, :5].astype(np.float32)
            ww = np.maximum(s, 1e-7) ** 2
            sim_kmer_t = sim_kmer_t + wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
        sim_t = ALIGN_W * sim_bit_t + (1.0 - ALIGN_W) * sim_kmer_t
        tp = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))
        t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
        t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))
        deep_t = w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"] + w["cnn"] * tp["cnn"]
        base_t = deep_t + SIM_W * sim_t
        rrf_t = rrf_pair(deep_t, sim_t, k_s)
        ens_t = base_t + b_s * col_norm(rrf_t)
        ens_v = hybrids[(k_s, b_s)]
        t_sub = search_t(ens_v, y_val)
        y_pred = (ens_t >= t_sub).astype(np.int8)
        os.makedirs(SUB_DIR, exist_ok=True)
        sub = pd.concat([test_df[["protein_id"]],
                         pd.DataFrame(y_pred, columns=label_cols)], axis=1)
        sub_path = os.path.join(SUB_DIR, "submission_I011_ensemble.csv")
        sub.to_csv(sub_path, index=False)
        report["submission"] = sub_path
        report["submit_k"] = k_s
        report["submit_beta"] = b_s
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  提交: {sub_path} ({len(sub)} 行, k={k_s} β={b_s})")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
