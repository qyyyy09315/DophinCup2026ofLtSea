"""
ECK005-cooccur — 训练共现图对分数做软传播（SCut 之前）
================================================================
依据:
  - ML-GCN (Chen, Wei, Wang & Guo, CVPR 2019, doi:10.1109/cvpr.2019.00532)
    标签共现图卷积分类器，不是决策后硬补。
  - Li et al., IEEE/ACM TCBB 2023, doi:10.1109/TCBB.2022.3170719
    蛋白功能显式 GO 共现；本赛标签匿名，用经验共现。
  - 对照: TCK002b 稠密 p@P 抹平（负）；ECK002 决策后硬补 Δ=0。
    本实验: 稀疏 P 作用在分数上，再 SCut。
  - 数据: CK005a miss 漏标 96.9% 与已预测标签在 train 共现；FN 分数 mean=0.27。
方案:
  复用 CK005a 四路分数。P 从 idx_train 算条件概率 P(j|i)，每行只留 top-K，
  对角置 0。score' = score + λ (score @ P)。
  网格: K∈{5,10,20} × λ∈{0.05,0.1,0.2,0.4} × {全局, 仅 miss}。
  主指标: 全局 F1 不掉 + miss F1。
用法:
  python code/E/ECK005-cooccur.py
  python code/E/ECK005-cooccur.py --gen-sub
输出:
  rel/ECK005_report.json
  submit/submission_ECK005_ensemble.csv（仅 --gen-sub 且有效）
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
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
REPORT_PATH = os.path.join(REL_DIR, "ECK005_report.json")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
ALIGN_W = 0.4
SIM_W = 0.75
TOP_K = 3
MAX_ACCEPT = 5
RANDOM_STATE = 42


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


def build_kmer_test(y_full):
    sim = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        d = np.load(os.path.join(REL_DIR, "TCK002a_neigh_full.npz" if k == 6
                                 else f"CK004_neigh_k{k}_full.npz"))
        n2 = d["idx_test"].shape[0]
        lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
        s = d["sim_test"][:, :5].astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        sim += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return sim


def cond_sparse(y, k):
    """P(j|i)：给定标签 i 阳性时 j 的条件概率；每行 top-k，对角 0。"""
    yf = y.astype(np.float64)
    co = yf.T @ yf
    freq = yf.sum(0)
    p = co / np.maximum(freq[:, None], 1.0)
    np.fill_diagonal(p, 0.0)
    out = np.zeros_like(p, np.float32)
    for i in range(p.shape[0]):
        js = np.argpartition(-p[i], k)[:k]
        out[i, js] = p[i, js]
    return out


def macro_f1(ens, y, t):
    pred = ens >= t
    f1s = []
    for i in range(y.shape[1]):
        yt = y[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum())
        fp = int((p & ~yt).sum())
        fn = int((~p & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def macro_rp(ens, y, t):
    pred = ens >= t
    rs, ps = [], []
    for i in range(y.shape[1]):
        yt = y[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum())
        fp = int((p & ~yt).sum())
        fn = int((~p & yt).sum())
        rs.append(tp / (tp + fn) if tp + fn else 0.0)
        ps.append(tp / (tp + fp) if tp + fp else 0.0)
    return float(np.mean(rs)), float(np.mean(ps))


def search_t(ens, y):
    cands = np.linspace(0.01, 0.99, 100)
    best = np.full(N_LABELS, 0.5)
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


def propagate(score, p, lam):
    extra = score @ p
    return score + lam * extra


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print("ECK005: 稀疏共现图对 CK005a 分数软传播")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim = ALIGN_W * agg_labels(idx_v, bits_v, y_train) + (1 - ALIGN_W) * build_kmer_val(y_train)
    vp = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    ens = (w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"]
           + w["cnn"] * vp["cnn"] + SIM_W * sim)

    def eval_ens(e):
        t = search_t(e, y_val)
        f_all = macro_f1(e, y_val, t)
        r, p = macro_rp(e, y_val, t)
        auc = float(np.mean([
            roc_auc_score(y_val[:, i], e[:, i])
            for i in range(N_LABELS)
            if 0 < y_val[:, i].sum() < len(y_val)]))
        f_miss = macro_f1(e[miss], y_val[miss], t)
        f_hit = macro_f1(e[hit], y_val[hit], t)
        return f_all, r, p, auc, f_miss, f_hit, t

    print("[基线 CK005a]")
    b_all, b_r, b_p, b_auc, b_miss, b_hit, t0th = eval_ens(ens)
    print(f"  F1={b_all:.4f} missF1={b_miss:.4f} hitF1={b_hit:.4f}")

    rows = [{
        "name": "baseline", "K": 0, "lam": 0.0, "scope": "none",
        "f1_100": round(b_all, 6), "recall": round(b_r, 6),
        "precision": round(b_p, 6), "auc": round(b_auc, 6),
        "f1_miss": round(b_miss, 6), "f1_hit": round(b_hit, 6),
    }]

    print("[网格 K × λ × scope]")
    p_cache = {}
    for K in [5, 10, 20]:
        p_cache[K] = cond_sparse(y_train, K)
        nz = float((p_cache[K] > 0).mean())
        print(f"  P K={K} 非零率 {nz:.4f} 行均值 {p_cache[K].sum(1).mean():.3f}")

    for K in tqdm([5, 10, 20], desc="K", unit="k"):
        pmat = p_cache[K]
        for lam in [0.05, 0.1, 0.2, 0.4]:
            extra = ens @ pmat
            for scope in ["all", "miss"]:
                e = ens.copy()
                if scope == "all":
                    e = ens + lam * extra
                else:
                    e[miss] = ens[miss] + lam * extra[miss]
                f_all, r, p, auc, f_miss, f_hit, _ = eval_ens(e)
                rec = {
                    "name": f"K{K}_l{lam}_{scope}",
                    "K": K, "lam": lam, "scope": scope,
                    "f1_100": round(f_all, 6), "recall": round(r, 6),
                    "precision": round(p, 6), "auc": round(auc, 6),
                    "f1_miss": round(f_miss, 6), "f1_hit": round(f_hit, 6),
                }
                rows.append(rec)
                print(f"  {rec['name']}: F1={f_all:.4f} miss={f_miss:.4f} "
                      f"Δ={f_all - b_all:+.4f}", flush=True)

    ranked = [r for r in rows[1:] if r["f1_100"] >= b_all - 0.001]
    if ranked:
        best = max(ranked, key=lambda r: (r["f1_miss"], r["f1_100"]))
    else:
        best = max(rows[1:], key=lambda r: r["f1_100"])
    gain_all = best["f1_100"] - b_all
    gain_miss = best["f1_miss"] - b_miss
    useful = gain_all >= 0.001 and best["name"] != "baseline"
    print(f"  最优 {best['name']} ΔF1={gain_all:+.4f} Δmiss={gain_miss:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "baseline": rows[0], "grid": rows, "best": best,
        "gain_all": round(gain_all, 6), "gain_miss": round(gain_miss, 6),
        "useful": useful, "seconds": round(time.time() - t0),
    }

    if not (gen_sub and useful):
        with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  报告: {REPORT_PATH}")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    print("[test] 最优配置生成提交...")
    tsv_t = os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
    full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
    idx_t, bits_t = parse_aln(tsv_t, test_df["protein_id"].values, full_map)
    sim_t = (ALIGN_W * agg_labels(idx_t, bits_t, y_full)
             + (1 - ALIGN_W) * build_kmer_test(y_full))
    tp = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))
    t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))
    ens_t = (w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"]
             + w["cnn"] * tp["cnn"] + SIM_W * sim_t)
    pmat = cond_sparse(y_full, best["K"])
    extra_t = ens_t @ pmat
    if best["scope"] == "all":
        ens_t = ens_t + best["lam"] * extra_t
    else:
        miss_t = idx_t[:, 0] < 0
        ens_t[miss_t] = ens_t[miss_t] + best["lam"] * extra_t[miss_t]
    e_val = ens.copy()
    extra_v = ens @ pmat
    if best["scope"] == "all":
        e_val = ens + best["lam"] * extra_v
    else:
        e_val[miss] = ens[miss] + best["lam"] * extra_v[miss]
    t = search_t(e_val, y_val)
    pred = (ens_t >= t).astype(np.int8)
    os.makedirs(SUB_DIR, exist_ok=True)
    sub = pd.concat([test_df[["protein_id"]],
                     pd.DataFrame(pred, columns=label_cols)], axis=1)
    sub_path = os.path.join(SUB_DIR, "submission_ECK005_ensemble.csv")
    sub.to_csv(sub_path, index=False)
    report["submission"] = sub_path
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  提交: {sub_path} ({len(sub)} 行)")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
