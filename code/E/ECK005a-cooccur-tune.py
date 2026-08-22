"""
ECK005a-cooccur-tune — ECK005 细网格：小 λ / PMI / 高分门控
================================================================
依据: 同 ECK005（ML-GCN / Li TCBB 2023）。
ECK005 最优 K=5 λ=0.05 仅 miss：missF1 +0.023，全局 −0.0005。
  λ 再大全局掉 → 本实验只扫更小 λ，并换边权与门控。
方案:
  复用 CK005a 分数。只改 miss 行。
  边权: cond = P(j|i) | pmi = max(log P(i,j)/(P(i)P(j)), 0)
  每行 top-K，K∈{3,5}；λ∈{0.01,0.02,0.03,0.05}
  门控: extra = (score * 1[score≥τ]) @ P，τ∈{0, 0.3, 0.5}
  主指标: 全局 F1 ≥ 基线 −0.001 且 miss 升；提交门槛全局 +0.001。
用法:
  python code/E/ECK005a-cooccur-tune.py
输出:
  rel/ECK005a_report.json
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
REPORT_PATH = os.path.join(REL_DIR, "ECK005a_report.json")
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


def sparse_graph(y, k, mode="cond"):
    yf = y.astype(np.float64)
    n = yf.shape[0]
    co = yf.T @ yf
    freq = yf.sum(0)
    if mode == "cond":
        w = co / np.maximum(freq[:, None], 1.0)
    else:
        pij = co / n
        pi = freq / n
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.log(pij / np.maximum(pi[:, None] * pi[None, :], 1e-12))
        w = np.maximum(w, 0.0)
        w[~np.isfinite(w)] = 0.0
    np.fill_diagonal(w, 0.0)
    out = np.zeros_like(w, np.float32)
    kk = min(k, w.shape[1] - 1)
    for i in range(w.shape[0]):
        js = np.argpartition(-w[i], kk)[:kk]
        out[i, js] = w[i, js]
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


def main():
    t0 = time.time()
    print("=" * 60)
    print("ECK005a: 细网格 λ/K + PMI + 高分门控（仅 miss）")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
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
        return f_all, r, p, auc, f_miss, f_hit

    print("[基线]")
    b_all, b_r, b_p, b_auc, b_miss, b_hit = eval_ens(ens)
    print(f"  F1={b_all:.4f} missF1={b_miss:.4f}")

    graphs = {}
    for mode in ["cond", "pmi"]:
        for K in [3, 5]:
            graphs[(mode, K)] = sparse_graph(y_train, K, mode)
            g = graphs[(mode, K)]
            print(f"  {mode} K={K} 非零 {float((g > 0).mean()):.4f} 行和 {g.sum(1).mean():.3f}")

    rows = [{
        "name": "baseline", "mode": "none", "K": 0, "lam": 0.0, "tau": 0.0,
        "f1_100": round(b_all, 6), "recall": round(b_r, 6),
        "precision": round(b_p, 6), "auc": round(b_auc, 6),
        "f1_miss": round(b_miss, 6), "f1_hit": round(b_hit, 6),
    }]

    print("[网格 mode × K × λ × τ，仅 miss]")
    for mode in ["cond", "pmi"]:
        for K in [3, 5]:
            pmat = graphs[(mode, K)]
            for tau in [0.0, 0.3, 0.5]:
                gated = ens * (ens >= tau)
                extra = gated @ pmat
                for lam in [0.01, 0.02, 0.03, 0.05]:
                    e = ens.copy()
                    e[miss] = ens[miss] + lam * extra[miss]
                    f_all, r, p, auc, f_miss, f_hit = eval_ens(e)
                    rec = {
                        "name": f"{mode}_K{K}_l{lam}_t{tau}",
                        "mode": mode, "K": K, "lam": lam, "tau": tau,
                        "f1_100": round(f_all, 6), "recall": round(r, 6),
                        "precision": round(p, 6), "auc": round(auc, 6),
                        "f1_miss": round(f_miss, 6), "f1_hit": round(f_hit, 6),
                    }
                    rows.append(rec)
                    print(f"  {rec['name']}: F1={f_all:.4f} miss={f_miss:.4f} "
                          f"Δ={f_all - b_all:+.4f}", flush=True)

    ranked = [r for r in rows[1:] if r["f1_100"] >= b_all - 0.001]
    if ranked:
        best = max(ranked, key=lambda r: (r["f1_100"], r["f1_miss"]))
    else:
        best = max(rows[1:], key=lambda r: r["f1_100"])
    gain_all = best["f1_100"] - b_all
    gain_miss = best["f1_miss"] - b_miss
    useful = gain_all >= 0.001
    print(f"  最优 {best['name']} ΔF1={gain_all:+.4f} Δmiss={gain_miss:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "baseline": rows[0], "grid": rows, "best": best,
        "gain_all": round(gain_all, 6), "gain_miss": round(gain_miss, 6),
        "useful": useful, "seconds": round(time.time() - t0),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
