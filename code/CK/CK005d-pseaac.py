"""
CK005d-pseaac — Chou 伪氨基酸组成接到 CK005a 四路
================================================================
依据:
  - Chou KC, Proteins 2001, doi:10.1002/prot.1035
    氨基酸频率丢掉序列阶；伪氨基酸组成用疏水/亲水相关滞后补回。
  - Chou KC, Bioinformatics 2004, doi:10.1093/bioinformatics/bth466
    两亲性 PseAAC 预测酶亚家族。
  - 待办「特征工程 CTD、PseAAC」尚未做；B001 只用了 20 维 AA 频率。
方案:
  Type-I PseAAC：20 维 AA 频率 + λ=10 维疏水相关滞后 + λ=10 维亲水相关滞后
  （JTT 标准化 Kyte–Doolittle / Hopp–Woods）。线性头在 train 上拟合 500 标签，
  概率作为第五路，网格扫 w_pse。主指标：全局 F1 不掉。
用法:
  python code/CK/CK005d-pseaac.py
输出:
  rel/CK005d_report.json
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
REPORT_PATH = os.path.join(REL_DIR, "CK005d_report.json")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
ALIGN_W = 0.4
SIM_W = 0.75
TOP_K = 3
MAX_ACCEPT = 5
LAMBDA = 10
WEIGHT_FREQ = 0.05
RANDOM_STATE = 42
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Kyte–Doolittle hydrophobicity / Hopp–Woods hydrophilicity
HYDRO = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
PHIL = {
    "A": -0.5, "C": -1.0, "D": 3.0, "E": 3.0, "F": -2.5,
    "G": 0.0, "H": -0.5, "I": -1.8, "K": 3.0, "L": -1.8,
    "M": -1.3, "N": 0.2, "P": 0.0, "Q": 0.2, "R": 3.0,
    "S": 0.3, "T": -0.4, "V": -1.5, "W": -3.4, "Y": -2.3,
}


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


def macro_f1(ens, y, t=0.5):
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


def macro_auc(y, p):
    aucs = []
    for i in range(y.shape[1]):
        yt = y[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, p[:, i]))
    return float(np.mean(aucs))


def search_t(ens, y):
    cands = np.linspace(0.01, 0.99, 100)
    best = np.full(N_LABELS, 0.5)
    for i in tqdm(range(N_LABELS), desc="    阈值", unit="lab", leave=False):
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


def _norm_scale(d):
    vals = np.array([d[a] for a in AMINO_ACIDS], np.float64)
    mu, sd = vals.mean(), vals.std() + 1e-8
    return {a: (d[a] - mu) / sd for a in AMINO_ACIDS}


def pseaac_one(seq, hydro, phil, lam=LAMBDA, w=WEIGHT_FREQ):
    L = len(seq)
    freq = np.zeros(20, np.float32)
    aa_idx = {a: i for i, a in enumerate(AMINO_ACIDS)}
    hs = np.zeros(L, np.float32)
    ps = np.zeros(L, np.float32)
    for i, c in enumerate(seq):
        if c in aa_idx:
            freq[aa_idx[c]] += 1
            hs[i] = hydro[c]
            ps[i] = phil[c]
    if L > 0:
        freq /= L
    theta = np.zeros(2 * lam, np.float32)
    for k in range(1, lam + 1):
        if L <= k:
            break
        d_h = hs[k:] - hs[:-k]
        d_p = ps[k:] - ps[:-k]
        theta[k - 1] = float((d_h * d_h).mean())
        theta[lam + k - 1] = float((d_p * d_p).mean())
    denom = freq.sum() + w * theta.sum() + 1e-8
    return np.concatenate([freq / denom, w * theta / denom])


def extract_pseaac(seqs, hydro, phil):
    out = np.zeros((len(seqs), 20 + 2 * LAMBDA), np.float32)
    for i, s in enumerate(tqdm(seqs, desc="    PseAAC", unit="seq")):
        out[i] = pseaac_one(s, hydro, phil)
    return out


def main():
    t0 = time.time()
    print("=" * 60)
    print(f"CK005d: Chou Type-I PseAAC λ={LAMBDA} 接到 CK005a")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    seqs = train_df["sequence"].values
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit

    sim_bit = agg_labels(idx_v, bits_v, y_train)
    sim_kmer = build_kmer_val(y_train)
    sim = ALIGN_W * sim_bit + (1 - ALIGN_W) * sim_kmer

    vp = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]
    ens0 = deep + SIM_W * sim

    hydro, phil = _norm_scale(HYDRO), _norm_scale(PHIL)
    print("[1/3] 提取 PseAAC ...")
    x_tr = extract_pseaac(seqs[idx_train], hydro, phil)
    x_va = extract_pseaac(seqs[idx_val], hydro, phil)
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_va = scaler.transform(x_va)

    print("[2/3] Ridge 多标签头 ...")
    head = Ridge(alpha=1.0)
    head.fit(x_tr, y_train.astype(np.float32))
    pse = np.clip(head.predict(x_va), 0.0, 1.0).astype(np.float32)

    def eval_ens(ens):
        t = search_t(ens, y_val)
        f_all = macro_f1(ens, y_val, t)
        r, p = macro_rp(ens, y_val, t)
        auc = macro_auc(y_val, ens)
        f_miss = macro_f1(ens[miss], y_val[miss], t)
        f_hit = macro_f1(ens[hit], y_val[hit], t)
        return f_all, r, p, auc, f_miss, f_hit

    print("[基线 CK005a]")
    b_all, b_r, b_p, b_auc, b_miss, b_hit = eval_ens(ens0)
    print(f"  F1={b_all:.4f} missF1={b_miss:.4f}")

    print("[独立 PseAAC]")
    p_all, p_r, p_p, p_auc, p_miss, p_hit = eval_ens(pse)
    print(f"  F1={p_all:.4f} missF1={p_miss:.4f} AUC={p_auc:.4f}")

    print("[3/3] 融合网格 w_pse")
    fusion = {}
    best_wp, best_key = 0.0, (b_all, b_miss)
    for wp in [0.0, 0.05, 0.1, 0.2, 0.3]:
        ens = ens0 + wp * pse
        f_all, r, p, auc, f_miss, f_hit = eval_ens(ens)
        fusion[str(wp)] = {
            "f1_100": round(f_all, 6), "f1_miss": round(f_miss, 6),
            "f1_hit": round(f_hit, 6), "recall": round(r, 6),
            "precision": round(p, 6), "auc": round(auc, 6),
        }
        print(f"  w={wp}: F1={f_all:.4f} missF1={f_miss:.4f} Δ={f_all - b_all:+.4f}")
        if f_all >= b_all - 0.001 and (f_all, f_miss) > best_key:
            best_wp, best_key = wp, (f_all, f_miss)

    gain_all = fusion[str(best_wp)]["f1_100"] - b_all
    gain_miss = fusion[str(best_wp)]["f1_miss"] - b_miss
    useful = gain_all >= 0.001 or (gain_miss >= 0.005 and gain_all >= -0.001)
    print(f"  最优 w={best_wp} ΔF1={gain_all:+.4f} Δmiss={gain_miss:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "lambda": LAMBDA,
        "baseline": {
            "f1_100": round(b_all, 6), "recall": round(b_r, 6),
            "precision": round(b_p, 6), "auc": round(b_auc, 6),
            "f1_miss": round(b_miss, 6), "f1_hit": round(b_hit, 6),
        },
        "pseaac_only": {
            "f1_100": round(p_all, 6), "recall": round(p_r, 6),
            "precision": round(p_p, 6), "auc": round(p_auc, 6),
            "f1_miss": round(p_miss, 6),
        },
        "fusion": fusion, "best_w": best_wp,
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
