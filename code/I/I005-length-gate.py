"""
I005-length-gate — 短序列上抬 CNN、压 sim（ProteInfer 式互补）
================================================================
依据:
  - ProteInfer (Sanderson, Bileschi, Belanger & Colwell, eLife 2023,
    doi:10.7554/eLife.80942)：CNN 与 BLAST 学到不同信息，简单集成超过任一路；
    远程同源时单路 CNN 低于 BLAST，但仍互补。
  - CK005a miss 分层：≤128aa miss 率 23.9%（410/1705），>1500aa 仅 2.2%。
    无同源缺口是短序列，不是截断。CK005 已证 miss 时关 sim 全局为负。
方案:
  复用 CK005a 四路概率。对长度 < L 的验证样本把 sim 权从 0.75 降到 s_short，
  并把腾出的质量加到 CNN（ProteInfer：比对弱时靠 CNN）。
  网格 L ∈ {128, 256, 400} × s_short ∈ {0.0, 0.25, 0.5, 0.75}。
  主指标: 全局 F1 不掉 + 短序列/miss F1 升。
用法:
  python code/I/I005-length-gate.py
输出:
  rel/I005_report.json
"""

import json
import os
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
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
REPORT_PATH = os.path.join(REL_DIR, "I005_report.json")
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


def main():
    t0 = time.time()
    print("=" * 60)
    print("I005: 短序列抬 CNN / 压 sim（ProteInfer 互补）")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    val_len = train_df["sequence"].str.len().values[idx_val].astype(np.int32)
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    print(f"  miss {int(miss.sum())}/{len(miss)}  short128={(val_len < 128).mean():.3f}")

    sim_bit = agg_labels(idx_v, bits_v, y_train)
    sim_kmer = build_kmer_val(y_train)
    sim = ALIGN_W * sim_bit + (1 - ALIGN_W) * sim_kmer

    vp = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    t001a = t1["t001a_val"]
    t002 = t2["t002"]
    cnn = vp["cnn"]

    def make_ens(sim_w_vec, cnn_extra=0.0):
        # 默认 deep = 0.1 t001a + 0.2 t002 + 0.2 cnn；短序列把 (SIM_W-s_short) 加到 cnn
        return (w["t001a"] * t001a + w["t002"] * t002
                + (w["cnn"] + cnn_extra) * cnn + sim_w_vec[:, None] * sim)

    def eval_ens(ens, t=None):
        if t is None:
            t = search_t(ens, y_val)
        f_all = macro_f1(ens, y_val, t)
        r, p = macro_rp(ens, y_val, t)
        auc = macro_auc(y_val, ens)
        f_miss = macro_f1(ens[miss], y_val[miss], t)
        f_hit = macro_f1(ens[hit], y_val[hit], t)
        f_s128 = macro_f1(ens[val_len < 128], y_val[val_len < 128], t)
        f_s256 = macro_f1(ens[val_len < 256], y_val[val_len < 256], t)
        return f_all, r, p, auc, f_miss, f_hit, f_s128, f_s256, t

    print("[基线 CK005a 固定权]")
    ens0 = make_ens(np.full(len(idx_val), SIM_W, np.float32), 0.0)
    b_all, b_r, b_p, b_auc, b_miss, b_hit, b_s128, b_s256, t0th = eval_ens(ens0)
    print(f"  F1={b_all:.4f} missF1={b_miss:.4f} s128={b_s128:.4f} s256={b_s256:.4f}")

    rows = []
    print("[长度门控网格]")
    for L in [128, 256, 400]:
        short = val_len < L
        for s_short in [0.0, 0.25, 0.5, 0.75]:
            sim_w = np.full(len(idx_val), SIM_W, np.float32)
            extra = np.zeros(len(idx_val), np.float32)
            sim_w[short] = s_short
            extra[short] = SIM_W - s_short
            # extra 是样本级 CNN 增量；make_ens 里 cnn_extra 是标量，改为逐样本
            ens = (w["t001a"] * t001a + w["t002"] * t002
                   + (w["cnn"] + extra)[:, None] * cnn + sim_w[:, None] * sim)
            f_all, r, p, auc, f_miss, f_hit, f_s128, f_s256, _ = eval_ens(ens)
            rec = {
                "L": L, "s_short": s_short, "n_short": int(short.sum()),
                "f1_100": round(f_all, 6), "recall": round(r, 6),
                "precision": round(p, 6), "auc": round(auc, 6),
                "f1_miss": round(f_miss, 6), "f1_hit": round(f_hit, 6),
                "f1_s128": round(f_s128, 6), "f1_s256": round(f_s256, 6),
                "d_all": round(f_all - b_all, 6),
                "d_miss": round(f_miss - b_miss, 6),
                "d_s128": round(f_s128 - b_s128, 6),
            }
            rows.append(rec)
            print(f"  L={L} s={s_short}: F1={f_all:.4f} miss={f_miss:.4f} "
                  f"s128={f_s128:.4f} Δ={f_all - b_all:+.4f}", flush=True)

    ranked = [r for r in rows if r["d_all"] >= -0.001]
    best = max(ranked, key=lambda r: (r["d_miss"], r["d_all"])) if ranked else rows[0]
    useful = best["d_miss"] >= 0.005 and best["d_all"] >= -0.001
    print(f"  最优 L={best['L']} s={best['s_short']} "
          f"Δmiss={best['d_miss']:+.4f} ΔF1={best['d_all']:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "baseline": {
            "f1_100": round(b_all, 6), "recall": round(b_r, 6),
            "precision": round(b_p, 6), "auc": round(b_auc, 6),
            "f1_miss": round(b_miss, 6), "f1_hit": round(b_hit, 6),
            "f1_s128": round(b_s128, 6), "f1_s256": round(b_s256, 6),
        },
        "grid": rows, "best": best, "useful": useful,
        "seconds": round(time.time() - t0),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
