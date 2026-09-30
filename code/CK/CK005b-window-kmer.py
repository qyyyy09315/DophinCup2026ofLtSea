"""
CK005b-window-kmer — miss 蛋白的短窗口 k-mer 局部同源
================================================================
依据:
  - Domain-PFP (Ibtehaz, Kagaya & Kihara, Commun Biol 2023,
    doi:10.1038/s42003-023-05476-9)：功能在结构域而非全长；本赛无 InterPro/GO
    映射，用查询短窗口 × 训练蛋白 6-mer 种子作零外部库近似。
  - pLM-BLAST (Kaminski et al., Bioinformatics 2023,
    doi:10.1093/bioinformatics/btad579)：远程同源常是局部的；E004a 冻结 PLM
    读出已失败，这里用窗口 k-mer 种子代替残基嵌入局部比对。
  - CK005a very-sensitive 后仍有 1748/22760 miss，missF1 仅 0.067。
方案:
  只处理 CK005a miss 查询：切重叠窗口（默认 64aa，步长 32），
  倒排 6-mer（丢弃 df>DF_MAX 的低复杂度种子）在 train 蛋白库取每窗 top-1，
  窗口命中标签按 idf 加权聚合。非 miss 行保持 CK005a 的 align+kmer sim。
  主指标: miss 子集 F1；全局不得低于 CK005a 0.4399 超过噪声。
用法:
  python code/CK/CK005b-window-kmer.py
  python code/CK/CK005b-window-kmer.py --win=48 --step=24 --k=6
输出:
  rel/CK005b_report.json
"""

import json
import os
import sys
import time
from collections import defaultdict

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
REPORT_PATH = os.path.join(REL_DIR, "CK005b_report.json")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
ALIGN_W = 0.4
SIM_W = 0.75
TOP_K = 3
MAX_ACCEPT = 5
WIN = 64
STEP = 32
KMER = 6
DF_MAX = 200
MIN_SHARED = 3
RANDOM_STATE = 42


def parse_args():
    win, step, k, df_max = WIN, STEP, KMER, DF_MAX
    for a in sys.argv[1:]:
        if a.startswith("--win="):
            win = int(a.split("=", 1)[1])
        elif a.startswith("--step="):
            step = int(a.split("=", 1)[1])
        elif a.startswith("--k="):
            k = int(a.split("=", 1)[1])
        elif a.startswith("--df-max="):
            df_max = int(a.split("=", 1)[1])
    return win, step, k, df_max


def windows(seq, win, step):
    n = len(seq)
    if n <= win:
        return [seq]
    out = [seq[i:i + win] for i in range(0, n - win + 1, step)]
    if out[-1] != seq[-win:]:
        out.append(seq[-win:])
    return out


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


def unique_kmers(seq, k):
    n = len(seq) - k + 1
    if n <= 0:
        return []
    return list({seq[i:i + k] for i in range(n)})


def build_posting(seqs, k, df_max):
    raw = defaultdict(list)
    for i, s in enumerate(tqdm(seqs, desc="    建倒排", unit="seq")):
        for mer in unique_kmers(s, k):
            raw[mer].append(i)
    posting, idf = {}, {}
    n = max(len(seqs), 1)
    n_drop = 0
    for mer, ids in raw.items():
        df = len(ids)
        if df > df_max:
            n_drop += 1
            continue
        posting[mer] = np.asarray(ids, np.int32)
        idf[mer] = float(np.log((n + 1) / (df + 1)) + 1.0)
    print(f"  6-mer 保留 {len(posting)} / 丢弃高频 {n_drop} (df>{df_max})", flush=True)
    return posting, idf


def window_sim(q_seqs, db_seqs, y_db, win, step, k, df_max, min_shared=MIN_SHARED):
    """miss 查询窗口 → train 蛋白 6-mer 倒排 top-1，按蛋白聚合标签。"""
    print(f"  切窗 win={win} step={step} k={k} df_max={df_max} ...", flush=True)
    posting, idf = build_posting(db_seqs, k, df_max)
    n_q = len(q_seqs)
    acc = np.zeros((n_q, N_LABELS), np.float32)
    wsum = np.zeros(n_q, np.float32)
    n_hit_win = 0
    n_win = 0
    top1_scores = []

    for qi, seq in enumerate(tqdm(q_seqs, desc="    窗口检索", unit="prot")):
        for wseq in windows(seq, win, step):
            n_win += 1
            parts, wts, cnts = [], [], []
            for mer in unique_kmers(wseq, k):
                ids = posting.get(mer)
                if ids is None:
                    continue
                parts.append(ids)
                wts.append(np.full(len(ids), idf[mer], np.float32))
                cnts.append(np.ones(len(ids), np.int16))
            if not parts:
                continue
            ids = np.concatenate(parts)
            ww = np.concatenate(wts)
            cc = np.concatenate(cnts)
            u, inv = np.unique(ids, return_inverse=True)
            sc = np.bincount(inv, weights=ww)
            sh = np.bincount(inv, weights=cc)
            j = int(sc.argmax())
            if sh[j] < min_shared:
                continue
            best_t, best_s = int(u[j]), float(sc[j])
            n_hit_win += 1
            top1_scores.append(best_s)
            acc[qi] += best_s * y_db[best_t]
            wsum[qi] += best_s

    sim = acc / np.maximum(wsum[:, None], 1e-7)
    cover = float((wsum > 0).mean())
    mean_s = float(np.mean(top1_scores)) if top1_scores else 0.0
    print(f"  miss 蛋白窗口命中率 {cover:.3f} | 窗命中 {n_hit_win}/{n_win} | "
          f"top1_idf mean={mean_s:.3f}", flush=True)
    return sim, cover, mean_s


def main():
    win, step, k, df_max = parse_args()
    t0 = time.time()
    print("=" * 60)
    print(f"CK005b: miss 窗口 k-mer 局部同源  win={win} step={step} k={k}")

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
    if not os.path.exists(tsv):
        raise FileNotFoundError("需要 CK005a very-sensitive val tsv")
    idx_v, bits_v = parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    print(f"  CK005a miss {int(miss.sum())}/{len(miss)} ({miss.mean():.3f})")

    sim_bit = agg_labels(idx_v, bits_v, y_train)
    sim_kmer = build_kmer_val(y_train)
    sim_base = ALIGN_W * sim_bit + (1 - ALIGN_W) * sim_kmer

    q_seqs = seqs[idx_val][miss]
    db_seqs = seqs[idx_train]
    sim_win, cover, mean_s = window_sim(
        q_seqs, db_seqs, y_train, win, step, k, df_max)

    vp = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * vp["cnn"]

    def eval_ens(sim):
        ens = deep + SIM_W * sim
        t = search_t(ens, y_val)
        f_all = macro_f1(ens, y_val, t)
        r, p = macro_rp(ens, y_val, t)
        auc = macro_auc(y_val, ens)
        f_miss = macro_f1(ens[miss], y_val[miss], t)
        f_hit = macro_f1(ens[hit], y_val[hit], t)
        return f_all, r, p, auc, f_miss, f_hit

    print("[基线 CK005a]")
    b_all, b_r, b_p, b_auc, b_miss, b_hit = eval_ens(sim_base)
    print(f"  F1={b_all:.4f} missF1={b_miss:.4f} hitF1={b_hit:.4f}")

    print("[miss 行混入窗口 sim]")
    fusion = {}
    for wp in [0.0, 0.3, 0.5, 0.7, 1.0]:
        sim = sim_base.copy()
        sim[miss] = wp * sim_win + (1 - wp) * sim_base[miss]
        f_all, r, p, auc, f_miss, f_hit = eval_ens(sim)
        fusion[str(wp)] = {
            "f1_100": round(f_all, 6), "f1_miss": round(f_miss, 6),
            "f1_hit": round(f_hit, 6), "recall": round(r, 6),
            "precision": round(p, 6), "auc": round(auc, 6),
        }
        print(f"  w_win={wp}: F1={f_all:.4f} missF1={f_miss:.4f} hitF1={f_hit:.4f}")

    ranked = []
    for wp_s, rec in fusion.items():
        if rec["f1_100"] >= b_all - 0.001:
            ranked.append((rec["f1_miss"], rec["f1_100"], float(wp_s)))
    best_wp = max(ranked)[2] if ranked else 0.0

    gain_miss = fusion[str(best_wp)]["f1_miss"] - b_miss
    gain_all = fusion[str(best_wp)]["f1_100"] - b_all
    useful = gain_miss >= 0.005 and gain_all >= -0.001
    print(f"  最优 w_win={best_wp} Δmiss={gain_miss:+.4f} ΔF1={gain_all:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "win": win, "step": step, "k": k, "df_max": df_max,
        "miss_n": int(miss.sum()), "window_cover": round(cover, 6),
        "window_top1_idf": round(mean_s, 6),
        "baseline_ck005a": {
            "f1_100": round(b_all, 6), "f1_miss": round(b_miss, 6),
            "f1_hit": round(b_hit, 6), "recall": round(b_r, 6),
            "precision": round(b_p, 6), "auc": round(b_auc, 6),
        },
        "fusion_w_win": fusion, "best_w_win": best_wp,
        "gain_miss": round(gain_miss, 6), "gain_all": round(gain_all, 6),
        "useful": useful, "seconds": round(time.time() - t0),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
