"""
A001-slice-audit — C011a 误差切片与廉价基线（分析性，CPU）
================================================================
不训练、不占 GPU。重建 C011a 分数，按同源/长度/标签基数切开，
对照先验、top-k 常标签、DIAMOND top-1 转移。不写腾讯表。
用法:
  python code/tools/A001-slice-audit.py
输出:
  rel/A001_report.json
  rel/A001_slices.csv
"""

import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT_PATH = os.path.join(REL_DIR, "A001_report.json")
CSV_PATH = os.path.join(REL_DIR, "A001_slices.csv")
SPLIT_SEED = 42
C011A_ALPHA = 0.50
CNN_W = 0.2
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
AA = "ACDEFGHIKLMNPQRSTVWY"


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def aa_freq(seqs):
    n = len(seqs)
    out = np.zeros((n, 20), np.float32)
    for i, s in enumerate(seqs):
        if not s:
            continue
        c = np.zeros(20, np.float32)
        for ch in s:
            j = AA.find(ch)
            if j >= 0:
                c[j] += 1
        out[i] = c / max(c.sum(), 1.0)
    return out


def slice_f1(pred, y, mask):
    if mask.sum() == 0:
        return None
    return round(i011.macro_f1(pred[mask], y[mask]), 6)


def eval_score(score, y, name):
    t = i011.search_t(score, y)
    pred = (score >= t).astype(np.int8)
    r, p = i011.macro_rp(pred, y)
    return {
        "name": name,
        "f1_100": round(i011.macro_f1(pred, y), 6),
        "recall": round(r, 6),
        "precision": round(p, 6),
        "auc": round(i011.macro_auc(y, score), 6),
        "pred_card": round(float(pred.sum(1).mean()), 3),
        "pred": pred,
        "t": t,
    }


def main():
    global i011, i019
    t0 = time.time()
    print("=" * 60)
    print("A001: C011a slice audit (CPU)")
    i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
    i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    seqs = train_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=SPLIT_SEED)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    train_ids = train_df["protein_id"].values[idx_train]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_ids)}
    lens = np.array([len(s) for s in seqs[idx_val]])
    card = y_val.sum(1)

    print("[dup] exact sequence overlap ...")
    tr_set = set(seqs)
    te_set = set(test_df["sequence"].values)
    va_set = set(seqs[idx_val])
    trn_set = set(seqs[idx_train])
    dup = {
        "train_n": int(len(seqs)),
        "train_unique": int(len(tr_set)),
        "test_n": int(len(test_df)),
        "test_unique": int(len(te_set)),
        "train_test_exact": int(len(tr_set & te_set)),
        "train_val_exact": int(len(trn_set & va_set)),
        "train_internal_dup": int(len(seqs) - len(tr_set)),
    }
    print(" ", dup)

    print("[aln] DIAMOND val ...")
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    bit1 = bits_v[:, 0].copy()
    bit1[miss] = 0.0
    pident = np.zeros(len(val_ids), np.float32)
    qpos = {q: i for i, q in enumerate(val_ids)}
    with open(tsv) as f:
        seen = np.zeros(len(val_ids), np.int8)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            q = p[0]
            if q not in qpos or seen[qpos[q]]:
                continue
            try:
                ev = float(p[3])
                pid = float(p[4])
            except ValueError:
                continue
            if ev > 0.001:
                continue
            pident[qpos[q]] = pid
            seen[qpos[q]] = 1

    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer

    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    cnn42 = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    cnn_lock = (1.0 - C011A_ALPHA) * cnn42 + C011A_ALPHA * c005_val
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + CNN_W * cnn_lock
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep, sim, K_RRF))
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t_i011 = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t_i011).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p_sk = i019.sinkhorn(ker, row_m, col_m)
    s_lock = i019.mix(s011, p_sk, ALPHA_SK)
    rec_lock = eval_score(s_lock, y_val, "c011a")
    pred_lock = rec_lock.pop("pred")
    rec_lock.pop("t")
    print(f"  C011a F1={rec_lock['f1_100']:.4f} miss={slice_f1(pred_lock, y_val, miss)}")

    print("[base] cheap baselines ...")
    prior = np.broadcast_to(pi.astype(np.float32), y_val.shape).copy()
    rec_prior = eval_score(prior, y_val, "prior")
    pred_prior = rec_prior.pop("pred")
    rec_prior.pop("t")
    k_card = int(round(float(y_train.sum(1).mean())))
    freq_rank = np.argsort(-pi)
    maj = np.zeros_like(y_val, np.int8)
    maj[:, freq_rank[:k_card]] = 1
    rec_maj = {
        "name": f"majority_top{k_card}",
        "f1_100": round(i011.macro_f1(maj, y_val), 6),
        "recall": round(i011.macro_rp(maj, y_val)[0], 6),
        "precision": round(i011.macro_rp(maj, y_val)[1], 6),
        "pred_card": round(float(maj.sum(1).mean()), 3),
    }
    rec_bit = eval_score(sim_bit, y_val, "diamond_top3")
    pred_bit = rec_bit.pop("pred")
    rec_bit.pop("t")
    rec_kmer = eval_score(sim_kmer, y_val, "kmer")
    rec_kmer.pop("pred")
    rec_kmer.pop("t")
    rec_cnn = eval_score(cnn_lock, y_val, "cnn_lock")
    rec_cnn.pop("pred")
    rec_cnn.pop("t")
    rec_t001a = eval_score(t1["t001a_val"], y_val, "t001a")
    rec_t001a.pop("pred")
    rec_t001a.pop("t")
    rec_deep = eval_score(deep, y_val, "deep")
    rec_deep.pop("pred")
    rec_deep.pop("t")
    top1 = np.zeros_like(y_val, np.int8)
    top1[hit] = y_train[idx_v[hit, 0]]
    rec_top1 = {
        "name": "diamond_top1_copy",
        "f1_100": round(i011.macro_f1(top1, y_val), 6),
        "f1_hit": round(i011.macro_f1(top1[hit], y_val[hit]), 6),
        "f1_miss": 0.0,
        "pred_card": round(float(top1.sum(1).mean()), 3),
    }
    print(f"  prior={rec_prior['f1_100']:.4f} maj={rec_maj['f1_100']:.4f} "
          f"top1={rec_top1['f1_100']:.4f} bit={rec_bit['f1_100']:.4f}")

    print("[slice] ...")
    slices = []

    def add_slice(key, mask):
        n_m = int(mask.sum())
        if n_m == 0:
            return
        row = {
            "slice": key,
            "n": n_m,
            "frac": round(n_m / len(y_val), 4),
            "card_mean": round(float(card[mask].mean()), 2),
            "len_mean": round(float(lens[mask].mean()), 1),
            "c011a": slice_f1(pred_lock, y_val, mask),
            "prior": slice_f1(pred_prior, y_val, mask),
            "diamond": slice_f1(pred_bit, y_val, mask),
            "top1": round(i011.macro_f1(top1[mask], y_val[mask]), 6),
        }
        slices.append(row)
        print(f"  {key:22s} n={n_m:5d} C011a={row['c011a']:.4f} "
              f"prior={row['prior']:.4f} bit={row['diamond']:.4f}")

    add_slice("all", np.ones(len(y_val), bool))
    add_slice("hit", hit)
    add_slice("miss", miss)
    add_slice("bit>=200", hit & (bit1 >= 200))
    add_slice("100<=bit<200", hit & (bit1 >= 100) & (bit1 < 200))
    add_slice("50<=bit<100", hit & (bit1 >= 50) & (bit1 < 100))
    add_slice("bit<50_hit", hit & (bit1 < 50))
    add_slice("pident>=80", hit & (pident >= 80))
    add_slice("50<=pident<80", hit & (pident >= 50) & (pident < 80))
    add_slice("pident<50_hit", hit & (pident < 50) & (pident > 0))
    add_slice("L<=128", lens <= 128)
    add_slice("128<L<=256", (lens > 128) & (lens <= 256))
    add_slice("256<L<=512", (lens > 256) & (lens <= 512))
    add_slice("512<L<=1500", (lens > 512) & (lens <= 1500))
    add_slice("L>1500", lens > 1500)
    add_slice("card<=10", card <= 10)
    add_slice("11<=card<=30", (card >= 11) & (card <= 30))
    add_slice("card>30", card > 30)
    add_slice("miss_L<=256", miss & (lens <= 256))
    add_slice("miss_L>256", miss & (lens > 256))

    freq = pi
    lab_f1 = []
    for j in range(y_val.shape[1]):
        yt = y_val[:, j].astype(bool)
        if yt.sum() == 0:
            continue
        pr = pred_lock[:, j].astype(bool)
        tp = int((pr & yt).sum())
        fp = int((pr & ~yt).sum())
        fn = int((~pr & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        lab_f1.append((float(freq[j]), f1, int(yt.sum())))
    lab_f1 = np.array(lab_f1)
    q = np.quantile(lab_f1[:, 0], [0.2, 0.4, 0.6, 0.8])
    freq_bins = []
    edges = [0.0, *q.tolist(), 1.0]
    for i in range(5):
        m = (lab_f1[:, 0] >= edges[i]) & (lab_f1[:, 0] < edges[i + 1] if i < 4 else lab_f1[:, 0] <= edges[i + 1])
        if m.sum() == 0:
            continue
        freq_bins.append({
            "bin": i,
            "n_labels": int(m.sum()),
            "freq_mean": round(float(lab_f1[m, 0].mean()), 5),
            "f1_mean": round(float(lab_f1[m, 1].mean()), 6),
            "pos_mean": round(float(lab_f1[m, 2].mean()), 1),
        })

    freq_tr = aa_freq(seqs[idx_train])
    freq_va = aa_freq(seqs[idx_val])
    mu = freq_tr.mean(0)
    d_aa = np.linalg.norm(freq_va - mu, axis=1)
    add_slice("aa_dist>p90", d_aa >= np.quantile(d_aa, 0.90))
    add_slice("aa_dist<=p50", d_aa <= np.quantile(d_aa, 0.50))

    os.makedirs(REL_DIR, exist_ok=True)
    pd.DataFrame(slices).to_csv(CSV_PATH, index=False)
    report = {
        "dup": dup,
        "n_val": int(len(y_val)),
        "n_hit": int(hit.sum()),
        "n_miss": int(miss.sum()),
        "c011a": rec_lock,
        "baselines": {
            "prior": rec_prior,
            "majority": rec_maj,
            "diamond_top1_copy": rec_top1,
            "diamond_top3": rec_bit,
            "kmer": rec_kmer,
            "cnn_lock": rec_cnn,
            "t001a": rec_t001a,
            "deep": rec_deep,
        },
        "label_freq_bins": freq_bins,
        "slices": slices,
        "seconds": round(time.time() - t0, 1),
        "note": "分析性，不写腾讯表",
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告 {REPORT_PATH}")
    print(f"  切片 {CSV_PATH}")
    print(f"  用时 {report['seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
