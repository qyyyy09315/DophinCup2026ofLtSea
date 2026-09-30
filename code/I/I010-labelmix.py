"""
I010-labelmix — 逐标签 deep/sim 半折混权
================================================================
依据:
  - Super Learner: van der Laan, Polley & Hubbard, Stat Appl Genet Mol Biol
    2007, doi:10.2202/1544-6115.1309。交叉验证选基学习器凸组合，避免
    全验证集搜权过拟合。
  - TC003e：CNN+Transformer 逐标签权在全 val 上搜，本地 0.2846，提交风险高。
  - CK005c：全局 ALIGN_W/SIM_W 扫描 +0.0005，平台未超 CK005a。
自己的改法:
  CK005a 是全局 ens = deep + 0.75·sim。这里冻结 deep 与 sim 两路，
  每标签只搜 SIM_W_j。权在半折上按 100 档 F1 选，另一折评估（对调平均）。
  尾标签用频率三分位共享一个权（收缩）。对照：全局 0.75、全 val 搜权上界。
  主指标全局 Macro F1，门槛 +0.001。不改 CNN / 检索。
用法:
  python code/I/I010-labelmix.py
输出:
  rel/I010_report.json
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
REPORT_PATH = os.path.join(REL_DIR, "I010_report.json")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
ALIGN_W = 0.4
SIM_W = 0.75
TOP_K = 3
MAX_ACCEPT = 5
RANDOM_STATE = 42
SIM_GRID = np.array([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50], dtype=np.float32)
N_BINS = 3


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


def best_t_one(score, yt):
    if yt.sum() == 0:
        return 0.5
    cands = np.linspace(0.01, 0.99, 100)
    pr = score[:, None] >= cands[None, :]
    ytm = yt[:, None].astype(bool)
    tp = (pr & ytm).sum(0)
    fp = (pr & ~ytm).sum(0)
    fn = (~pr & ytm).sum(0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(cands[f1t.argmax()])


def f1_at_t(score, yt, t):
    if yt.sum() == 0:
        return 0.0
    p = score >= t
    yt = yt.astype(bool)
    tp = int((p & yt).sum())
    fp = int((p & ~yt).sum())
    fn = int((~p & yt).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


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


def eval_ens(ens, y, hit, miss, name, extra=None):
    t = search_t(ens, y)
    pred = (ens >= t).astype(np.int8)
    rec = {
        "name": name,
        "f1_100": round(macro_f1(pred, y), 6),
        "recall": round(macro_rp(pred, y)[0], 6),
        "precision": round(macro_rp(pred, y)[1], 6),
        "auc": round(macro_auc(y, ens), 6),
        "f1_miss": round(macro_f1(pred[miss], y[miss]), 6) if miss.any() else 0.0,
        "f1_hit": round(macro_f1(pred[hit], y[hit]), 6) if hit.any() else 0.0,
        "pred_card": round(float(pred.sum(1).mean()), 3),
    }
    if extra:
        rec.update(extra)
    return rec, t, pred


def pick_sw(deep, sim, y, rows):
    """在 rows 上按 100 档 F1 为每个标签选 SIM_W。"""
    sw = np.full(N_LABELS, SIM_W, dtype=np.float32)
    for i in tqdm(range(N_LABELS), desc="    选权", unit="lab", leave=False):
        yt = y[rows, i]
        if yt.sum() == 0:
            continue
        best_f, best_w = -1.0, SIM_W
        d, s = deep[rows, i], sim[rows, i]
        for w in SIM_GRID:
            score = d + w * s
            t = best_t_one(score, yt)
            f = f1_at_t(score, yt, t)
            if f > best_f:
                best_f, best_w = f, float(w)
        sw[i] = best_w
    return sw


def apply_sw(deep, sim, sw):
    return deep + sw[None, :] * sim


def freq_bins(pos, n_bins=N_BINS):
    """按训练正例数三分位。"""
    active = pos > 0
    qs = np.quantile(pos[active], np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(pos, qs, right=True)
    bins[~active] = 0
    return bins.astype(np.int32)


def pick_sw_binned(deep, sim, y, rows, bins):
    sw = np.full(N_LABELS, SIM_W, dtype=np.float32)
    for b in range(int(bins.max()) + 1):
        cols = np.where(bins == b)[0]
        if cols.size == 0:
            continue
        best_f, best_w = -1.0, SIM_W
        for w in SIM_GRID:
            f1s = []
            for i in cols:
                yt = y[rows, i]
                if yt.sum() == 0:
                    continue
                score = deep[rows, i] + w * sim[rows, i]
                t = best_t_one(score, yt)
                f1s.append(f1_at_t(score, yt, t))
            f = float(np.mean(f1s)) if f1s else -1.0
            if f > best_f:
                best_f, best_w = f, float(w)
        sw[cols] = best_w
    return sw


def oof_mix(deep, sim, y, picker, extra=None):
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]
    oof = np.zeros_like(deep, dtype=np.float32)
    sw_mean = np.zeros(N_LABELS, dtype=np.float32)
    for i, hold in enumerate(folds):
        fit = folds[1 - i]
        sw = picker(deep, sim, y, fit)
        oof[hold] = apply_sw(deep[hold], sim[hold], sw)
        sw_mean += sw
    sw_mean /= 2.0
    extra = extra or {}
    extra["sw_mean"] = round(float(sw_mean.mean()), 4)
    extra["sw_ne_075"] = int((np.abs(sw_mean - SIM_W) > 1e-6).sum())
    return oof, sw_mean, extra


def main():
    t0 = time.time()
    print("=" * 60)
    print("I010: 逐标签 deep/sim 半折混权")

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

    rows = []
    print("[1/4] 全局 SIM_W=0.75（CK005a）...")
    rec, _, _ = eval_ens(deep + SIM_W * sim, y_val, hit, miss, "global_0.75")
    rows.append(rec)
    print(f"  {rec['name']:24s} F1={rec['f1_100']:.4f} miss={rec['f1_miss']:.4f}")

    print("[2/4] 全 val 逐标签搜权（上界，易过拟合）...")
    sw_full = pick_sw(deep, sim, y_val, np.arange(len(y_val)))
    rec, _, _ = eval_ens(
        apply_sw(deep, sim, sw_full), y_val, hit, miss, "insample_perlabel",
        {"sw_mean": round(float(sw_full.mean()), 4),
         "sw_ne_075": int((np.abs(sw_full - SIM_W) > 1e-6).sum())})
    rows.append(rec)
    print(f"  {rec['name']:24s} F1={rec['f1_100']:.4f} 改权标签 {rec['sw_ne_075']}")

    print("[3/4] 半折逐标签（OOF）...")
    oof, sw_oof, extra = oof_mix(deep, sim, y_val, pick_sw)
    rec, _, _ = eval_ens(oof, y_val, hit, miss, "oof_perlabel", extra)
    rows.append(rec)
    print(f"  {rec['name']:24s} F1={rec['f1_100']:.4f} 改权标签 {rec['sw_ne_075']}")

    print("[4/4] 半折频率三分位共享权...")
    pos = y_train.sum(0).astype(np.float64)
    bins = freq_bins(pos)
    def picker_bin(d, s, y, rows_fit, _bins=bins):
        return pick_sw_binned(d, s, y, rows_fit, _bins)
    oof_b, sw_b, extra_b = oof_mix(deep, sim, y_val, picker_bin)
    extra_b["bin_w"] = [round(float(sw_b[bins == b].mean()), 4)
                        for b in range(int(bins.max()) + 1) if (bins == b).any()]
    rec, _, _ = eval_ens(oof_b, y_val, hit, miss, "oof_freqbin", extra_b)
    rows.append(rec)
    print(f"  {rec['name']:24s} F1={rec['f1_100']:.4f} bin_w={extra_b['bin_w']}")

    base = rows[0]["f1_100"]
    honest = [r for r in rows if r["name"].startswith("oof_")]
    best = max(honest, key=lambda r: r["f1_100"])
    gain = best["f1_100"] - base
    useful = bool(gain >= 0.001)
    print(f"  诚实最优 {best['name']} F1={best['f1_100']:.4f} Δ={gain:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "baseline": rows[0],
        "best_oof": best,
        "gain_all_f1": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "seconds": round(time.time() - t0, 1),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
