"""
CK005j-gated-clip — 门控后再截 hit 过预测
================================================================
依据:
  - Yang Y, SIGIR 2001, doi:10.1145/383952.383975（SCut vs 截断）
  - CK005h：RCut-oracle 0.5245 几乎全在 hit 截断；hit 里仅 57% 真的多报，
    一刀切会误伤另外 43%。
  - CK005e/f/i：估 k / 分位截断均未过 +0.001。
方案:
  复用 CK005a 分数 + SCut。不回归 k。
  目标 g=1[SCut个数 > 真基数]（只在 hit 上有意义）。
  特征：SCut 基数、ens 和、log 长度、n(score≥0.4/0.5/0.6)、行 p70/p80 分位、
        行 max、hit 标记。
  半折 LogisticRegression 学 P(g=1)，另一折评估（再对调平均）。
  对 P≥τ 的 hit 行做截断：
    clip_ratio：保留 SCut 中分数最高的 round(n_scut × r) 个
    clip_p：SCut ∧ (score ≥ 该行 q_p)
  τ ∈ {0.5,0.6,0.7,0.8} × r ∈ {0.7,0.8,0.85} / p ∈ {0.70,0.80}
  对照：无门控、全 hit 截（应差于门控若门控有用）。
  主指标全局 F1，门槛 +0.001。
用法:
  python code/CK/CK005j-gated-clip.py
  python code/CK/CK005j-gated-clip.py --gen-sub
输出:
  rel/CK005j_report.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
REPORT_PATH = os.path.join(REL_DIR, "CK005j_report.json")
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


def pack(name, pred, y_val, hit, miss, extra=None):
    f_all = macro_f1(pred, y_val)
    r, p = macro_rp(pred, y_val)
    rec = {
        "name": name,
        "f1_100": round(f_all, 6),
        "recall": round(r, 6),
        "precision": round(p, 6),
        "f1_miss": round(macro_f1(pred[miss], y_val[miss]), 6),
        "f1_hit": round(macro_f1(pred[hit], y_val[hit]), 6),
        "pred_card": round(float(pred.sum(1).mean()), 3),
        "miss_pred_card": round(float(pred[miss].sum(1).mean()), 3),
    }
    if extra:
        rec.update(extra)
    return rec


def make_features(ens, scut, val_len, hit):
    scard = scut.sum(1).astype(np.float32)
    p70 = np.quantile(ens, 0.70, axis=1)
    p80 = np.quantile(ens, 0.80, axis=1)
    return np.column_stack([
        scard,
        ens.sum(1),
        np.log1p(val_len),
        (ens >= 0.4).sum(1),
        (ens >= 0.5).sum(1),
        (ens >= 0.6).sum(1),
        p70, p80, ens.max(1),
        hit.astype(np.float32),
    ]).astype(np.float32)


def oof_gate(X, y_gate, hit):
    """半折 Logistic，只在 hit 上拟合；输出全量 P(g=1)。"""
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_gate)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]
    proba = np.zeros(n, np.float32)
    aucs = []
    for i, hold in enumerate(folds):
        fit = folds[1 - i]
        fit_h = fit[hit[fit]]
        if fit_h.size < 50 or y_gate[fit_h].sum() < 10 or (1 - y_gate[fit_h]).sum() < 10:
            proba[hold] = 0.5
            continue
        clf = LogisticRegression(max_iter=400, class_weight="balanced")
        clf.fit(X[fit_h], y_gate[fit_h])
        proba[hold] = clf.predict_proba(X[hold])[:, 1]
        hold_h = hold[hit[hold]]
        if hold_h.size and y_gate[hold_h].min() != y_gate[hold_h].max():
            aucs.append(roc_auc_score(y_gate[hold_h], proba[hold_h]))
    return proba, (float(np.mean(aucs)) if aucs else 0.5)


def clip_ratio_rows(ens, scut, mask, ratio):
    pred = scut.copy()
    idx = np.where(mask)[0]
    order = np.argsort(-ens, axis=1)
    scard = scut.sum(1)
    for i in idx:
        keep = max(1, int(round(float(scard[i]) * ratio)))
        have = int(scard[i])
        if have <= keep:
            continue
        pred[i] = 0
        # keep top-keep among currently positive if possible
        pos = np.flatnonzero(scut[i])
        ranked = [j for j in order[i] if scut[i, j]]
        pred[i, ranked[:keep]] = 1
    return pred


def clip_p_rows(ens, scut, mask, p):
    pred = scut.copy()
    thr = np.quantile(ens, p, axis=1)
    q = ens >= thr[:, None]
    inter = (scut & q).astype(np.int8)
    empty = inter.sum(1) == 0
    inter[empty] = scut[empty]
    pred[mask] = inter[mask]
    return pred


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print("CK005j: 门控后再截 hit 过预测")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    val_len = train_df["sequence"].str.len().values[idx_val].astype(np.float32)
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    card = y_val.sum(1).astype(np.float32)

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
    ens = (w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"]
           + w["cnn"] * vp["cnn"] + SIM_W * sim)
    auc = macro_auc(y_val, ens)

    t = search_t(ens, y_val)
    scut = (ens >= t).astype(np.int8)
    scard = scut.sum(1).astype(np.float32)
    y_gate = (scard > card).astype(np.int8)
    print(f"  SCut card={scard.mean():.1f}  hit 多报率={y_gate[hit].mean():.3f}  "
          f"miss 多报率={y_gate[miss].mean():.3f}")

    X = make_features(ens, scut, val_len, hit)
    gprob, gauc = oof_gate(X, y_gate, hit)
    print(f"  门控 OOF AUC(hit)={gauc:.3f}  Pmean hit={gprob[hit].mean():.3f}")

    rows = [pack("scut", scut, y_val, hit, miss)]
    print(f"  SCut F1={rows[0]['f1_100']:.4f}")

    print("[1/2] 无门控对照（全 hit 截）...")
    for r in (0.70, 0.80, 0.85):
        pred = clip_ratio_rows(ens, scut, hit, r)
        rec = pack(f"nogate_ratio{r:.2f}", pred, y_val, hit, miss,
                   {"delta": round(macro_f1(pred, y_val) - rows[0]["f1_100"], 6)})
        rows.append(rec)
        print(f"  {rec['name']:24s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f} "
              f"card={rec['pred_card']:.1f}")
    for p in (0.70, 0.80):
        pred = clip_p_rows(ens, scut, hit, p)
        rec = pack(f"nogate_p{p:.2f}", pred, y_val, hit, miss,
                   {"delta": round(macro_f1(pred, y_val) - rows[0]["f1_100"], 6)})
        rows.append(rec)
        print(f"  {rec['name']:24s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f} "
              f"card={rec['pred_card']:.1f}")

    print("[2/2] 门控网格...")
    for tau in tqdm([0.50, 0.60, 0.70, 0.80], desc="    τ", unit="t"):
        gated = hit & (gprob >= tau)
        print(f"  τ={tau:.2f} 触发 {int(gated.sum())}/{int(hit.sum())} hit "
              f"({gated.sum()/max(hit.sum(),1):.2f})")
        for r in (0.70, 0.80, 0.85):
            pred = clip_ratio_rows(ens, scut, gated, r)
            rec = pack(
                f"gate_t{tau:.2f}_r{r:.2f}", pred, y_val, hit, miss,
                {"tau": tau, "ratio": r, "n_clip": int(gated.sum()),
                 "delta": round(macro_f1(pred, y_val) - rows[0]["f1_100"], 6)})
            rows.append(rec)
            print(f"    {rec['name']:26s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f} "
                  f"card={rec['pred_card']:.1f}")
        for p in (0.70, 0.80):
            pred = clip_p_rows(ens, scut, gated, p)
            rec = pack(
                f"gate_t{tau:.2f}_p{p:.2f}", pred, y_val, hit, miss,
                {"tau": tau, "p": p, "n_clip": int(gated.sum()),
                 "delta": round(macro_f1(pred, y_val) - rows[0]["f1_100"], 6)})
            rows.append(rec)
            print(f"    {rec['name']:26s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f} "
                  f"card={rec['pred_card']:.1f}")

    base = rows[0]
    ranked = [r for r in rows if r["name"] != "scut"]
    best = max(ranked, key=lambda r: r["f1_100"])
    gain = best["f1_100"] - base["f1_100"]
    useful = bool(gain >= 0.001)
    print(f"  最优 {best['name']} F1={best['f1_100']:.4f} Δ={gain:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "baseline_scut": base,
        "best": best,
        "gain_all_f1": round(gain, 6),
        "useful": useful,
        "auc_ens": round(auc, 6),
        "gate_auc_hit": round(gauc, 6),
        "hit_overpred_rate": round(float(y_gate[hit].mean()), 4),
        "grid": rows,
        "seconds": round(time.time() - t0, 1),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    if gen_sub and not useful:
        print("  --gen-sub 忽略（未过 +0.001）")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
