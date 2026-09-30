"""
CK005c-weight-scan — CK005a 融合权重 / 聚合余量扫描
================================================================
依据:
  - DeepGOPlus (Kulmanov & Hoehndorf, Bioinformatics 2019,
    doi:10.1093/bioinformatics/btz595)：序列相似 + CNN 加权融合。
  - CK005 联合扫描（2026-08-16）定 ALIGN_W=0.4 / SIM_W=0.75，当时是
    default DIAMOND；CK005a 换成 very-sensitive 后未按 100 档 F1 重扫。
  - TCK002 四路权在 F1@0.5 上选；CK004 k-mer 内部权在加入 align 前扫。
方案:
  复用 CK005a very-sensitive e=0.001 缓存，deep 路冻结。
  四组网格，主指标 100 档 Macro F1，提交门槛全局 +0.001：
    1) ALIGN_W ∈ {0.2,0.3,0.4,0.5,0.6} × SIM_W ∈ {0.60..0.90 步 0.05}
    2) 四路局部扰动（t001a/t002/cnn/SIM_W，ALIGN_W=0.4）
    3) k-mer 内部权 10 组（k6 主导 / 等权 / 高低 k 平移）
    4) 比对聚合 top-3/5 × bitscore / bitscore²
用法:
  python code/CK/CK005c-weight-scan.py
  python code/CK/CK005c-weight-scan.py --gen-sub   # 按最优 bit_top5 出提交
输出:
  rel/CK005c_report.json
  submit/submission_CK005c_ensemble.csv（--gen-sub）
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
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
REPORT_PATH = os.path.join(REL_DIR, "CK005c_report.json")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
ALIGN_W = 0.4
SIM_W = 0.75
TOP_K = 3
MAX_ACCEPT = 5
RANDOM_STATE = 42
K_RANGE = [3, 4, 5, 6, 7, 8]
KMER_W_DEFAULT = [0.0, 0.1, 0.2, 0.4, 0.2, 0.1]


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


def load_per_k(y_train):
    per_k = []
    for k in K_RANGE:
        prefix = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{prefix}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_train[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        s = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        per_k.append((lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None])
    return np.stack(per_k, axis=0)


def fuse_kmer(per_k, weights):
    w = np.asarray(weights, np.float32)
    w = w / (w.sum() + 1e-12)
    return (w[:, None, None] * per_k).sum(0)


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


def macro_auc(y, p):
    aucs = []
    for i in range(y.shape[1]):
        yt = y[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, p[:, i]))
    return float(np.mean(aucs))


def eval_mask(ens, y, t, mask):
    return macro_f1(ens[mask], y[mask], t)


def rec_of(name, f1, rec, prec, auc, f1_miss, f1_hit, extra=None):
    d = {
        "name": name,
        "f1_100": round(f1, 6),
        "recall": round(rec, 6),
        "precision": round(prec, 6),
        "auc": round(auc, 6),
        "f1_miss": round(f1_miss, 6),
        "f1_hit": round(f1_hit, 6),
    }
    if extra:
        d.update(extra)
    return d


def build_kmer_test(y_full):
    sim = 0.0
    for k, wt in zip(K_RANGE, KMER_W_DEFAULT):
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


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print("CK005c: CK005a 融合权重 / 聚合余量扫描")

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

    print("[1/5] kmer / deep / align...")
    per_k = load_per_k(y_train)
    sim_kmer0 = fuse_kmer(per_k, KMER_W_DEFAULT)
    sim_bit3 = agg_labels(idx_v, bits_v, y_train, 3)
    sim_bit5 = agg_labels(idx_v, bits_v, y_train, 5)
    sim_bit2_3 = agg_labels(idx_v, bits_v ** 2, y_train, 3)
    sim_bit2_5 = agg_labels(idx_v, bits_v ** 2, y_train, 5)

    vp = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    t001a, t002, cnn = t1["t001a_val"], t2["t002"], vp["cnn"]
    deep0 = w["t001a"] * t001a + w["t002"] * t002 + w["cnn"] * cnn

    def evaluate(ens):
        t = search_t(ens, y_val)
        f1 = macro_f1(ens, y_val, t)
        rec, prec = macro_rp(ens, y_val, t)
        auc = macro_auc(y_val, ens)
        f1m = eval_mask(ens, y_val, t, miss) if miss.any() else 0.0
        f1h = eval_mask(ens, y_val, t, hit) if hit.any() else 0.0
        return f1, rec, prec, auc, f1m, f1h

    ens0 = deep0 + SIM_W * (ALIGN_W * sim_bit3 + (1 - ALIGN_W) * sim_kmer0)
    f1_0, r_0, p_0, a_0, m_0, h_0 = evaluate(ens0)
    baseline = rec_of("baseline", f1_0, r_0, p_0, a_0, m_0, h_0,
                      {"align_w": ALIGN_W, "sim_w": SIM_W,
                       "t001a": w["t001a"], "t002": w["t002"], "cnn": w["cnn"],
                       "agg": "bit_top3"})
    print(f"  基线 F1={f1_0:.4f} R={r_0:.4f} P={p_0:.4f} AUC={a_0:.4f} miss={m_0:.4f}")

    print("[2/5] ALIGN_W × SIM_W...")
    aligns = [0.20, 0.30, 0.40, 0.50, 0.60]
    sims = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    grid_ws = []
    for wa in tqdm(aligns, desc="    ALIGN_W", unit="wa"):
        sim = wa * sim_bit3 + (1 - wa) * sim_kmer0
        for ws in sims:
            ens = deep0 + ws * sim
            f1, rec, prec, auc, f1m, f1h = evaluate(ens)
            grid_ws.append(rec_of(
                f"wa{wa:.2f}_ws{ws:.2f}", f1, rec, prec, auc, f1m, f1h,
                {"align_w": wa, "sim_w": ws, "delta": round(f1 - f1_0, 6)}))
    best_ws = max(grid_ws, key=lambda r: r["f1_100"])
    print(f"  最优 {best_ws['name']} F1={best_ws['f1_100']:.4f} "
          f"Δ={best_ws['delta']:+.4f}")

    print("[3/5] 四路局部扰动...")
    cands = [
        (0.10, 0.20, 0.20, 0.75),
        (0.10, 0.20, 0.25, 0.70),
        (0.10, 0.25, 0.20, 0.70),
        (0.15, 0.20, 0.20, 0.70),
        (0.10, 0.20, 0.15, 0.80),
        (0.10, 0.15, 0.20, 0.80),
        (0.05, 0.20, 0.20, 0.80),
        (0.10, 0.20, 0.30, 0.65),
        (0.10, 0.30, 0.20, 0.65),
        (0.20, 0.20, 0.20, 0.65),
        (0.10, 0.20, 0.10, 0.85),
        (0.10, 0.10, 0.20, 0.85),
        (0.00, 0.20, 0.20, 0.85),
        (0.15, 0.15, 0.20, 0.75),
        (0.05, 0.25, 0.20, 0.75),
        (0.05, 0.15, 0.30, 0.75),
        (0.00, 0.25, 0.25, 0.75),
        (0.10, 0.10, 0.30, 0.75),
        (0.10, 0.30, 0.10, 0.75),
        (0.10, 0.15, 0.15, 0.85),
    ]
    sim = ALIGN_W * sim_bit3 + (1 - ALIGN_W) * sim_kmer0
    grid_four = []
    for a, b, c, ws in tqdm(cands, desc="    四路", unit="w"):
        ens = a * t001a + b * t002 + c * cnn + ws * sim
        f1, rec, prec, auc, f1m, f1h = evaluate(ens)
        grid_four.append(rec_of(
            f"{a:.2f}/{b:.2f}/{c:.2f}/{ws:.2f}", f1, rec, prec, auc, f1m, f1h,
            {"t001a": a, "t002": b, "cnn": c, "sim_w": ws,
             "delta": round(f1 - f1_0, 6)}))
    best_four = max(grid_four, key=lambda r: r["f1_100"])
    print(f"  最优 {best_four['name']} F1={best_four['f1_100']:.4f} "
          f"Δ={best_four['delta']:+.4f}")

    print("[4/5] k-mer 内部权...")
    kmer_cands = [
        ("k6-dom", [0.0, 0.1, 0.2, 0.4, 0.2, 0.1]),
        ("k6-more", [0.0, 0.05, 0.15, 0.5, 0.2, 0.1]),
        ("k6-only", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ("k5k6", [0.0, 0.0, 0.4, 0.6, 0.0, 0.0]),
        ("k4-7eq", [0.0, 0.25, 0.25, 0.25, 0.25, 0.0]),
        ("k3-8eq", [1 / 6] * 6),
        ("shift-lowk", [0.05, 0.15, 0.25, 0.35, 0.15, 0.05]),
        ("shift-highk", [0.0, 0.05, 0.15, 0.35, 0.25, 0.20]),
        ("drop-k3k8", [0.0, 0.15, 0.25, 0.40, 0.20, 0.0]),
        ("k6k7", [0.0, 0.1, 0.15, 0.45, 0.25, 0.05]),
    ]
    grid_kmer = []
    for name, ww in tqdm(kmer_cands, desc="    kmer权", unit="w"):
        sim_k = fuse_kmer(per_k, ww)
        ens = deep0 + SIM_W * (ALIGN_W * sim_bit3 + (1 - ALIGN_W) * sim_k)
        f1, rec, prec, auc, f1m, f1h = evaluate(ens)
        wn = [round(float(x), 3) for x in (np.array(ww) / np.sum(ww))]
        grid_kmer.append(rec_of(
            name, f1, rec, prec, auc, f1m, f1h,
            {"w": wn, "delta": round(f1 - f1_0, 6)}))
    best_kmer = max(grid_kmer, key=lambda r: r["f1_100"])
    print(f"  最优 {best_kmer['name']} F1={best_kmer['f1_100']:.4f} "
          f"Δ={best_kmer['delta']:+.4f}")

    print("[5/5] 比对聚合...")
    grid_agg = []
    for name, ab in tqdm([
        ("bit_top3", sim_bit3),
        ("bit_top5", sim_bit5),
        ("bit2_top3", sim_bit2_3),
        ("bit2_top5", sim_bit2_5),
    ], desc="    聚合", unit="var"):
        ens = deep0 + SIM_W * (ALIGN_W * ab + (1 - ALIGN_W) * sim_kmer0)
        f1, rec, prec, auc, f1m, f1h = evaluate(ens)
        grid_agg.append(rec_of(
            name, f1, rec, prec, auc, f1m, f1h,
            {"delta": round(f1 - f1_0, 6)}))
    best_agg = max(grid_agg, key=lambda r: r["f1_100"])
    print(f"  最优 {best_agg['name']} F1={best_agg['f1_100']:.4f} "
          f"Δ={best_agg['delta']:+.4f}")

    all_rows = grid_ws + grid_four + grid_kmer + grid_agg
    best = max(all_rows, key=lambda r: r["f1_100"])
    gain = best["f1_100"] - f1_0
    useful = bool(gain >= 0.001)
    print(f"  全局最优 {best['name']} F1={best['f1_100']:.4f} Δ={gain:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "baseline": baseline,
        "best": best,
        "gain_all_f1": round(gain, 6),
        "useful": useful,
        "best_by_group": {
            "align_sim": best_ws,
            "four_way": best_four,
            "kmer_w": best_kmer,
            "align_agg": best_agg,
        },
        "grid_align_sim": grid_ws,
        "grid_four": grid_four,
        "grid_kmer": grid_kmer,
        "grid_agg": grid_agg,
        "seconds": round(time.time() - t0, 1),
    }

    if gen_sub:
        print("[sub] bit_top5 出提交...")
        test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        tsv_t = os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
        if not os.path.exists(tsv_t):
            raise FileNotFoundError("需要 CK005a test tsv")
        full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
        idx_t, bits_t = parse_aln(tsv_t, test_df["protein_id"].values, full_map)
        sim_bit5_t = agg_labels(idx_t, bits_t, y_full, 5)
        sim_kmer_t = build_kmer_test(y_full)
        sim_t = ALIGN_W * sim_bit5_t + (1 - ALIGN_W) * sim_kmer_t
        tp = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))
        t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
        t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))
        ens_t = (w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"]
                 + w["cnn"] * tp["cnn"] + SIM_W * sim_t)
        ens_val = deep0 + SIM_W * (ALIGN_W * sim_bit5 + (1 - ALIGN_W) * sim_kmer0)
        best_t = search_t(ens_val, y_val)
        y_pred = (ens_t >= best_t[None, :]).astype(np.int8)
        os.makedirs(os.path.join(PROJECT_ROOT, "submit"), exist_ok=True)
        sub_path = os.path.join(PROJECT_ROOT, "submit", "submission_CK005c_ensemble.csv")
        sub = pd.concat([test_df[["protein_id"]],
                         pd.DataFrame(y_pred, columns=label_cols)], axis=1)
        sub.to_csv(sub_path, index=False)
        report["submission"] = sub_path
        report["submission_config"] = "bit_top5 ALIGN_W=0.4 SIM_W=0.75"
        print(f"  提交: {sub_path} ({len(sub)} 行)")

    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
