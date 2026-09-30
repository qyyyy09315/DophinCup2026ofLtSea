"""
I006b-oracle-rcut — oracle RCut 上界分解
================================================================
依据:
  - Yang Y, SIGIR 2001, doi:10.1145/383952.383975
    SCut（逐标签）vs RCut（每样本 top-k）。
  - I006：真 k 的 RCut-oracle 全局 F1 0.5245 / missF1 0.1986；
    实务 k 估不准（线性 MAE=15，邻居 MAE=17）。只报了单点上界。
方案:
  复用 CK005a 四路分数。k 一律用验证集真基数（oracle，非提交）。
  分解四问，主指标仍 100 档口径的 Macro F1（RCut 本身不再搜阈值）：
    1) 复现 rcut_oracle / fill_oracle / union / clip，对照 SCut
    2) 只改 miss / 只改 hit / 全量，看增益来源
    3) k 误差曲线：真 k+Δ（Δ∈{-8..+8}）与真 k×scale
    4) 排序诊断：top-k 捕获率、真标签落入 top-k 的比例；
       按长度 / 真基数分层
用法:
  python code/I/I006b-oracle-rcut.py
输出:
  rel/I006b_report.json
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
REPORT_PATH = os.path.join(REL_DIR, "I006b_report.json")
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


def rcut(order, k_hat, n_labels=N_LABELS):
    n = order.shape[0]
    k_hat = np.clip(np.rint(k_hat), 1, n_labels).astype(np.int32)
    pred = np.zeros((n, n_labels), np.int8)
    for i in range(n):
        pred[i, order[i, :k_hat[i]]] = 1
    return pred


def fill_from_scut(order, scut, k_hat):
    pred = scut.copy()
    k_hat = np.clip(np.rint(k_hat), 1, pred.shape[1]).astype(np.int32)
    for i in range(pred.shape[0]):
        have = int(pred[i].sum())
        need = int(k_hat[i])
        if have >= need:
            continue
        for j in order[i]:
            if pred[i, j] == 0:
                pred[i, j] = 1
                have += 1
                if have >= need:
                    break
    return pred


def clip_to_k(order, scut, k_hat):
    pred = scut.copy()
    k_hat = np.clip(np.rint(k_hat), 1, pred.shape[1]).astype(np.int32)
    for i in range(pred.shape[0]):
        need = int(k_hat[i])
        if int(pred[i].sum()) <= need:
            continue
        pred[i] = 0
        pred[i, order[i, :need]] = 1
    return pred


def apply_mask(base, overlay, mask):
    pred = base.copy()
    pred[mask] = overlay[mask]
    return pred


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


def bin_f1(pred, y, mask):
    if mask.sum() == 0:
        return None
    return round(macro_f1(pred[mask], y[mask]), 6)


def main():
    t0 = time.time()
    print("=" * 60)
    print("I006b: oracle RCut 上界分解（真 k，非提交）")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    val_len = train_df["sequence"].str.len().values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    card = y_val.sum(1).astype(np.float32)
    k_true = np.clip(np.rint(card), 1, N_LABELS).astype(np.int32)

    tsv = os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    if not os.path.exists(tsv):
        raise FileNotFoundError("需要 CK005a very-sensitive val tsv")
    idx_v, bits_v = parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    print(f"  hit {int(hit.sum())}/{len(hit)} miss {int(miss.sum())}  "
          f"真k mean={card.mean():.1f} miss真k={card[miss].mean():.1f}")

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
    order = np.argsort(-ens, axis=1)
    print(f"  SCut card={scut.sum(1).mean():.1f}  miss card={scut[miss].sum(1).mean():.1f}")

    print("[1/4] oracle 变体...")
    pred_rcut = rcut(order, k_true)
    pred_fill = fill_from_scut(order, scut, k_true)
    pred_clip = clip_to_k(order, scut, k_true)
    pred_union = np.maximum(scut, pred_rcut)
    variants = [
        pack("scut", scut, y_val, hit, miss),
        pack("rcut_oracle", pred_rcut, y_val, hit, miss),
        pack("fill_oracle", pred_fill, y_val, hit, miss),
        pack("clip_oracle", pred_clip, y_val, hit, miss),
        pack("union_oracle", pred_union, y_val, hit, miss),
    ]
    for r in variants:
        print(f"  {r['name']:14s} F1={r['f1_100']:.4f} miss={r['f1_miss']:.4f} "
              f"hit={r['f1_hit']:.4f} card={r['pred_card']:.1f}")

    print("[2/4] hit / miss 来源...")
    source = [
        pack("rcut_oracle_miss", apply_mask(scut, pred_rcut, miss), y_val, hit, miss),
        pack("rcut_oracle_hit", apply_mask(scut, pred_rcut, hit), y_val, hit, miss),
        pack("rcut_oracle_all", pred_rcut, y_val, hit, miss),
        pack("fill_oracle_miss", apply_mask(scut, pred_fill, miss), y_val, hit, miss),
        pack("fill_oracle_hit", apply_mask(scut, pred_fill, hit), y_val, hit, miss),
    ]
    for r in source:
        print(f"  {r['name']:18s} F1={r['f1_100']:.4f} miss={r['f1_miss']:.4f} "
              f"hit={r['f1_hit']:.4f}")

    print("[3/4] k 误差曲线...")
    k_delta = []
    for d in tqdm([-8, -4, -2, -1, 0, 1, 2, 4, 8], desc="    Δk", unit="d"):
        pred = rcut(order, k_true + d)
        k_delta.append(pack(
            f"rcut_k{d:+d}", pred, y_val, hit, miss,
            {"delta": d, "mae_vs_true": abs(d)}))
    k_scale = []
    for s in tqdm([0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5], desc="    ×k", unit="s"):
        pred = rcut(order, np.maximum(1, np.rint(k_true * s)))
        k_scale.append(pack(
            f"rcut_x{s:g}", pred, y_val, hit, miss, {"scale": s}))
    print("  Δk:")
    for r in k_delta:
        print(f"    {r['name']:12s} F1={r['f1_100']:.4f} miss={r['f1_miss']:.4f}")
    print("  ×k:")
    for r in k_scale:
        print(f"    {r['name']:12s} F1={r['f1_100']:.4f} miss={r['f1_miss']:.4f}")

    print("[4/4] 排序诊断 + 分层...")
    capture = np.zeros(len(idx_val), np.float32)
    n_true_in_topk = np.zeros(len(idx_val), np.int32)
    for i in tqdm(range(len(idx_val)), desc="    捕获率", unit="row"):
        k = int(k_true[i])
        top = order[i, :k]
        yt = y_val[i].astype(bool)
        nt = int(yt.sum())
        if nt == 0:
            capture[i] = 1.0
            continue
        hit_pos = int(yt[top].sum())
        n_true_in_topk[i] = hit_pos
        capture[i] = hit_pos / nt
    ranking = {
        "mean_capture_at_true_k": round(float(capture.mean()), 4),
        "miss_capture": round(float(capture[miss].mean()), 4),
        "hit_capture": round(float(capture[hit].mean()), 4),
        "mean_true_in_topk": round(float(n_true_in_topk.mean()), 3),
        "gap_to_perfect": round(1.0 - float(capture.mean()), 4),
    }
    print(f"  top-k 捕获率 {ranking['mean_capture_at_true_k']:.3f} "
          f"(hit {ranking['hit_capture']:.3f} / miss {ranking['miss_capture']:.3f})")

    len_bins = [
        ("L<=128", val_len <= 128),
        ("L129-256", (val_len > 128) & (val_len <= 256)),
        ("L257-400", (val_len > 256) & (val_len <= 400)),
        ("L401-1500", (val_len > 400) & (val_len <= 1500)),
        ("L>1500", val_len > 1500),
    ]
    card_bins = [
        ("k1-5", card <= 5),
        ("k6-10", (card > 5) & (card <= 10)),
        ("k11-20", (card > 10) & (card <= 20)),
        ("k21-40", (card > 20) & (card <= 40)),
        ("k>40", card > 40),
    ]

    def strata(pred):
        out = {}
        for name, m in len_bins + card_bins:
            out[name] = {
                "n": int(m.sum()),
                "f1_scut": bin_f1(scut, y_val, m),
                "f1_oracle": bin_f1(pred, y_val, m),
                "capture": round(float(capture[m].mean()), 4) if m.any() else None,
            }
        return out

    by_bin = strata(pred_rcut)
    for name, rec in by_bin.items():
        print(f"  {name:12s} n={rec['n']:5d} SCut={rec['f1_scut']} "
              f"oracle={rec['f1_oracle']} cap={rec['capture']}")

    oracle = variants[1]
    base = variants[0]
    report = {
        "baseline_scut": base,
        "oracle_rcut": oracle,
        "gain_all_f1": round(oracle["f1_100"] - base["f1_100"], 6),
        "gain_miss_f1": round(oracle["f1_miss"] - base["f1_miss"], 6),
        "auc_ens": round(auc, 6),
        "variants": variants,
        "source": source,
        "k_delta": k_delta,
        "k_scale": k_scale,
        "ranking": ranking,
        "strata": by_bin,
        "useful": False,
        "note": "oracle 用验证真 k，非提交；分解 I006 单点上界",
        "seconds": round(time.time() - t0, 1),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
