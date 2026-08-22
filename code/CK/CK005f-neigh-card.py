"""
CK005f-neigh-card — hit 用 DIAMOND 邻居标签数当每样本 k
================================================================
依据:
  - Yang Y, SIGIR 2001, doi:10.1145/383952.383975
    SCut（逐标签）vs RCut（每样本 top-k）。CK005e oracle RCut 全局 0.5245，
    说明排序够、缺的是 k。线性估 k 失败（hold MAE=15）。
  - Ghawi & Pfeffer, iiWAS 2021, doi:10.1145/3487664.3487702
    RCut+PCut 混合阈值。
  - RAPT (Jayawardena et al., 2026, arXiv:2605.16535)
    检索增强的实例级事后阈值，打实例相关基数。
  - 实测：hit 的 DIAMOND top-3 邻居标签数 mean=28.2 vs 真 26.0（corr=0.35）；
    miss 无邻居，保持 SCut（长度先验会把 miss k 补到 23，真只有 16）。
方案:
  复用 CK005a 四路分数 + very-sensitive e=0.001 比对。
  对 hit：k_hat = top-3 邻居标签数的 bitscore 加权平均。
  网格：rcut_hit / fill_hit / union(SCut, RCut_hit) × scale∈{0.8,1.0,1.2}。
  miss 行一律保持 SCut。主指标：全局 F1 不掉。
用法:
  python code/CK/CK005f-neigh-card.py
  python code/CK/CK005f-neigh-card.py --gen-sub
输出:
  rel/CK005f_report.json
  submit/submission_CK005f_ensemble.csv（仅 --gen-sub 且有效）
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
REPORT_PATH = os.path.join(REL_DIR, "CK005f_report.json")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
ALIGN_W = 0.4
SIM_W = 0.75
TOP_K = 3
MAX_ACCEPT = 5
RANDOM_STATE = 42
EVALUE = 0.001


def parse_aln(tsv, query_ids, id_map, evalue=EVALUE):
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


def neigh_card(idx, w, y_space, top_k=TOP_K):
    """hit 行：top-k 邻居标签数的 bitscore 加权平均；miss 为 0。"""
    n = idx.shape[0]
    cards = np.zeros(n, np.float32)
    for i in range(n):
        ww, cc = [], []
        for j in range(top_k):
            if idx[i, j] < 0:
                continue
            ww.append(max(float(w[i, j]), 1e-7))
            cc.append(float(y_space[idx[i, j]].sum()))
        if ww:
            cards[i] = float(np.average(cc, weights=ww))
    return cards


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


def rcut_rows(ens, k_hat, mask):
    pred = np.zeros(ens.shape, np.int8)
    k_hat = np.clip(np.rint(k_hat), 1, ens.shape[1]).astype(np.int32)
    order = np.argsort(-ens, axis=1)
    idx = np.where(mask)[0]
    for i in tqdm(idx, desc="    RCut", unit="row", leave=False):
        pred[i, order[i, :k_hat[i]]] = 1
    return pred


def fill_rows(ens, scut, k_hat, mask):
    pred = scut.copy()
    k_hat = np.clip(np.rint(k_hat), 1, ens.shape[1]).astype(np.int32)
    order = np.argsort(-ens, axis=1)
    idx = np.where(mask)[0]
    for i in tqdm(idx, desc="    补满", unit="row", leave=False):
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


def clip_rows(ens, scut, k_hat, mask):
    """hit 上若 SCut 超过 k_hat 则按分数截到 k_hat。"""
    pred = scut.copy()
    k_hat = np.clip(np.rint(k_hat), 1, ens.shape[1]).astype(np.int32)
    order = np.argsort(-ens, axis=1)
    idx = np.where(mask)[0]
    for i in tqdm(idx, desc="    截断", unit="row", leave=False):
        need = int(k_hat[i])
        if int(pred[i].sum()) <= need:
            continue
        pred[i] = 0
        pred[i, order[i, :need]] = 1
    return pred


def union_rows(ens, scut, k_hat, mask):
    r = rcut_rows(ens, k_hat, mask)
    pred = scut.copy()
    pred[mask] = np.maximum(scut[mask], r[mask])
    return pred


def apply_hit(ens, scut, k_hat, hit, mode):
    if mode == "rcut":
        pred = scut.copy()
        pred[hit] = rcut_rows(ens, k_hat, hit)[hit]
        return pred
    if mode == "fill":
        return fill_rows(ens, scut, k_hat, hit)
    if mode == "clip":
        return clip_rows(ens, scut, k_hat, hit)
    if mode == "union":
        return union_rows(ens, scut, k_hat, hit)
    raise ValueError(mode)


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print("CK005f: hit 邻居标签数当 k（miss 保持 SCut）")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
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

    t = search_t(ens, y_val)
    scut = (ens >= t).astype(np.int8)
    dcard = neigh_card(idx_v, bits_v, y_train)
    print(f"  真基数 {card.mean():.1f}  SCut {scut.sum(1).mean():.1f}  "
          f"hit邻域k {dcard[hit].mean():.1f}（真 {card[hit].mean():.1f}）")
    print(f"  hit 邻域k corr={np.corrcoef(card[hit], dcard[hit])[0, 1]:.3f}  "
          f"MAE={np.mean(np.abs(card[hit] - dcard[hit])):.2f}")

    def pack(name, pred):
        f_all = macro_f1(pred, y_val)
        r, p = macro_rp(pred, y_val)
        f_miss = macro_f1(pred[miss], y_val[miss])
        f_hit = macro_f1(pred[hit], y_val[hit])
        return {
            "name": name,
            "f1_100": round(f_all, 6), "recall": round(r, 6), "precision": round(p, 6),
            "f1_miss": round(f_miss, 6), "f1_hit": round(f_hit, 6),
            "pred_card": round(float(pred.sum(1).mean()), 3),
            "hit_pred_card": round(float(pred[hit].sum(1).mean()), 3),
            "miss_pred_card": round(float(pred[miss].sum(1).mean()), 3),
        }

    rows = [pack("scut", scut)]
    print(f"  SCut: F1={rows[0]['f1_100']:.4f} hitF1={rows[0]['f1_hit']:.4f} "
          f"missF1={rows[0]['f1_miss']:.4f}")

    print("[网格 mode × scale，只改 hit]")
    for mode in ["rcut", "fill", "clip", "union"]:
        for scale in [0.8, 1.0, 1.2]:
            kh = dcard * scale
            pred = apply_hit(ens, scut, kh, hit, mode)
            rec = pack(f"{mode}_x{scale:g}", pred)
            rec["mode"] = mode
            rec["scale"] = scale
            rows.append(rec)
            print(f"  {rec['name']}: F1={rec['f1_100']:.4f} hitF1={rec['f1_hit']:.4f} "
                  f"Δ={rec['f1_100'] - rows[0]['f1_100']:+.4f}", flush=True)

    base = rows[0]
    ranked = [r for r in rows[1:] if r["f1_100"] >= base["f1_100"] - 0.001]
    if ranked:
        best = max(ranked, key=lambda r: (r["f1_100"], r["f1_hit"]))
    else:
        best = max(rows[1:], key=lambda r: r["f1_100"])
    gain_all = best["f1_100"] - base["f1_100"]
    gain_hit = best["f1_hit"] - base["f1_hit"]
    useful = gain_all >= 0.001 or (gain_hit >= 0.005 and gain_all >= -0.001)
    print(f"  最优 {best['name']} ΔF1={gain_all:+.4f} Δhit={gain_hit:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    auc = macro_auc(y_val, ens)
    report = {
        "baseline_scut": rows[0],
        "dcard_hit_corr": round(float(np.corrcoef(card[hit], dcard[hit])[0, 1]), 4),
        "dcard_hit_mae": round(float(np.mean(np.abs(card[hit] - dcard[hit]))), 4),
        "grid": rows, "best": best,
        "gain_all": round(gain_all, 6), "gain_hit": round(gain_hit, 6),
        "useful": useful, "auc": round(auc, 6),
        "seconds": round(time.time() - t0),
    }

    if not (gen_sub and useful):
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  报告: {REPORT_PATH}")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    print("[test] 最优配置生成提交...")
    tsv_t = os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
    if not os.path.exists(tsv_t):
        raise FileNotFoundError("需要 CK005a test very-sensitive tsv")
    full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
    idx_t, bits_t = parse_aln(tsv_t, test_df["protein_id"].values, full_map)
    hit_t = idx_t[:, 0] >= 0
    sim_bit_t = agg_labels(idx_t, bits_t, y_full)
    sim_kmer_t = build_kmer_test(y_full)
    sim_t = ALIGN_W * sim_bit_t + (1 - ALIGN_W) * sim_kmer_t
    tp = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))
    t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))
    ens_t = (w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"]
             + w["cnn"] * tp["cnn"] + SIM_W * sim_t)
    scut_t = (ens_t >= t).astype(np.int8)
    dcard_t = neigh_card(idx_t, bits_t, y_full)
    mode, scale = best["mode"], best["scale"]
    pred_t = apply_hit(ens_t, scut_t, dcard_t * scale, hit_t, mode)
    os.makedirs(SUB_DIR, exist_ok=True)
    sub = pd.concat([test_df[["protein_id"]],
                     pd.DataFrame(pred_t, columns=label_cols)], axis=1)
    sub_path = os.path.join(SUB_DIR, "submission_CK005f_ensemble.csv")
    sub.to_csv(sub_path, index=False)
    report["submission"] = sub_path
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  提交: {sub_path} ({len(sub)} 行)")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
