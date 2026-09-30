"""
I009-split-scut — hit / miss 分层逐标签阈值
================================================================
依据:
  - Yang Y, SIGIR 2001, doi:10.1145/383952.383975（SCut 逐标签）
  - InterLabelGO+ (Liu, Zhang & Freddolino, Bioinformatics 2024,
    doi:10.1093/bioinformatics/btae655)：同源强时靠比对、弱时靠序列模型，
    按命中质量切换，不是按长度（I005 已关）。
  - I006b：hit 过预测、miss 欠预测；同一套 100 档阈值无法同时服务两群。
    不估 k、不改分数，只把阈值按 DIAMOND 是否命中拆开。
方案:
  复用 CK005a 四路分数。SCut 在 hit 子集 / miss 子集 / 全量 上分别搜
  100 档逐标签阈值，再组合：
    split：hit 用 t_hit，miss 用 t_miss
    hit_only：hit 用 t_hit，miss 用全量 t
    miss_only：miss 用 t_miss，hit 用全量 t
  对照：全量 SCut（CK005a）。主指标全局 F1，门槛 +0.001。
用法:
  python code/I/I009-split-scut.py
  python code/I/I009-split-scut.py --gen-sub
输出:
  rel/I009_report.json
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
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
REPORT_PATH = os.path.join(REL_DIR, "I009_report.json")
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


def search_t(ens, y, mask=None):
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    e, yt_all = ens[mask], y[mask]
    cands = np.linspace(0.01, 0.99, 100)
    best = np.full(N_LABELS, 0.5)
    for i in range(N_LABELS):
        yt = yt_all[:, i]
        if yt.sum() == 0:
            continue
        pr = e[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0)
        fp = (pr & ~ytm).sum(0)
        fn = (~pr & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best[i] = cands[f1t.argmax()]
    return best


def apply_split(ens, t_hit, t_miss, hit):
    pred = np.zeros(ens.shape, np.int8)
    pred[hit] = (ens[hit] >= t_hit[None, :]).astype(np.int8)
    pred[~hit] = (ens[~hit] >= t_miss[None, :]).astype(np.int8)
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


def pack(name, pred, y, hit, miss, extra=None):
    r, p = macro_rp(pred, y)
    rec = {
        "name": name,
        "f1_100": round(macro_f1(pred, y), 6),
        "recall": round(r, 6),
        "precision": round(p, 6),
        "f1_miss": round(macro_f1(pred[miss], y[miss]), 6),
        "f1_hit": round(macro_f1(pred[hit], y[hit]), 6),
        "pred_card": round(float(pred.sum(1).mean()), 3),
        "miss_pred_card": round(float(pred[miss].sum(1).mean()), 3),
    }
    if extra:
        rec.update(extra)
    return rec


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print("I009: hit/miss 分层 SCut")

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
    ens = (w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"]
           + w["cnn"] * vp["cnn"] + SIM_W * sim)
    auc = float(np.mean([
        roc_auc_score(y_val[:, i], ens[:, i])
        for i in range(N_LABELS) if 0 < y_val[:, i].sum() < len(y_val)]))

    print("[1/2] 三套阈值...")
    t_all = search_t(ens, y_val)
    t_hit = search_t(ens, y_val, hit)
    t_miss = search_t(ens, y_val, miss)
    print(f"  t_all mean={t_all.mean():.3f}  t_hit mean={t_hit.mean():.3f}  "
          f"t_miss mean={t_miss.mean():.3f}")
    print(f"  hit 阈值相对全量: Δmean={t_hit.mean()-t_all.mean():+.3f}  "
          f"miss Δmean={t_miss.mean()-t_all.mean():+.3f}")

    pred_all = (ens >= t_all[None, :]).astype(np.int8)
    pred_split = apply_split(ens, t_hit, t_miss, hit)
    pred_hit_only = apply_split(ens, t_hit, t_all, hit)
    pred_miss_only = apply_split(ens, t_all, t_miss, hit)

    rows = [
        pack("scut_all", pred_all, y_val, hit, miss),
        pack("split", pred_split, y_val, hit, miss),
        pack("hit_only", pred_hit_only, y_val, hit, miss),
        pack("miss_only", pred_miss_only, y_val, hit, miss),
    ]
    base_f1 = rows[0]["f1_100"]
    for rec in rows:
        rec["delta"] = round(rec["f1_100"] - base_f1, 6)
        print(f"  {rec['name']:12s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f} "
              f"miss={rec['f1_miss']:.4f} hit={rec['f1_hit']:.4f} "
              f"card={rec['pred_card']:.1f}")

    best = max(rows, key=lambda r: r["f1_100"])
    gain = best["f1_100"] - base_f1
    useful = bool(best["name"] != "scut_all" and gain >= 0.001)
    print(f"  最优 {best['name']} F1={best['f1_100']:.4f} Δ={gain:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "baseline_scut": rows[0],
        "best": best,
        "gain_all_f1": round(gain, 6),
        "useful": useful,
        "auc_ens": round(auc, 6),
        "t_stats": {
            "all_mean": round(float(t_all.mean()), 4),
            "hit_mean": round(float(t_hit.mean()), 4),
            "miss_mean": round(float(t_miss.mean()), 4),
        },
        "grid": rows,
        "seconds": round(time.time() - t0, 1),
    }

    if gen_sub and useful:
        print("[2/2] test 分层阈值...")
        test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        tsv_t = os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
        if not os.path.exists(tsv_t):
            raise FileNotFoundError("需要 CK005a test tsv")
        full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
        idx_t, bits_t = parse_aln(tsv_t, test_df["protein_id"].values, full_map)
        hit_t = idx_t[:, 0] >= 0
        sim_bit_t = agg_labels(idx_t, bits_t, y_full)
        sim_k_t = 0.0
        for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
            if wt <= 0:
                continue
            d = np.load(os.path.join(REL_DIR, "TCK002a_neigh_full.npz" if k == 6
                                     else f"CK004_neigh_k{k}_full.npz"))
            n2 = d["idx_test"].shape[0]
            lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
            s = d["sim_test"][:, :5].astype(np.float32)
            ww = np.maximum(s, 1e-7) ** 2
            sim_k_t += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
        sim_t = ALIGN_W * sim_bit_t + (1 - ALIGN_W) * sim_k_t
        tp = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))
        t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
        t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))
        ens_t = (w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"]
                 + w["cnn"] * tp["cnn"] + SIM_W * sim_t)
        name = best["name"]
        if name == "split":
            pred_t = apply_split(ens_t, t_hit, t_miss, hit_t)
        elif name == "hit_only":
            pred_t = apply_split(ens_t, t_hit, t_all, hit_t)
        else:
            pred_t = apply_split(ens_t, t_all, t_miss, hit_t)
        os.makedirs(SUB_DIR, exist_ok=True)
        sub = pd.concat([test_df[["protein_id"]],
                         pd.DataFrame(pred_t, columns=label_cols)], axis=1)
        sub_path = os.path.join(SUB_DIR, "submission_I009_ensemble.csv")
        sub.to_csv(sub_path, index=False)
        report["submission"] = sub_path
        print(f"  提交: {sub_path} ({len(sub)} 行)")
    elif gen_sub:
        print("  --gen-sub 忽略（未过 +0.001）")

    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
