"""
I006-rcut — 样本级标签基数校正（RCut / 补满）
================================================================
依据:
  - Yang Y, SIGIR 2001, doi:10.1145/383952.383975
    文本分类阈值三策略：SCut（逐标签，本赛 100 档）/ PCut / RCut（每样本取 top-k）。
  - Read, Puurula & Bifet, ICDM 2014, doi:10.1109/icdm.2014.38
    Meta-label / 基数是多标签的一阶结构。
  - 实测 CK005a：val 真基数 mean=25.2；预测 29.5。miss 子集真 16.2、预测仅 8.6
    → miss 是欠预测，不是排错。
方案:
  复用 CK005a 四路分数。先 SCut 得逐标签阈值，再按 k_hat 补满或截断。
  k_hat 网格：真基数上界（oracle）/ 线性(ens_sum,len) / 固定倍数。
  主指标: 全局 F1 不掉 + miss F1。
用法:
  python code/I/I006-rcut.py
输出:
  rel/I006_report.json
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
REPORT_PATH = os.path.join(REL_DIR, "I006_report.json")
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


def apply_k(ens, k_hat, mode="fill"):
    """mode=fill: SCut 后若不足 k 则按分数补满；rcut: 纯 top-k。"""
    n, c = ens.shape
    k_hat = np.clip(np.rint(k_hat), 1, c).astype(np.int32)
    order = np.argsort(-ens, axis=1)
    if mode == "rcut":
        pred = np.zeros((n, c), np.int8)
        for i in range(n):
            pred[i, order[i, :k_hat[i]]] = 1
        return pred
    # fill: 先占位，下面用 scut 并集
    return order, k_hat


def fill_from_scut(ens, scut, k_hat):
    pred = scut.copy()
    n, c = ens.shape
    k_hat = np.clip(np.rint(k_hat), 1, c).astype(np.int32)
    order = np.argsort(-ens, axis=1)
    for i in tqdm(range(n), desc="    补满", unit="row", leave=False):
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


def main():
    t0 = time.time()
    print("=" * 60)
    print("I006: 样本级基数校正（RCut / SCut+补满）")

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
    ens = (w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"]
           + w["cnn"] * vp["cnn"] + SIM_W * sim)

    t = search_t(ens, y_val)
    scut = (ens >= t).astype(np.int8)
    pcard = scut.sum(1).astype(np.float32)
    print(f"  真基数 mean={card.mean():.1f}  SCut基数={pcard.mean():.1f}  "
          f"miss真={card[miss].mean():.1f} miss预测={pcard[miss].mean():.1f}")

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
            "miss_pred_card": round(float(pred[miss].sum(1).mean()), 3),
        }

    rows = [pack("scut", scut)]
    print(f"  SCut: F1={rows[0]['f1_100']:.4f} missF1={rows[0]['f1_miss']:.4f}")

    # oracle k = 真基数（上界，看排序够不够）
    pred_or = np.zeros_like(scut)
    order = np.argsort(-ens, axis=1)
    for i in range(len(idx_val)):
        k = max(1, int(card[i]))
        pred_or[i, order[i, :k]] = 1
    rows.append(pack("rcut_oracle", pred_or))
    print(f"  RCut-oracle: F1={rows[-1]['f1_100']:.4f} missF1={rows[-1]['f1_miss']:.4f}")

    pred_fill_or = fill_from_scut(ens, scut, card)
    rows.append(pack("fill_oracle", pred_fill_or))
    print(f"  Fill-oracle: F1={rows[-1]['f1_100']:.4f} missF1={rows[-1]['f1_miss']:.4f}")

    # 线性 k_hat：ens_sum + loglen + scut_card，半样本拟合防泄漏
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(idx_val))
    mid = len(perm) // 2
    fit, hold = perm[:mid], perm[mid:]
    x = np.stack([ens.sum(1), np.log1p(val_len), pcard], 1)
    reg = Ridge(alpha=1.0)
    reg.fit(x[fit], card[fit])
    kh = np.clip(reg.predict(x), 1, 80)
    print(f"  k_hat corr={np.corrcoef(card, kh)[0, 1]:.3f}  hold MAE="
          f"{np.mean(np.abs(card[hold] - kh[hold])):.2f}")

    for name, pred in [
        ("rcut_khat", None),
        ("fill_khat", None),
        ("fill_khat_miss_only", None),
        ("fill_scut1.5_miss", None),
    ]:
        if name == "rcut_khat":
            pred = np.zeros_like(scut)
            for i in range(len(idx_val)):
                pred[i, order[i, :int(kh[i])]] = 1
        elif name == "fill_khat":
            pred = fill_from_scut(ens, scut, kh)
        elif name == "fill_khat_miss_only":
            k2 = pcard.copy()
            k2[miss] = np.maximum(pcard[miss], kh[miss])
            pred = fill_from_scut(ens, scut, k2)
        else:
            k2 = pcard.copy()
            k2[miss] = np.maximum(pcard[miss], 1.5 * np.maximum(pcard[miss], 1))
            pred = fill_from_scut(ens, scut, k2)
        rec = pack(name, pred)
        rows.append(rec)
        print(f"  {name}: F1={rec['f1_100']:.4f} missF1={rec['f1_miss']:.4f} "
              f"card={rec['pred_card']:.1f}")

    base = rows[0]
    ranked = [r for r in rows if r["f1_100"] >= base["f1_100"] - 0.001]
    best = max(ranked, key=lambda r: (r["f1_miss"], r["f1_100"]))
    useful = (best["f1_miss"] - base["f1_miss"] >= 0.005
              and best["f1_100"] - base["f1_100"] >= -0.001
              and best["name"] != "scut"
              and "oracle" not in best["name"])
    print(f"  实务最优 {best['name']} ΔF1={best['f1_100'] - base['f1_100']:+.4f} "
          f"Δmiss={best['f1_miss'] - base['f1_miss']:+.4f} → "
          f"{'有效' if useful else '无增益'}")

    auc = float(np.mean([
        roc_auc_score(y_val[:, i], ens[:, i])
        for i in range(N_LABELS)
        if 0 < y_val[:, i].sum() < len(y_val)]))
    report = {
        "baseline_scut": rows[0], "grid": rows, "best": best,
        "useful": useful, "auc_ens": round(auc, 6),
        "khat_corr": round(float(np.corrcoef(card, kh)[0, 1]), 4),
        "seconds": round(time.time() - t0),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
