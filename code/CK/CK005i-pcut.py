"""
CK005i-pcut — 实例分位数截断（不估 k）
================================================================
依据:
  - Yang Y, SIGIR 2001, doi:10.1145/383952.383975
    PCut：按比例/分位截断，不是回归每样本基数。
  - CK005e/f：k 回归 MAE≈15，邻居 MAE≈17，RCut 实务失败。
  - CK005h：0.52 上界几乎全在 hit 截断；真 k±1 才保住，估 k 路封死。
方案:
  复用 CK005a 四路分数 + SCut。不预测 k。
  对每条样本用自身分数的分位数 τ 做截断，再与 SCut 组合：
    - intersect：SCut ∧ (score ≥ 该行 q_p)
    - replace：只保留 score ≥ 该行 q_p（至少 1 个）
    - hit_only：只改 hit 行（CK005h 增益来源）
  p ∈ {0.50, 0.60, 0.70, 0.80, 0.90, 0.95}
  对照：全局固定分数阈（非分位）× {hit, all}
  主指标：全局 100 档口径 Macro F1，门槛 +0.001。
用法:
  python code/CK/CK005i-pcut.py
  python code/CK/CK005i-pcut.py --gen-sub   # 仅有效时
输出:
  rel/CK005i_report.json
  submit/submission_CK005i_ensemble.csv（仅 --gen-sub 且有效）
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
REPORT_PATH = os.path.join(REL_DIR, "CK005i_report.json")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
ALIGN_W = 0.4
SIM_W = 0.75
TOP_K = 3
MAX_ACCEPT = 5
RANDOM_STATE = 42
PS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


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


def row_quantile_mask(ens, p):
    """每行 score ≥ 该行 p 分位；保证每行至少 1 个。"""
    thr = np.quantile(ens, p, axis=1).astype(np.float32)
    mask = ens >= thr[:, None]
    empty = ~mask.any(axis=1)
    if empty.any():
        top = ens[empty].argmax(axis=1)
        mask[empty, top] = True
    return mask


def apply_scope(base, overlay, scope, hit):
    pred = base.copy()
    if scope == "all":
        pred = overlay
    elif scope == "hit":
        pred[hit] = overlay[hit]
    else:
        raise ValueError(scope)
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


def load_ens_val():
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
    return train_df, label_cols, y_full, y_val, hit, ens, w


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print("CK005i: 实例分位数截断（PCut，不估 k）")

    train_df, label_cols, y_full, y_val, hit, ens, w = load_ens_val()
    miss = ~hit
    print(f"  hit {int(hit.sum())}/{len(hit)} miss {int(miss.sum())}")

    t = search_t(ens, y_val)
    scut = (ens >= t).astype(np.int8)
    auc = macro_auc(y_val, ens)
    rows = [pack("scut", scut, y_val, hit, miss)]
    print(f"  SCut F1={rows[0]['f1_100']:.4f} card={rows[0]['pred_card']:.1f}")

    print("[1/2] 行分位网格...")
    for p in tqdm(PS, desc="    p", unit="p"):
        qmask = row_quantile_mask(ens, p)
        qpred = qmask.astype(np.int8)
        inter = (scut & qmask).astype(np.int8)
        empty = inter.sum(1) == 0
        if empty.any():
            inter[empty] = scut[empty]
        for mode, overlay in [("intersect", inter), ("replace", qpred)]:
            for scope in ("hit", "all"):
                pred = apply_scope(scut, overlay, scope, hit)
                rec = pack(
                    f"{mode}_p{p:.2f}_{scope}", pred, y_val, hit, miss,
                    {"p": p, "mode": mode, "scope": scope,
                     "delta": round(macro_f1(pred, y_val) - rows[0]["f1_100"], 6)})
                rows.append(rec)
                print(f"  {rec['name']:28s} F1={rec['f1_100']:.4f} "
                      f"d={rec['delta']:+.4f} miss={rec['f1_miss']:.4f} "
                      f"card={rec['pred_card']:.1f}")

    print("[2/2] 全局固定阈对照...")
    for thr in (0.40, 0.50, 0.60):
        gmask = ens >= thr
        gpred = gmask.astype(np.int8)
        empty = gpred.sum(1) == 0
        if empty.any():
            top = ens[empty].argmax(1)
            gpred[empty, top] = 1
        inter = (scut & gmask).astype(np.int8)
        empty = inter.sum(1) == 0
        if empty.any():
            inter[empty] = scut[empty]
        for mode, overlay in [("intersect", inter), ("replace", gpred)]:
            pred = apply_scope(scut, overlay, "hit", hit)
            rec = pack(
                f"{mode}_t{thr:.2f}_hit", pred, y_val, hit, miss,
                {"thr": thr, "mode": mode, "scope": "hit",
                 "delta": round(macro_f1(pred, y_val) - rows[0]["f1_100"], 6)})
            rows.append(rec)
            print(f"  {rec['name']:28s} F1={rec['f1_100']:.4f} "
                  f"d={rec['delta']:+.4f} card={rec['pred_card']:.1f}")

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
        "grid": rows,
        "seconds": round(time.time() - t0, 1),
    }

    if gen_sub and useful:
        print("[sub] test 端重算最优配置...")
        test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        tsv_t = os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
        if not os.path.exists(tsv_t):
            raise FileNotFoundError("需要 CK005a test tsv")
        full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
        idx_t, bits_t = parse_aln(tsv_t, test_df["protein_id"].values, full_map)
        sim_bit_t = agg_labels(idx_t, bits_t, y_full)
        sim_kmer_t = build_kmer_test(y_full)
        sim_t = ALIGN_W * sim_bit_t + (1 - ALIGN_W) * sim_kmer_t
        tp = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))
        t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
        t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))
        ens_t = (w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"]
                 + w["cnn"] * tp["cnn"] + SIM_W * sim_t)
        scut_t = (ens_t >= t[None, :]).astype(np.int8)
        hit_t = idx_t[:, 0] >= 0
        p = float(best.get("p", 0.8))
        mode = best.get("mode", "intersect")
        scope = best.get("scope", "hit")
        qmask = row_quantile_mask(ens_t, p)
        if mode == "intersect":
            overlay = (scut_t & qmask).astype(np.int8)
            empty = overlay.sum(1) == 0
            if empty.any():
                overlay[empty] = scut_t[empty]
        else:
            overlay = qmask.astype(np.int8)
        pred_t = apply_scope(scut_t, overlay, scope, hit_t)
        os.makedirs(SUB_DIR, exist_ok=True)
        sub = pd.concat([test_df[["protein_id"]],
                         pd.DataFrame(pred_t, columns=label_cols)], axis=1)
        sub_path = os.path.join(SUB_DIR, "submission_CK005i_ensemble.csv")
        sub.to_csv(sub_path, index=False)
        report["submission"] = sub_path
        print(f"  提交: {sub_path} ({len(sub)} 行)")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
