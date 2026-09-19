"""
I088 — 融合算子 A/B (CAFA6 银牌复盘的可迁移发现, 零训练成本)
==============================================================
CAFA6 (Kaggle, 2259 队) 银牌方案 22 次提交的实证:
  - 同水平强分支「等权算术平均」是最难打败的融合; 任何向弱分支倾斜的加权都掉分;
  - 概率尖锐度(高分计数)与分数强相关; 任何 top-K/过滤后处理全负。
我们从未测过简单算术平均 vs 现行 RRF+Sinkhorn。全部用缓存分数, 零训练。
变体 (deep5 = 0.65*deep4+0.35*plm 多种子; sim = 0.4*sim2+0.6*kmer):
  A0 现状:      s = deep5 + 0.75*sim + 0.5*RRF;  mixed = 0.4*s + 0.6*Sinkhorn(s)
  A1 去 RRF:    s = deep5 + 0.75*sim;            mixed = 0.4*s + 0.6*Sinkhorn(s)
  A2 去 Sinkhorn: s = deep5 + 0.75*sim + 0.5*RRF; 直接 SCut(s)
  A3 等权平均:  m = 0.5*deep5 + 0.5*sim; 直接 SCut(m)
  A4 等权+Sinkhorn: mixed = 0.4*m + 0.6*Sinkhorn(m)
  A5 max 组合:  max(col_norm(deep5), col_norm(sim)); 直接 SCut (DeepGOPlus 式)
"""
import os, io, json, time
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
REPORT = os.path.join(REL_DIR, "I088_report.json")
N_LABELS = 500
BASE = 0.4475

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
macro_f1, search_scut, ranks, col_norm, sinkhorn = (
    i064.macro_f1, i064.search_scut, i064.ranks, i064.col_norm, i064.sinkhorn)


def sharpness(S):
    return float((S >= 0.9).mean())


def parse_aln(tsv, qids, idmap, ev=0.001, mx=5):
    idx = np.full((len(qids), mx), -1, np.int64)
    bits = np.zeros((len(qids), mx), np.float32)
    qp = {q: i for i, q in enumerate(qids)}
    with open(tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            q, t = p[0], p[1]
            if q not in qp or t not in idmap: continue
            i = qp[q]
            try: bs, e = float(p[2]), float(p[3])
            except ValueError: continue
            if e > ev: continue
            j = 0
            while j < mx and bits[i, j] > bs: j += 1
            if j < mx:
                idx[i, j:] = np.roll(idx[i, j:], 1); bits[i, j:] = np.roll(bits[i, j:], 1)
                idx[i, j] = idmap[t]; bits[i, j] = bs
    return idx, bits


def agg(idx, w, ys, tk=3):
    n = idx.shape[0]
    valid = idx[:, :tk] >= 0
    lab = np.zeros((n, tk, ys.shape[1]), np.float32)
    for j in range(tk): lab[:, j] = ys[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :tk], 1e-7); ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)


def build_sim(y_tr, n_va, val_pids, train_map, val_map):
    idx_v, bits_v = parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                              val_pids, train_map)
    hit = idx_v[:, 0] >= 0; miss = ~hit
    sim1 = agg(idx_v, bits_v, y_tr)
    v2v = {}
    with open(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            q, t = p[0], p[1]
            if q == t or q not in val_map or t not in val_map: continue
            qi, ti = val_map[q], val_map[t]
            v2v.setdefault(qi, [])
            if len(v2v[qi]) < 5: v2v[qi].append((ti, float(p[2])))
    sim2 = sim1.copy()
    for qi in range(n_va):
        if miss[qi] and qi in v2v:
            acc = np.zeros(N_LABELS, np.float32); aw = 0.0
            for ti, bs in v2v[qi]:
                if hit[ti]: acc += sim1[ti] * bs; aw += bs
            if aw > 0: sim2[qi] = acc / aw
    kmer = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0: continue
        pre = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{pre}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_tr[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        sm = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(sm, 1e-7) ** 2
        kmer += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return 0.4 * sim2 + 0.6 * kmer


def with_sinkhorn(s, y_tr, y_va, n_va):
    t_tmp = search_scut(s, y_va)
    row_m = np.maximum((s >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_tr.mean(0) * n_va, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s - s.max(axis=1, keepdims=True)) / 0.20)
    p = sinkhorn(ker, row_m, col_m)
    return 0.40 * s + 0.60 * p


def main():
    t0 = time.time()
    print("=" * 100)
    print("  I088: Fusion operator A/B (CAFA6 silver-medal transferable findings)")
    print("=" * 100)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}

    cnn = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    plm = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn + 0.5 * c005)
    deep5 = 0.65 * deep4 + 0.35 * plm
    sim = build_sim(y_tr, n_va, val_pids, train_map, val_map)

    rrf = 0.50 * col_norm(1.0 / (30 + ranks(deep5) + 1) + 1.0 / (30 + ranks(sim) + 1))
    s_cur = deep5 + 0.75 * sim + rrf
    m_mean = 0.5 * deep5 + 0.5 * sim

    variants = [
        ("A0 现状(RRF+Sinkhorn)", with_sinkhorn(s_cur, y_tr, y_va, n_va)),
        ("A1 去RRF(+Sinkhorn)", with_sinkhorn(deep5 + 0.75 * sim, y_tr, y_va, n_va)),
        ("A2 去Sinkhorn(RRF)", s_cur),
        ("A3 等权算术平均", m_mean),
        ("A4 等权+Sinkhorn", with_sinkhorn(m_mean, y_tr, y_va, n_va)),
        ("A5 max组合(col_norm)", np.maximum(col_norm(deep5), col_norm(sim))),
    ]
    res = {}
    print(f"\n{'变体':24s} | {'F1':>7s} | {'dF1':>8s} | {'AUC':>7s} | {'尖锐度>=0.9':>10s}")
    print("-" * 70)
    for name, S in variants:
        f1 = macro_f1(S, y_va, search_scut(S, y_va))
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_va, S, average="macro")
        res[name] = {"f1": f1, "auc": float(auc), "sharp": sharpness(S)}
        print(f"{name:24s} | {f1:7.4f} | {f1 - BASE:+8.4f} | {auc:7.4f} | {sharpness(S):10.4f}", flush=True)

    best = max(res, key=lambda k: res[k]["f1"])
    print(f"\n  最优: {best} ({res[best]['f1']:.4f}, {res[best]['f1'] - BASE:+.4f} vs 基线)")
    print("  注: 若 A3/A4 接近或超过 A0, 支持 CAFA6 '等权平均最难打败' 的结论;")
    print("      若 A2/A1 与 A0 接近, 说明 RRF/Sinkhorn 在 F1 口径下近乎零贡献。")
    with io.open(REPORT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
