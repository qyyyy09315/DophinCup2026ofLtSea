"""
I072 — 低 FPR 区（排序头部）分路诊断
=====================================
动机:
  I070/I071 定位到缺口在"低 FPR 区的召回"：同一 FPR=0.069 处，榜首召回 ~0.82，我方仅 ~0.53。
  且 RCut 路线已关闭（逐样本基数不可预测）。故必须回到"分数本身的头部排序质量"。
本诊断回答: 在低 FPR 区，各分路与融合结果的召回分别是多少？是否存在被融合稀释的"头部专家"？
指标:
  - 部分 AUC (pAUC) in FPR∈[0, 0.10]
  - 固定 FPR 下的召回: FPR = 0.02 / 0.05 / 0.069 / 0.10
  - 全 AUC
输出: rel/I072_report.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
REPORT = os.path.join(REL_DIR, "I072_report.json")
N_LABELS = 500
W_PLM = 0.35
FPR_GRID = [0.02, 0.05, 0.069, 0.10]

from importlib.machinery import SourceFileLoader
i66c = SourceFileLoader("i66c", os.path.join(PROJECT_ROOT, "code", "I", "I066c-linear-calibration.py")).load_module()


def branch_metrics(S, y, name):
    """逐标签 ROC -> 汇总 AUC / pAUC / 固定 FPR 召回"""
    aucs, paucs = [], []
    rec_at = {f: [] for f in FPR_GRID}
    from sklearn.metrics import auc as sk_auc
    for j in range(N_LABELS):
        yt = y[:, j]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        fpr, tpr, _ = roc_curve(yt, S[:, j])
        if len(fpr) < 2:
            continue
        aucs.append(sk_auc(fpr, tpr))
        m = fpr <= 0.10
        if m.sum() >= 2:
            paucs.append(sk_auc(fpr[m], tpr[m]) / 0.10)   # 归一化到 [0,1]
        else:
            paucs.append(0.0)
        for f in FPR_GRID:
            idx = np.searchsorted(fpr, f, side="right") - 1
            rec_at[f].append(tpr[max(idx, 0)])
    out = {
        "name": name,
        "auc": float(np.mean(aucs)),
        "pauc_0_10": float(np.mean(paucs)),
    }
    for f in FPR_GRID:
        out[f"recall@fpr{f}"] = float(np.mean(rec_at[f]))
    return out


def main():
    t0 = time.time()
    print("=" * 82)
    print("  I072: Head-of-Ranking (Low-FPR) Diagnosis by Branch")
    print("=" * 82)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]

    # 各分路
    cnn_v = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]      # C002b
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]      # CSRA
    t1v = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2v = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    plm_v = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]

    deep4_v = wc["t001a"]*t1v + wc["t002"]*t2v + wc["cnn"]*(0.5*cnn_v + 0.5*c005_v)
    deep5_v = (1-W_PLM)*deep4_v + W_PLM*plm_v

    # 同源两路
    idx_v, bits_v = i66c.parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                                   val_pids, train_map)
    hit_v = idx_v[:, 0] >= 0
    sim1_v = i66c.agg(idx_v, bits_v, y_tr)
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
    sim2_v = i66c.transitive_2hop(sim1_v, hit_v, ~hit_v, v2v, n_va)
    kmer_v = i66c.build_kmers(y_tr, "val")
    sim_branch = 0.4*sim2_v + 0.6*kmer_v

    # 融合结果
    s_v = deep5_v + 0.75*sim_branch + 0.50*i66c.col_norm(
        1.0/(30+i66c.ranks(deep5_v)+1) + 1.0/(30+i66c.ranks(sim_branch)+1))
    t_tmp = i66c.search_scut(s_v, y_va)
    row_m = np.maximum((s_v >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_tr.mean(0)*n_va, 1e-6); col_m *= row_m.sum()/col_m.sum()
    p_v = i66c.sinkhorn(np.exp((s_v - s_v.max(axis=1, keepdims=True))/0.20), row_m, col_m)
    final_v = 0.40*s_v + 0.60*p_v

    branches = [
        ("C002b (膨胀CNN)", cnn_v),
        ("C005 (CSRA)", c005_v),
        ("T001a (RoFormer)", t1v),
        ("T002 (RoFormer)", t2v),
        ("PLM-MLP 1664D (多种子)", plm_v),
        ("Deep-4路", deep4_v),
        ("Deep-5路(+PLM)", deep5_v),
        ("Sim (比对+2hop+kmer)", sim_branch),
        ("Sim 仅比对(1hop)", sim1_v),
        ("Sim 仅 k-mer", kmer_v),
        ("RRF 融合 (未 Sinkhorn)", s_v),
        ("最终 F (RRF+Sinkhorn)", final_v),
    ]

    print(f"\n{'分路':26s} | {'AUC':>7s} | {'pAUC[0,.1]':>10s} | " +
          " | ".join([f"R@FPR{f:<4}" for f in FPR_GRID]))
    print("-" * 82)
    rows = []
    for name, S in branches:
        m = branch_metrics(S, y_va, name)
        rows.append(m)
        print(f"{name:26s} | {m['auc']:7.4f} | {m['pauc_0_10']:10.4f} | " +
              " | ".join([f"{m[f'recall@fpr{f}']:9.4f}" for f in FPR_GRID]))

    # 排序: 按 FPR=0.069 召回
    key = "recall@fpr0.069"
    best = max(rows, key=lambda r: r[key])
    print("\n" + "=" * 82)
    print(f"  FPR=0.069 处召回最高: {best['name']} -> {best[key]:.4f}")
    print(f"  最终融合结果在该点召回: {[r for r in rows if r['name']=='最终 F (RRF+Sinkhorn)'][0][key]:.4f}")
    print(f"  pAUC[0,0.10] 最高: {max(rows, key=lambda r: r['pauc_0_10'])['name']}")
    print("=" * 82)

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
