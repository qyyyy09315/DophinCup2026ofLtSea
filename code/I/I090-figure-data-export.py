"""
I090 — 论文图表数据导出 (不训练, 全部缓存分数)
================================================
为论文新增图导出两组数据到 doc/figures/fig_data_v2.json:
  1) 分支级细粒度平均 ROC 曲线 (FPR∈[0,0.2] 共 201 点, 逐标签 ROC 插值平均)
     分支: plm1664多种子 / deep5 / sim / 最终融合 —— 支撑「排序头部诊断」图
  2) 「独立增益不传导」设计法则散点数据 (I063/I074/I077/I087 汇编)
"""
import os, io, json, time
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc as sk_auc

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
OUT = os.path.join(PROJECT_ROOT, "doc", "figures", "fig_data_v2.json")
N_LABELS = 500
W_PLM = 0.35
FPR_GRID = np.linspace(0.0, 0.2, 201)

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
search_scut, full_pipe, ranks, col_norm, sinkhorn = (
    i064.search_scut, i064.full_pipe, i064.ranks, i064.col_norm, i064.sinkhorn)


def mean_roc(S, y):
    """逐标签 ROC 在公共 FPR 网格插值后取均值; 附 AUC/pAUC/固定FPR召回"""
    tprs, aucs, paucs = [], [], []
    rec = {f: [] for f in (0.02, 0.05, 0.069, 0.10)}
    for j in range(y.shape[1]):
        yt = y[:, j]
        if yt.min() == yt.max():
            continue
        fpr, tpr, _ = roc_curve(yt, S[:, j])
        if len(fpr) < 2:
            continue
        tprs.append(np.interp(FPR_GRID, fpr, tpr))
        aucs.append(sk_auc(fpr, tpr))
        m = fpr <= 0.10
        paucs.append(sk_auc(fpr[m], tpr[m]) / 0.10 if m.sum() >= 2 else 0.0)
        for f in rec:
            k = np.searchsorted(fpr, f, side="right") - 1
            rec[f].append(float(tpr[max(k, 0)]))
    return {"fpr": FPR_GRID.tolist(),
            "tpr_mean": np.mean(tprs, axis=0).tolist(),
            "auc": float(np.mean(aucs)), "pauc01": float(np.mean(paucs)),
            **{f"r@{f}": float(np.mean(rec[f])) for f in rec}}


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


def main():
    t0 = time.time()
    print("=" * 90)
    print("  I090: figure data export (fine ROC curves + design-law scatter)")
    print("=" * 90)

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
    deep5 = (1 - W_PLM) * deep4 + W_PLM * plm

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
    sim = 0.4 * sim2 + 0.6 * kmer

    _, _, mixed, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=plm, w=W_PLM)
    assert abs(i064.macro_f1(mixed, y_va, search_scut(mixed, y_va)) - 0.4475) < 0.0005

    print("  计算分支平均 ROC ...", flush=True)
    out = {"roc": {}}
    for name, S in [("plm1664", plm), ("deep5", deep5), ("sim", sim), ("fused", mixed)]:
        out["roc"][name] = mean_roc(S, y_va)
        r = out["roc"][name]
        print(f"    {name:9s} AUC={r['auc']:.4f} pAUC01={r['pauc01']:.4f} R@0.069={r['r@0.069']:.4f}")

    # ---- 设计法则散点 (来自 I063/I074/I077/I087 报告与日志) ----
    out["law"] = {
        "I063_2176D特征扩容": {"d_standalone": 0.2867 - 0.2590, "d_fused": 0.4462 - 0.4471},
        "I074_a0.5": {"d_standalone": 0.2715 - 0.2602, "d_fused": 0.4465 - 0.4472},
        "I074_a1.0_w3.33": {"d_standalone": 0.2709 - 0.2602, "d_fused": 0.4463 - 0.4472},
        "I074_a1.0_w6.0": {"d_standalone": 0.2738 - 0.2602, "d_fused": 0.4458 - 0.4472},
        "I087_自训练": {"d_standalone": 0.2681 - 0.2626, "d_fused": 0.4465 - 0.4469},
    }
    out["law_note"] = "x=独立F1增量, y=融合F1增量; 各点均相对同实验自身基线"

    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"\n  Saved: {OUT}")
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
