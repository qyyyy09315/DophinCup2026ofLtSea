"""
I095 — 平台二值口径差距分解（分析性，不写腾讯表格）
====================================================
背景（157 轮）：新锁 I081 = 45.9546 / 提交 AUC(0/1) 0.71459；09-23/24 榜单前十门槛 46.4291
（AUC 0.7155）、榜首 49.3984（AUC 0.7312）。我方二值 AUC 与第十名近持平而 F1 低 0.47 ——
同均衡度下 F1 更低 ⇒ 操作点偏高 TPR/高 FPR，PPV 被 ~1% 流行率惩罚（157 轮假说）。

三件套分解（canonical 融合分，I065 口径；val 上操作，与平台测试口径的差异注明）：
  a) 操作点画像：逐标签 (TPR, FPR) 分布 + 二值 AUC(0/1) = mean((TPR+TNR)/2) 复算；
  b) FPR 约束斜率：沿各标签 ROC 把 FPR 压到 k 倍（k=0.75/0.5/0.25，取 FPR≤k·FPR_j 的最大
     TPR，真实可达操作点），得 Macro F1(k) 曲线 → 反解「进前十 +0.47 平台分」所需的 k；
  c) 恒均衡度跨度：固定 (TPR+TNR)/2 = A0，扫 (t,f) 混合结构 → 同 AUC 下 F1 的可达跨度，
     定位我方与榜首在该曲线上的相对位置（含 #4 反例解释）。
输出：rel/I095_report.json（只写实验日志）。
"""
import os
import io
import json
import time

import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
TARGET_GAP = 0.0047  # 进前十差 +0.47 平台分 ≈ +0.0047 测试 Macro F1

i064 = SourceFileLoader(
    "i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")
).load_module()
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe


def build_sim(y_tr, n_va, val_pids, train_map, val_map):
    def parse_aln(tsv, qids, idmap, ev=0.001, mx=5):
        idx = np.full((len(qids), mx), -1, np.int64)
        bits = np.zeros((len(qids), mx), np.float32)
        qp = {q: i for i, q in enumerate(qids)}
        with open(tsv) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 5:
                    continue
                q, t = p[0], p[1]
                if q not in qp or t not in idmap:
                    continue
                i = qp[q]
                try:
                    bs, e = float(p[2]), float(p[3])
                except ValueError:
                    continue
                if e > ev:
                    continue
                j = 0
                while j < mx and bits[i, j] > bs:
                    j += 1
                if j < mx:
                    idx[i, j:] = np.roll(idx[i, j:], 1)
                    bits[i, j:] = np.roll(bits[i, j:], 1)
                    idx[i, j] = idmap[t]
                    bits[i, j] = bs
        return idx, bits

    def agg(idx, w, ys, tk=3):
        n = idx.shape[0]
        valid = idx[:, :tk] >= 0
        lab = np.zeros((n, tk, ys.shape[1]), np.float32)
        for j in range(tk):
            lab[:, j] = ys[idx[:, j].clip(min=0)]
        lab[~valid] = 0.0
        ww = np.maximum(w[:, :tk], 1e-7)
        ww[~valid] = 0.0
        return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)

    idx_v, bits_v = parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                              val_pids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim1 = agg(idx_v, bits_v, y_tr)
    v2v = {}
    with open(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            q, t = p[0], p[1]
            if q == t or q not in val_map or t not in val_map:
                continue
            qi, ti = val_map[q], val_map[t]
            v2v.setdefault(qi, [])
            if len(v2v[qi]) < 5:
                v2v[qi].append((ti, float(p[2])))
    sim2 = sim1.copy()
    for qi in range(n_va):
        if miss[qi] and qi in v2v:
            acc = np.zeros(N_LABELS, np.float32)
            aw = 0.0
            for ti, bs in v2v[qi]:
                if hit[ti]:
                    acc += sim1[ti] * bs
                    aw += bs
            if aw > 0:
                sim2[qi] = acc / aw
    kmer = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        pre = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{pre}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_tr[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        sm = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(sm, 1e-7) ** 2
        kmer += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return 0.4 * sim2 + 0.6 * kmer


def f1_from_rates(tpr, fpr, pi):
    tp = tpr * pi
    fp = fpr * (1 - pi)
    fn = (1 - tpr) * pi
    p = tp / np.maximum(tp + fp, 1e-12)
    r = tpr
    return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)


def main():
    t0 = time.time()
    print("=" * 88)
    print("  I095: binary-metric gap decomposition (analytic)")
    print("=" * 88)

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
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn + 0.5 * c005)
    print("  building sim ...")
    sim = build_sim(y_tr, n_va, val_pids, train_map, val_map)
    plm = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    f1c, aucc, mixed, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=plm, w=0.35)
    assert abs(f1c - 0.4475) < 0.0005, f1c
    print(f"  canonical anchor OK: F1={f1c:.4f}")

    rep = {"experiment": "I095", "analytic": True, "anchor_f1": f1c}

    # ---- a) 操作点画像 ----
    ths = search_scut(mixed, y_va)
    yb = y_va.astype(bool)
    preds = mixed >= ths
    P = yb.sum(0).astype(np.float64)
    N = (~yb).sum(0).astype(np.float64)
    tp = (preds & yb).sum(0).astype(np.float64)
    fp = (preds & ~yb).sum(0).astype(np.float64)
    tpr = tp / np.maximum(P, 1.0)
    fpr = fp / np.maximum(N, 1.0)
    pi = y_va.mean(0)
    bin_auc = float(np.mean((tpr + (1 - fpr)) / 2))
    rep["operating_point"] = {
        "binary_auc_val": bin_auc,
        "platform_binary_auc_ref": 0.71459,
        "macro_tpr": float(tpr.mean()), "macro_fpr": float(fpr.mean()),
        "macro_ppv": float(np.mean(tp / np.maximum(tp + fp, 1e-9))),
    }
    print(f"  [a] 二值 AUC(val)={bin_auc:.4f}（平台 0.71459 参照） "
          f"TPR={tpr.mean():.4f} FPR={fpr.mean():.4f} PPV={rep['operating_point']['macro_ppv']:.4f}")

    # ---- b) FPR 约束斜率（沿 ROC 真实可达） ----
    slope = {}
    rocs = []
    for j in tqdm(range(N_LABELS), desc="roc"):
        if P[j] == 0 or N[j] == 0:
            rocs.append(None)
            continue
        f, tt, _ = roc_curve(y_va[:, j], mixed[:, j])
        rocs.append((f, tt))
    for k in (0.75, 0.5, 0.25):
        tpr_k = np.zeros(N_LABELS)
        for j in range(N_LABELS):
            if rocs[j] is None:
                tpr_k[j] = tpr[j]
                continue
            f, tt = rocs[j]
            cap = k * max(fpr[j], 1.0 / max(N[j], 1.0))
            m = f <= cap
            tpr_k[j] = tt[m].max() if m.any() else 0.0
        f1_k = float(f1_from_rates(tpr_k, np.minimum(k * np.maximum(fpr, 1.0 / N), 1.0), pi).mean())
        slope[f"k{k:g}"] = {"f1": f1_k, "delta": f1_k - f1c,
                            "macro_tpr_kept": float(tpr_k.mean())}
        print(f"  [b] FPR×{k:.2f}: F1={f1_k:.4f} (Δ{slope[f'k{k:g}']['delta']:+.4f}) "
              f"TPR 保持 {tpr_k.mean():.4f}")
    rep["fpr_slope"] = slope
    best_delta = max(slope[f"k{k:g}"]["delta"] for k in (0.75, 0.5, 0.25))
    rep["fpr_slope"]["verdict"] = (
        "可达" if best_delta >= TARGET_GAP else
        "不可达：沿 ROC 的 FPR 压缩在 val 上恒损 F1（SCut 已是逐标签 F1 最优），"
        "差距只能来自分数序/判别力")
    print(f"  [b] 进前十目标 Δ+{TARGET_GAP:.4f}: " + rep["fpr_slope"]["verdict"])

    # ---- c) 恒均衡度跨度 ----
    A0 = bin_auc
    mixes = []
    for t in np.linspace(0.05, 0.95, 19):
        f = t + 1 - 2 * A0
        if 0.0 <= f <= 1.0:
            f1_mix = float(f1_from_rates(np.full(N_LABELS, t), np.full(N_LABELS, f), pi).mean())
            mixes.append({"tpr": float(t), "fpr": float(f), "f1": f1_mix})
    span = (min(m["f1"] for m in mixes), max(m["f1"] for m in mixes)) if mixes else (0, 0)
    rep["iso_balacc_span"] = {"A0": A0, "f1_min": span[0], "f1_max": span[1],
                              "curve": mixes[::3]}
    print(f"  [c] 同二值 AUC={A0:.4f} 下 F1 可达跨度 {span[0]:.4f} ~ {span[1]:.4f} "
          f"（混合结构自由度即决策空间）")

    # 频带分解（头/尾四分位）
    bands = np.array_split(np.argsort(-y_tr.sum(0)), 4)
    rep["band_macro"] = {
        "head-Q4": {"tpr": float(tpr[bands[0]].mean()), "fpr": float(fpr[bands[0]].mean())},
        "tail-Q1": {"tpr": float(tpr[bands[3]].mean()), "fpr": float(fpr[bands[3]].mean())},
    }

    with io.open(os.path.join(REL_DIR, "I095_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\nCompleted in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
