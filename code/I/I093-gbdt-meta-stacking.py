"""
I093 — GBDT 元学习器堆叠（Friedman 残差提升完整版 vs I057 线性元学习器简化版）
==============================================================================
背景：I057 sl3/sl4 用 NNLS / 逐标签 LR 元学习器（−0.0055 / −0.0108 OOF）关闭了堆叠轴；
156 轮判定「非线性元学习器（GBDT）未试」，且预案写明「仅 I090–I092 无果时考虑」——现已全部
无果，激活本刀。

设计（半折 OOF，与 I057/I090 同协议；判据 = OOF 融合 Macro F1 较裸基线 ≥ +0.001）：
  特征（逐标签行内）：t001a / t002 / cnn(0.5cnn+0.5c005) / sim / plm1664 五路分数（X_meta-5）；
  扩展（X_meta-7）：+ 行特征（DIAMOND 命中数、top1 bitscore 归一）。
  臂：LR-7（线性对照）/ GBDT-5 / GBDT-7（HistGradientBoosting，depth=3）。
  每折：在折 A 上逐标签拟合元学习器 → 预测折 B → 交换；OOF 分数上再半折拟合 SCut 取评测。
基线锚：canonical 融合分同协议裸 SCut OOF = 0.4386（I090 实测），样本内 0.4475。
文献：Friedman, Annals of Statistics 2001, doi:10.1214/aos/1013203451（梯度提升）；
Super Learner 堆叠（NNLS 元学习器为经典形态，I057 已测）。
输出：rel/I093_report.json
"""
import os
import io
import json
import time

import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from joblib import Parallel, delayed
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
BASELINE_OOF = 0.4386  # I090 实测：canonical 融合分半折 OOF 裸 SCut
GATE = 0.001

i064 = SourceFileLoader(
    "i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")
).load_module()
macro_f1, search_scut = i064.macro_f1, i064.search_scut


def per_label_f1(S, y, ths):
    preds = S >= ths
    yb = y.astype(bool)
    tp = (preds & yb).sum(0)
    fp = (preds & ~yb).sum(0)
    fn = (~preds & yb).sum(0)
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)


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
    return 0.4 * sim2 + 0.6 * kmer, hit.astype(np.float32), bits_v[:, 0].copy()


def fit_label(kind, X, yj, Xte):
    """逐标签元学习器：LR 或 GBDT（HistGradientBoosting，Friedman 族非线性二阶段）。"""
    if yj.min() == yj.max():  # 折内无正例或全正：回退常数
        return np.full(Xte.shape[0], float(yj.mean()))
    if kind == "lr":
        m = LogisticRegression(max_iter=200, C=1.0)
        m.fit(X, yj.astype(np.int32))
        return m.predict_proba(Xte)[:, 1]
    m = HistGradientBoostingRegressor(max_iter=80, max_depth=3,
                                      learning_rate=0.06, early_stopping=False)
    m.fit(X, yj)
    return m.predict(Xte)


def oof_scores(kind, X, y, folds):
    """半折 OOF 元学习：A 拟合→B 预测，交换；返回 OOF 分数矩阵。X 形状 (n, k, 500)。"""
    S = np.zeros_like(y, dtype=np.float64)
    a, b = folds
    for fit_idx, eval_idx in ((a, b), (b, a)):
        cols = Parallel(n_jobs=-1)(
            delayed(fit_label)(kind, X[fit_idx][:, :, j], y[fit_idx][:, j],
                               X[eval_idx][:, :, j])
            for j in tqdm(range(N_LABELS), desc=f"{kind}/fold"))
        S[eval_idx] = np.stack(cols, axis=1)
    return S


def macro_f1_preds(preds, y):
    yb = y.astype(bool)
    tp = (preds & yb).sum(0)
    fp = (preds & ~yb).sum(0)
    fn = (~preds & yb).sum(0)
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    return float(np.mean(np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)))


def main():
    t0 = time.time()
    print("=" * 88)
    print("  I093: GBDT meta-stacking (Friedman full second stage) vs linear meta")
    print("=" * 88)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.float32)
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
    sim, hit, bits1 = build_sim(y_tr, n_va, val_pids, train_map, val_map)
    plm = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]

    # 元特征：(n_va, 5, 500) 与 (n_va, 7, 500) 的逐标签视图 -> 实际拟合用 (n, k) 每标签切片
    chans = [t1, t2, 0.5 * cnn + 0.5 * c005, sim, plm]
    row_feat = np.stack([hit, np.log1p(bits1) / 10.0], axis=1)  # (n_va, 2)
    X5 = np.stack(chans, axis=1)                                  # (n_va, 5, 500)
    X7 = np.concatenate([
        X5,
        np.broadcast_to(row_feat[:, None, :], (n_va, N_LABELS, 2)).transpose(0, 2, 1),
    ], axis=1)

    a_idx, b_idx = train_test_split(np.arange(n_va), test_size=0.5, random_state=42)
    folds = (a_idx, b_idx)
    rep = {"experiment": "I093", "baseline_oof_anchor": BASELINE_OOF,
           "gate": GATE, "arms": {}}

    # 协议校验（非判据）：线性混合分走同一 OOF 口径
    base_scores = 0.35 * plm + 0.65 * deep4 + 0.75 * sim
    preds = np.zeros_like(y_va, dtype=bool)
    for fi, ei in ((a_idx, b_idx), (b_idx, a_idx)):
        ths = search_scut(base_scores[fi], y_va[fi])
        preds[ei] = base_scores[ei] >= ths
    rep["sanity_linear_blend_oof"] = macro_f1_preds(preds, y_va.astype(np.int8))
    print(f"  sanity(线性混合 OOF) = {rep['sanity_linear_blend_oof']:.4f}（非判据）")

    for kind, use7, key in (("lr", True, "LR-7"), ("gbdt", False, "GBDT-5"),
                            ("gbdt", True, "GBDT-7")):
        S = oof_scores(kind, X7 if use7 else X5, y_va, folds)
        preds = np.zeros_like(y_va, dtype=bool)
        for fi, ei in ((a_idx, b_idx), (b_idx, a_idx)):
            ths = search_scut(S[fi], y_va[fi])
            preds[ei] = S[ei] >= ths
        f1_oof = macro_f1_preds(preds, y_va.astype(np.int8))
        ths_all = search_scut(S, y_va)
        f1_in = macro_f1(S, y_va, ths_all)
        fl = per_label_f1(S, y_va, ths_all)
        bands = np.array_split(np.argsort(-y_tr.sum(0)), 4)[::-1]
        rep["arms"][key] = {
            "f1_oof": f1_oof, "f1_in": f1_in,
            "delta_oof_vs_anchor": f1_oof - BASELINE_OOF,
            "band_f1_tail_to_head": [float(fl[b].mean()) for b in bands],
        }
        print(f"  {key:8s} OOF={f1_oof:.4f} (Δ{f1_oof - BASELINE_OOF:+.4f})  in-sample={f1_in:.4f}")

    best_key = max(rep["arms"], key=lambda k: rep["arms"][k]["f1_oof"])
    best_d = rep["arms"][best_key]["delta_oof_vs_anchor"]
    rep["best"] = best_key
    rep["best_delta"] = best_d
    rep["conclusion"] = ("有效，进入融合评估" if best_d >= GATE
                         else "关闭（未过 +0.001 换锁线）")
    with io.open(os.path.join(REL_DIR, "I093_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"\n  best = {best_key}  ΔOOF = {best_d:+.4f}  → {rep['conclusion']}")
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
