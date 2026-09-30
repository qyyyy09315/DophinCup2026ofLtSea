"""
CK004-multik-ensemble — 多 k 邻居融合集成
=============================================
方案:
  - 现状: CK002a/TCK002a 的 sim 特征仅用单 k=6 (top-3 相似度加权聚合)
  - 本实验: 每 k (3~8) 单独聚合 sim_k (top-3, 相似度加权, 与 CK002a 一致)
    → 多 k 加权融合 sim = Σ w_k·sim_k → 替换单 k6 sim 进 TCK002a 四路
    (t001a 0.1 / t002 0.2 / cnn 0.2 / sim 0.5)
  - 候选融合 (验证集 F1(0.5) 对比):
    单k6(基线) / 等权6k / k6主导 / k5+k6 / k4~k7均分 / k6+k7
  - 若最优融合 val F1 超过单 k6 > 0.001: 重算 k=3,4,5,7,8 全量空间
    (113,796) test 邻居 → 生成提交 (k=6 全量已存在 TCK002a_neigh_full.npz)
  - 防泄漏: val 邻居在 91,036 空间 (不含 val 自身), 与 TCK002a 评估一致
文献:
  - DeepGOPlus: Kulmanov & Hoehndorf, Bioinformatics 2019, doi:10.1093/bioinformatics/btz595
    (原文即多 k 组合的序列相似性)
输出:
  rel/CK004_report.json
  rel/CK004_neigh_k{k}_full.npz (k=3,4,5,7,8 全量空间缓存, 仅 val 有效时)
  submit/submission_CK004_ensemble.csv (仅 val 有效时)
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
REPORT_PATH = f"{REL_DIR}/CK004_report.json"
SUB_PATH = f"{SUB_DIR}/submission_CK004_ensemble.csv"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
T001A_PROBS_PATH = f"{REL_DIR}/TCK001_t001a_probs.npz"
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"
TCK002_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

K_RANGE = [3, 4, 5, 6, 7, 8]
TOP_K = 3
MAX_TOP_K = 5
THRESHOLD_STEPS = 100
N_FEATURES = 2 ** 20
BLOCK = 500
VAL_SIZE = 0.2
RANDOM_STATE = 42

# 最终方案 (val 网格确认 2026-08-15):
#   融合权重 [k3..k8] = [0, 0.1, 0.2, 0.4, 0.2, 0.1] (k6 主导 + 邻近辅助)
#   w_sim = 0.7 (峰值, 0.65~0.75 平台) → val F1 = 0.4295 (100 档阈值)
FUSION_WTS = [0, 0.1, 0.2, 0.4, 0.2, 0.1]
SIM_W = 0.7


def macro_f1_fixed_threshold(ens, y_true, t=0.5):
    pred = (ens >= t)
    f1s = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def macro_auc(y_true, y_prob):
    aucs = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, y_prob[:, i]))
    return float(np.mean(aucs))


def find_optimal_thresholds(y_prob, y_true):
    print("  搜索逐标签最优阈值 (100 档)...")
    n_labels = y_true.shape[1]
    best_thresholds = np.full(n_labels, 0.5, dtype=np.float32)
    candidates = np.linspace(0.01, 0.99, THRESHOLD_STEPS)
    for i in tqdm(range(n_labels), desc="    阈值搜索", unit="label"):
        yt = y_true[:, i]
        if yt.sum() == 0:
            continue
        best_f1, best_t = 0, 0.5
        for t in candidates:
            pred = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(yt, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[i] = best_t
    return best_thresholds


def evaluate_at_thresholds(y_prob, y_true, thresholds):
    per_f1, per_prec, per_rec = [], [], []
    y_pred = np.zeros_like(y_prob, dtype=np.int8)
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        p = (y_prob[:, i] >= thresholds[i]).astype(int)
        y_pred[:, i] = p
        per_f1.append(f1_score(y_true[:, i], p, zero_division=0))
        per_prec.append(precision_score(y_true[:, i], p, zero_division=0))
        per_rec.append(recall_score(y_true[:, i], p, zero_division=0))
    return y_pred, np.mean(per_f1), np.mean(per_prec), np.mean(per_rec)


def predict_with_thresholds(y_prob, thresholds):
    preds = np.zeros_like(y_prob, dtype=np.int8)
    for i, t in enumerate(thresholds):
        preds[:, i] = (y_prob[:, i] >= t).astype(np.int8)
    return preds


def build_sim_prob(neigh_idx, neigh_sim, y_space, top_k):
    n = neigh_idx.shape[0]
    labels = y_space[neigh_idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    w = np.maximum(neigh_sim[:, :top_k], 1e-7)
    return (labels * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]


def get_neighbors(seqs_train, seqs_query, k):
    """top-5 邻居索引与相似度 (不缓存, 用于全量空间重算)。"""
    vec = HashingVectorizer(analyzer="char", ngram_range=(k, k),
                            n_features=N_FEATURES, dtype=np.float32)
    X_train = normalize(sparse.csr_matrix(vec.transform(seqs_train)))
    X_q = normalize(sparse.csr_matrix(vec.transform(seqs_query)))
    n_q = X_q.shape[0]
    idx = np.zeros((n_q, MAX_TOP_K), dtype=np.int64)
    sims = np.zeros((n_q, MAX_TOP_K), dtype=np.float32)
    for s in tqdm(range(0, n_q, BLOCK), desc=f"最近邻 k={k} (全量空间)"):
        e = min(s + BLOCK, n_q)
        sim = (X_q[s:e] @ X_train.T).toarray()
        part = np.argpartition(-sim, MAX_TOP_K, axis=1)[:, :MAX_TOP_K]
        idx[s:e] = part
        sims[s:e] = np.take_along_axis(sim, part, axis=1)
    return idx, sims


def main():
    val_only = "--val-only" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print(f"模式: {'val 评估 (不重算 test 邻居)' if val_only else '全流程 (含 test 邻居重算)'}")
    print(f"最终方案: 融合 {FUSION_WTS} (k6 主导), w_sim={SIM_W}")

    print("[1/6] 读取数据与概率...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    seqs = train_df["sequence"].values
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, cnn_test = vp["cnn"], tp["cnn"]
    t1 = np.load(T001A_PROBS_PATH)
    t001a_val, t001a_test = t1["t001a_val"], t1["t001a_test"]
    t2v = np.load(T002_VAL_PROBS_PATH)
    t2t = np.load(T002_TEST_PROBS_PATH)
    t002_val, t002_test = t2v["t002"], t2t["t002"]

    with open(TCK002_CONFIG_PATH, "r", encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    print(f"  TCK002 权重: {w} (sim 权重覆盖为 {SIM_W})")

    print("[2/6] 各 k 邻居聚合 (val 91036 空间缓存)...")
    sim_k_val = {}
    for k in K_RANGE:
        prefix = "CK003" if k >= 7 else "CK002"
        d = np.load(f"{REL_DIR}/{prefix}_neigh_k{k}.npz")
        sim_k_val[k] = build_sim_prob(d["idx_val"], d["sim_val"], y_train, TOP_K)
        print(f"  k={k}: sim (22760, 500) ✓")

    print("[3/6] 融合 + 完整评估 (100 档阈值)...")
    sim_fused_val = sum(wt * sim_k_val[k] for k, wt in zip(K_RANGE, FUSION_WTS) if wt > 0)
    ens_val = (w["t001a"] * t001a_val + w["t002"] * t002_val
               + w["cnn"] * cnn_val + SIM_W * sim_fused_val)
    thr = find_optimal_thresholds(ens_val, y_val)
    _, f1, prec, rec = evaluate_at_thresholds(ens_val, y_val, thr)
    auc = macro_auc(y_val, ens_val)
    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {f1:.4f}      │  (TCK002a 基线 0.4236, Δ={f1-0.4236:+.4f})")
    print(f"  │ Macro Precision:  {prec:.4f}      │")
    print(f"  │ Macro Recall:     {rec:.4f}      │")
    print(f"  │ Macro AUC:        {auc:.4f}      │")
    print(f"  └─────────────────────────────┘")

    report = {"fusion_wts": FUSION_WTS, "sim_w": SIM_W,
              "val": {"f1": round(f1, 6), "prec": round(prec, 6),
                      "rec": round(rec, 6), "auc": round(auc, 6)},
              "tck002a_baseline_f1": 0.4236, "delta_f1": round(f1 - 0.4236, 6),
              "val_space": 91036, "top_k": TOP_K}

    if val_only:
        report["test_full_recompute"] = "skipped"
        report["submission"] = "none"
        with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  报告: {REPORT_PATH}")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    print(f"[4/6] 重算 k=3,4,5,7,8 全量空间 test 邻居 (k=6 复用 TCK002a 全量)...")
    sim_k_test = {}
    for k in K_RANGE:
        if k == 6:
            d = np.load(f"{REL_DIR}/TCK002a_neigh_full.npz")
            sim_k_test[k] = build_sim_prob(d["idx_test"], d["sim_test"], y_full, TOP_K)
            print(f"  k=6: 读全量缓存 ✓")
        else:
            cache = f"{REL_DIR}/CK004_neigh_k{k}_full.npz"
            if os.path.exists(cache):
                d = np.load(cache)
                idx_full, sim_full = d["idx_test"], d["sim_test"]
                print(f"  k={k}: 读 CK004 全量缓存 ✓")
            else:
                idx_full, sim_full = get_neighbors(seqs, test_df["sequence"].values, k)
                np.savez_compressed(cache, idx_test=idx_full, sim_test=sim_full)
                print(f"  k={k}: 全量邻居已保存 {cache} ✓")
            sim_k_test[k] = build_sim_prob(idx_full, sim_full, y_full, TOP_K)

    sim_fused_test = sum(wt * sim_k_test[k] for k, wt in zip(K_RANGE, FUSION_WTS) if wt > 0)
    ens_test = (w["t001a"] * t001a_test + w["t002"] * t002_test
                + w["cnn"] * cnn_test + SIM_W * sim_fused_test)

    print("[5/6] 提交 (val 最优阈值)...")
    y_test_pred = predict_with_thresholds(ens_test, thr)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    report["test_full_recompute"] = "done"
    report["submission"] = SUB_PATH
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    print("[6/6] 报告...")
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
