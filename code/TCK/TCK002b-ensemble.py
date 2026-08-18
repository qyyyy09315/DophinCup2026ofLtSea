
"""
TCK002b-ensemble — 标签共现校正 (InterLabelGO+ 思路)
======================================================
方案:
  - 在 TCK002a 集成概率 (四路加权, 测试集全量邻居空间) 上做标签依赖校正
  - 共现矩阵: 从训练集 (91,036) 计算 C[i][j] = 标签 i,j 共现次数
    (去对角, 行归一化 → 条件概率 P(j|i); 共现 < MIN_COOC 的稀疏化)
  - 传播: p' = (1-α)·p + α·(p @ P), 一次迭代
  - α ∈ {0.1..0.5} 固定阈值 0.5 F1 选权 → 最优 α 做 100 档阈值搜索 → 提交
输出:
  submit/submission_TCK002b_ensemble.csv
  rel/TCK002b_ensemble_val_metrics.json
  models/tck002b-ensemble.json
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{SUB_DIR}/submission_TCK002b_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/TCK002b_ensemble_val_metrics.json"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/tck002b-ensemble.json"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
NEIGH_K6_PATH = f"{REL_DIR}/CK002_neigh_k6.npz"          # 验证集邻居 (91,036 空间)
NEIGH_FULL_PATH = f"{REL_DIR}/TCK002a_neigh_full.npz"    # 测试集邻居 (113,796 空间)
T001A_PROBS_PATH = f"{REL_DIR}/TCK001_t001a_probs.npz"
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"
TCK002_CONFIG_PATH = f"{MODEL_DIR}/tck002-ensemble.json"  # 权重

K = 6
TOP_K = 3
AGG = "sim_weighted"
MIN_COOC = 5          # 共现次数阈值, 低于则视为噪声
ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5]
THRESHOLD_STEPS = 100
VAL_SIZE = 0.2
RANDOM_STATE = 42


def macro_auc(y_true, y_prob):
    aucs = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, y_prob[:, i]))
    return float(np.mean(aucs))


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


def find_optimal_thresholds(y_prob, y_true):
    print("  搜索逐标签最优阈值 (100 档)...")
    n_labels = y_true.shape[1]
    best_thresholds = np.full(n_labels, 0.5, dtype=np.float32)
    candidates = np.linspace(0.01, 0.99, THRESHOLD_STEPS)
    for i in range(n_labels):
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


def predict_with_thresholds(y_prob, thresholds):
    preds = np.zeros_like(y_prob, dtype=np.int8)
    for i, t in enumerate(thresholds):
        preds[:, i] = (y_prob[:, i] >= t).astype(np.int8)
    return preds


def build_sim_prob(neigh_idx, neigh_sim, y_train, top_k, mode):
    n = neigh_idx.shape[0]
    labels = y_train[neigh_idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    if mode == "mean":
        return labels.mean(axis=1)
    w = np.maximum(neigh_sim[:, :top_k], 1e-7)
    return (labels * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]


def build_cooc_matrix(y_train):
    """标签共现条件概率矩阵 P (i → j), 去对角 + 稀疏化。"""
    C = y_train.T @ y_train          # (500, 500) 共现计数
    np.fill_diagonal(C, 0)
    row_sum = C.sum(axis=1, keepdims=True)
    P = np.zeros_like(C, dtype=np.float32)
    valid = row_sum[:, 0] > 0
    P[valid] = C[valid] / np.maximum(row_sum[valid], 1)
    P[C < MIN_COOC] = 0.0            # 低共现视为噪声
    return P


def propagate(p, P, alpha):
    """一次传播: p' = (1-α)·p + α·(p @ P)。"""
    return (1.0 - alpha) * p + alpha * (p @ P)


def main():
    t0 = time.time()
    print("=" * 60)

    print("[1/6] 读取数据与概率...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    seqs = train_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    with open(TCK002_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    w = cfg["best_w"]
    print(f"  复用 TCK002 权重: {w}")

    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, val_true = vp["cnn"], vp["y_true"]
    cnn_test = tp["cnn"]

    d6 = np.load(NEIGH_K6_PATH)
    sim_val = build_sim_prob(d6["idx_val"], d6["sim_val"], y_train, TOP_K, AGG)
    dfull = np.load(NEIGH_FULL_PATH)
    sim_test = build_sim_prob(dfull["idx_test"], dfull["sim_test"], y_full, TOP_K, AGG)

    t1 = np.load(T001A_PROBS_PATH)
    t001a_val, t001a_test = t1["t001a_val"], t1["t001a_test"]
    t2v = np.load(T002_VAL_PROBS_PATH)
    t2t = np.load(T002_TEST_PROBS_PATH)
    t002_val, t002_test = t2v["t002"], t2t["t002"]

    ens_val = (w["t001a"] * t001a_val + w["t002"] * t002_val
               + w["cnn"] * cnn_val + w["sim"] * sim_val)
    ens_test = (w["t001a"] * t001a_test + w["t002"] * t002_test
                + w["cnn"] * cnn_test + w["sim"] * sim_test)
    print(f"  集成概率: val={ens_val.shape}, test={ens_test.shape}")

    print("[2/6] 构建标签共现条件概率矩阵...")
    P = build_cooc_matrix(y_train)
    nnz = (P > 0).sum()
    print(f"  P: {P.shape}, 非零 {nnz} ({nnz / P.size:.1%}), 每标签平均非零 {nnz / 500:.1f}")

    print("[3/6] α 网格搜索 (固定阈值 0.5 F1)...")
    best_alpha, best_f1 = 0.1, -1.0
    for alpha in ALPHA_GRID:
        pv = propagate(ens_val, P, alpha)
        f1 = macro_f1_fixed_threshold(pv, val_true)
        print(f"  α={alpha:.1f}: F1(0.5)={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_alpha = f1, alpha
    print(f"  最优 α={best_alpha:.1f}, F1(0.5)={best_f1:.4f}")
    print(f"  (传播前基线 F1(0.5)={macro_f1_fixed_threshold(ens_val, val_true):.4f})")

    print("[4/6] 最优 α 阈值搜索...")
    pv = propagate(ens_val, P, best_alpha)
    pt = propagate(ens_test, P, best_alpha)
    opt_thresholds = find_optimal_thresholds(pv, val_true)
    print(f"  阈值: mean={opt_thresholds.mean():.3f}, <0.5: {(opt_thresholds < 0.5).sum()}/{len(label_cols)}")

    print("[5/6] 评估...")
    y_val_pred, macro_f1, macro_prec, macro_rec = evaluate_at_thresholds(pv, val_true, opt_thresholds)
    auc = macro_auc(val_true, pv)
    hamming = (val_true != y_val_pred).mean()
    total_pos_true = int(val_true.sum())
    total_pos_pred = int(y_val_pred.sum())
    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_prec:.4f}      │")
    print(f"  │ Macro Recall:     {macro_rec:.4f}      │")
    print(f"  │ Macro AUC:        {auc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming:.4f}      │")
    print(f"  │ 预测/真实:        {total_pos_pred}/{total_pos_true}      │")
    print(f"  └─────────────────────────────┘")

    print("[6/6] 测试集预测并保存...")
    y_test_pred = predict_with_thresholds(pt, opt_thresholds)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    config = {
        "model": "tck002b-ensemble",
        "note": "label co-occurrence propagation on TCK002a probs (InterLabelGO-style)",
        "min_cooc": MIN_COOC,
        "alpha": round(float(best_alpha), 4),
        "thresholds": opt_thresholds.tolist(),
        "label_cols": label_cols,
    }
    with open(ENSEMBLE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": "tck002b-ensemble",
        "strategy": "co-occurrence propagation p' = (1-a)p + a(p@P)",
        "min_cooc": MIN_COOC,
        "alpha": round(float(best_alpha), 4),
        "f1_05_before": round(macro_f1_fixed_threshold(ens_val, val_true), 6),
        "f1_05_after": round(best_f1, 6),
        "threshold_stats": {
            "min": round(float(opt_thresholds.min()), 4),
            "max": round(float(opt_thresholds.max()), 4),
            "mean": round(float(opt_thresholds.mean()), 4),
            "n_below_05": int((opt_thresholds < 0.5).sum()),
        },
        "val_macro_f1": round(macro_f1, 6),
        "val_macro_precision": round(macro_prec, 6),
        "val_macro_recall": round(macro_rec, 6),
        "val_macro_auc": round(auc, 6),
        "val_hamming_loss": round(hamming, 6),
        "val_total_positive_true": total_pos_true,
        "val_total_positive_pred": total_pos_pred,
        "val_pred_positive_ratio": round(y_val_pred.mean(), 6),
        "total_time_seconds": round(time.time() - t0, 1),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  配置: {ENSEMBLE_CONFIG_PATH}, 指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
