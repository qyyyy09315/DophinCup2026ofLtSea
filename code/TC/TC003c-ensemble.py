
"""
TC003c-ensemble — Stacking 元学习集成 (C002b + B007 → 逐标签逻辑回归)
======================================================================
方案:
  - 对照 TC003 (固定加权平均): 用学习式组合替代人工权重
  - 第一层: C002b / B007 概率 (复用 TC003 缓存)
  - 第二层: 对每个标签独立拟合 LogisticRegression([p_cnn, p_b007] -> y)
    (500 个小模型, sklearn, 分钟级)
  - 输出概率 = LR 预测概率, 再逐标签阈值搜索 (50 档)
  - 备注: 验证集上拟合 + 评估, LR 容量小过拟合风险有限 (与阈值搜索同性质)
输出:
  submit/submission_TC003c_ensemble.csv
  rel/TC003c_ensemble_val_metrics.json
  models/tc003c-ensemble.json
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
SUB_PATH = f"{SUB_DIR}/submission_TC003c_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/TC003c_ensemble_val_metrics.json"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/tc003c-ensemble.json"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"

LR_C = 1.0
THRESHOLD_STEPS = 50
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
    print("  搜索逐标签最优阈值...")
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


def main():
    t_start = time.time()
    print("=" * 60)

    print("[1/5] 读取概率缓存...")
    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, tf_val, y_true = vp["cnn"], vp["transformer"], vp["y_true"]
    cnn_test, tf_test = tp["cnn"], tp["transformer"]
    X_val = np.stack([cnn_val, tf_val], axis=-1)     # (N, 500, 2)
    X_test = np.stack([cnn_test, tf_test], axis=-1)  # (N, 500, 2)
    print(f"  val: {cnn_val.shape}, test: {cnn_test.shape}, 特征: [p_cnn, p_b007]")

    print("[2/5] 逐标签 Stacking 逻辑回归拟合...")
    n_labels = y_true.shape[1]
    stack_val = np.zeros_like(cnn_val, dtype=np.float32)
    stack_test = np.zeros_like(cnn_test, dtype=np.float32)
    for i in range(n_labels):
        lr = LogisticRegression(C=LR_C, max_iter=500)
        lr.fit(X_val[:, i, :], y_true[:, i])
        stack_val[:, i] = lr.predict_proba(X_val[:, i, :])[:, 1]
        stack_test[:, i] = lr.predict_proba(X_test[:, i, :])[:, 1]
    print(f"  stacking 完成: val={stack_val.shape}, test={stack_test.shape}")

    print("[3/5] 阈值搜索...")
    opt_thresholds = find_optimal_thresholds(stack_val, y_true)
    print(f"  阈值: mean={opt_thresholds.mean():.3f}, <0.5: {(opt_thresholds < 0.5).sum()}/{n_labels}")

    print("[4/5] 评估...")
    y_val_pred, macro_f1, macro_prec, macro_rec = evaluate_at_thresholds(stack_val, y_true, opt_thresholds)
    auc = macro_auc(y_true, stack_val)
    subset_acc = accuracy_score(y_true, y_val_pred)
    hamming = (y_true != y_val_pred).mean()
    total_pos_true = int(y_true.sum())
    total_pos_pred = int(y_val_pred.sum())
    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_prec:.4f}      │")
    print(f"  │ Macro Recall:     {macro_rec:.4f}      │")
    print(f"  │ Macro AUC:        {auc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming:.4f}      │")
    print(f"  │ 预测/真实:        {total_pos_pred}/{total_pos_true}      │")
    print(f"  └─────────────────────────────┘")

    print("[5/5] 测试集预测并保存...")
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in pd.read_csv(TRAIN_PATH, nrows=1).columns if c.startswith("label_")]
    os.makedirs(SUB_DIR, exist_ok=True)
    y_test_pred = predict_with_thresholds(stack_test, opt_thresholds)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    config = {
        "model": "tc003c-ensemble",
        "strategy": "stacking: per-label LogisticRegression on [p_cnn, p_b007]",
        "lr_c": LR_C,
        "thresholds": opt_thresholds.tolist(),
        "label_cols": label_cols,
    }
    with open(ENSEMBLE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": "tc003c-ensemble",
        "features": "stacking: LogisticRegression([p_cnn, p_b007]) per label",
        "lr_c": LR_C,
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
        "val_subset_accuracy": round(subset_acc, 6),
        "val_hamming_loss": round(hamming, 6),
        "val_total_positive_true": total_pos_true,
        "val_total_positive_pred": total_pos_pred,
        "val_pred_positive_ratio": round(y_val_pred.mean(), 6),
        "total_time_seconds": round(time.time() - t_start, 1),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  配置: {ENSEMBLE_CONFIG_PATH}, 指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
