
"""
TC003d-ensemble — 几何平均集成 (C002b + B007, p^w * q^(1-w))
================================================================
方案:
  - 对照 TC003 (算术平均): 聚合函数改为加权几何平均
  - 几何平均对两个模型的低置信度敏感: 只有两模型都高时才高,
    稀疏多标签场景下预期提升精度
  - 权重网格 11 档, 选权指标 = 固定阈值 0.5 的 Macro F1
  - 选定权重后逐标签阈值搜索 (50 档)
输出:
  submit/submission_TC003d_ensemble.csv
  rel/TC003d_ensemble_val_metrics.json
  models/tc003d-ensemble.json
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score)
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{SUB_DIR}/submission_TC003d_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/TC003d_ensemble_val_metrics.json"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/tc003d-ensemble.json"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"

WEIGHT_STEPS = 11
THRESHOLD_STEPS = 50


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


def geom_ens(p, q, w):
    """加权几何平均, 数值稳定: exp(w*log p + (1-w)*log q), 概率 0 处 clamp。"""
    lp = np.log(np.clip(p, 1e-7, 1.0))
    lq = np.log(np.clip(q, 1e-7, 1.0))
    return np.exp(w * lp + (1.0 - w) * lq)


def main():
    t_start = time.time()
    print("=" * 60)

    print("[1/5] 读取概率缓存...")
    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, tf_val, y_true = vp["cnn"], vp["transformer"], vp["y_true"]
    cnn_test, tf_test = tp["cnn"], tp["transformer"]
    print(f"  val: {cnn_val.shape}, test: {cnn_test.shape}")

    print("[2/5] 几何平均权重网格搜索 (固定阈值 0.5 Macro F1)...")
    weights = np.linspace(0.0, 1.0, WEIGHT_STEPS)
    best_w, best_f1 = 0.5, -1.0
    for w in weights:
        ens = geom_ens(cnn_val, tf_val, w)
        f1 = macro_f1_fixed_threshold(ens, y_true)
        if f1 > best_f1:
            best_f1, best_w = f1, w
    print(f"  最优 w_cnn={best_w:.2f}, F1(0.5)={best_f1:.4f}")

    print("[3/5] 集成概率阈值搜索...")
    ens_val = geom_ens(cnn_val, tf_val, best_w)
    ens_test = geom_ens(cnn_test, tf_test, best_w)
    opt_thresholds = find_optimal_thresholds(ens_val, y_true)
    print(f"  阈值: mean={opt_thresholds.mean():.3f}, <0.5: {(opt_thresholds < 0.5).sum()}/{y_true.shape[1]}")

    print("[4/5] 评估...")
    y_val_pred, macro_f1, macro_prec, macro_rec = evaluate_at_thresholds(ens_val, y_true, opt_thresholds)
    auc = macro_auc(y_true, ens_val)
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
    y_test_pred = predict_with_thresholds(ens_test, opt_thresholds)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    config = {
        "model": "tc003d-ensemble",
        "strategy": "weighted geometric mean p^w * q^(1-w)",
        "weight_selection_metric": "macro_f1 @ threshold=0.5",
        "best_w_cnn": round(float(best_w), 4),
        "thresholds": opt_thresholds.tolist(),
        "label_cols": label_cols,
    }
    with Path(ENSEMBLE_CONFIG_PATH).open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": "tc003d-ensemble",
        "features": "weighted geometric mean: cnn-c002b + transformer-cls",
        "best_w_cnn": round(float(best_w), 4),
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
    with Path(METRICS_PATH).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  配置: {ENSEMBLE_CONFIG_PATH}, 指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
