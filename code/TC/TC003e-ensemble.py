
"""
TC003e-ensemble — 标签级权重集成 (C002b + B007, 每标签独立最优 w)
=====================================================================
方案:
  - 对照 TC003 (全局单一 w): 权重粒度细化到每个标签
  - 每个标签在 11 档权重网格上独立搜索最优 w, 并在该 w 下做
    50 档阈值搜索 (w 与阈值联合优化, 自由度最大)
  - 预期: 本地 F1 最高 (过拟合风险也最大) —— 用于验证
    "本地越高提交降幅越大" 现象是否由阈值/权重过拟合主导
输出:
  submit/submission_TC003e_ensemble.csv
  rel/TC003e_ensemble_val_metrics.json
  models/tc003e-ensemble.json
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
SUB_PATH = f"{SUB_DIR}/submission_TC003e_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/TC003e_ensemble_val_metrics.json"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/tc003e-ensemble.json"

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


def main():
    t_start = time.time()
    print("=" * 60)

    print("[1/5] 读取概率缓存...")
    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, tf_val, y_true = vp["cnn"], vp["transformer"], vp["y_true"]
    cnn_test, tf_test = tp["cnn"], tp["transformer"]
    n_labels = y_true.shape[1]
    print(f"  val: {cnn_val.shape}, test: {cnn_test.shape}")

    print("[2/5] 逐标签权重+阈值联合搜索 (11 w x 50 t)...")
    weights = np.linspace(0.0, 1.0, WEIGHT_STEPS)
    candidates = np.linspace(0.01, 0.99, THRESHOLD_STEPS)
    best_w = np.full(n_labels, 0.5, dtype=np.float32)
    best_thr = np.full(n_labels, 0.5, dtype=np.float32)
    n_done = 0
    for i in range(n_labels):
        yt = y_true[:, i]
        if yt.sum() == 0:
            continue
        p1, p2 = cnn_val[:, i], tf_val[:, i]
        best_f1, bw, bt = 0, 0.5, 0.5
        for w in weights:
            ens_i = w * p1 + (1.0 - w) * p2
            for t in candidates:
                pred = (ens_i >= t).astype(int)
                f1 = f1_score(yt, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, bw, bt = f1, w, t
        best_w[i], best_thr[i] = bw, bt
        n_done += 1
    print(f"  {n_done}/{n_labels} 标签完成")
    print(f"  权重分布: mean={best_w.mean():.3f}, <0.5: {(best_w < 0.5).sum()}, =1.0: {(best_w >= 1.0).sum()}, =0.0: {(best_w <= 0.0).sum()}")

    print("[3/5] 构建标签级集成概率...")
    # 线性组合天然处理 w=0/1 边界 (0*p1+1*p2=p2, 1*p1+0*p2=p1)
    ens_val = best_w[None, :] * cnn_val + (1.0 - best_w[None, :]) * tf_val
    ens_test = best_w[None, :] * cnn_test + (1.0 - best_w[None, :]) * tf_test

    print("[4/5] 评估...")
    y_val_pred, macro_f1, macro_prec, macro_rec = evaluate_at_thresholds(ens_val, y_true, best_thr)
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
    y_test_pred = predict_with_thresholds(ens_test, best_thr)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    config = {
        "model": "tc003e-ensemble",
        "strategy": "per-label weight + threshold joint search",
        "best_w": best_w.tolist(),
        "thresholds": best_thr.tolist(),
        "label_cols": label_cols,
    }
    with Path(ENSEMBLE_CONFIG_PATH).open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": "tc003e-ensemble",
        "features": "per-label weight: cnn-c002b + transformer-cls",
        "weight_stats": {
            "mean": round(float(best_w.mean()), 4),
            "n_lt_05": int((best_w < 0.5).sum()),
            "n_eq_1": int((best_w >= 1.0).sum()),
            "n_eq_0": int((best_w <= 0.0).sum()),
        },
        "threshold_stats": {
            "min": round(float(best_thr.min()), 4),
            "max": round(float(best_thr.max()), 4),
            "mean": round(float(best_thr.mean()), 4),
            "n_below_05": int((best_thr < 0.5).sum()),
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
