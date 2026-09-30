"""
protbert-cls — ProtBERT 嵌入 + 逻辑回归逐标签分类
====================================================
输入: rel/train_embeddings.parquet, rel/test_embeddings.parquet
模型: LogisticRegression (liblinear, class_weight=balanced), Binary Relevance
输出:
  submit/submission_ProtBERT.csv   — 测试集预测
  rel/ProtBERT_val_metrics.json    — 验证集指标
  models/protbert-cls.joblib       — 分类模型
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, accuracy_score)
from sklearn.model_selection import train_test_split
from joblib import dump, Parallel, delayed
from tqdm import tqdm
from pathlib import Path

REL_DIR = "rel"
SUB_DIR = "submit"
MODEL_DIR = "models"
TRAIN_EMB_PATH = f"{REL_DIR}/train_embeddings.parquet"
TEST_EMB_PATH = f"{REL_DIR}/test_embeddings.parquet"
SUB_PATH = f"{SUB_DIR}/submission_ProtBERT.csv"
METRICS_PATH = f"{REL_DIR}/ProtBERT_val_metrics.json"
MODEL_PATH = f"{MODEL_DIR}/protbert-cls.joblib"

MIN_POS = 10
RANDOM_STATE = 42
VAL_SIZE = 0.2
MAX_ITER = 1000
N_JOBS = 6              # 不全用 16 核，避免卡顿（实测 8 线程仅 2.2x 加速）

EMB_COLS = [f"emb_{i}" for i in range(1024)]


def safe_metric(fn, y_true, y_pred):
    if y_true.sum() == 0:
        return np.nan
    return fn(y_true, y_pred, zero_division=0)


def main():
    t_start = time.time()
    print("=" * 60)

    print("[1/5] 读取嵌入 parquet...")
    train = pd.read_parquet(TRAIN_EMB_PATH)
    test = pd.read_parquet(TEST_EMB_PATH)
    label_cols = [c for c in train.columns if c.startswith("label_")]
    print(f"  train: {train.shape}, test: {test.shape}, labels: {len(label_cols)}")

    X_full = train[EMB_COLS].values.astype(np.float32)
    y_full = train[label_cols].values.astype(np.int8)
    X_test = test[EMB_COLS].values.astype(np.float32)
    print(f"  X_train: {X_full.shape}, X_test: {X_test.shape}")

    print("[2/5] 划分训练/验证集 (80/20)...")
    idx_train, idx_val = train_test_split(
        np.arange(len(train)), test_size=VAL_SIZE, random_state=RANDOM_STATE)
    X_tr, X_val = X_full[idx_train], X_full[idx_val]
    y_tr, y_val = y_full[idx_train], y_full[idx_val]
    print(f"  训练: {len(idx_train)}, 验证: {len(idx_val)}")

    print("[3/5] 逐标签训练 LogisticRegression (并行)...")

    def fit_label(i, col):
        y_col = y_tr[:, i]
        if y_col.sum() < MIN_POS:
            return i, col, None, None, None, None
        lr = LogisticRegression(
            max_iter=MAX_ITER, class_weight="balanced",
            solver="liblinear", random_state=RANDOM_STATE)
        lr.fit(X_tr, y_col)
        return (i, col, lr,
                lr.predict(X_val),
                lr.predict_proba(X_val)[:, 1],
                lr.predict(X_test))

    y_val_pred = np.zeros_like(y_val, dtype=np.int8)
    y_test_pred = np.zeros((len(test), len(label_cols)), dtype=np.int8)
    y_val_prob = np.zeros((len(idx_val), len(label_cols)), dtype=np.float32)
    models = {}
    skipped = 0
    t_train = time.time()

    results = Parallel(n_jobs=N_JOBS, backend="threading", return_as="generator")(
        delayed(fit_label)(i, col) for i, col in enumerate(label_cols))
    pbar = tqdm(results, total=len(label_cols), desc="训练标签", unit="label",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for i, col, lr, vp, vprob, tp in pbar:
        if lr is None:
            skipped += 1
        else:
            models[col] = lr
            y_val_pred[:, i] = vp
            y_val_prob[:, i] = vprob
            y_test_pred[:, i] = tp
        pbar.set_postfix(skip=skipped)

    train_time = time.time() - t_train
    print(f"  训练完成: {train_time:.0f}s, 训练标签: {len(models)}, 跳过: {skipped}")

    print("[4/5] 验证集评估...")
    per_f1, per_prec, per_rec, per_auc = [], [], [], []
    for i in range(len(label_cols)):
        if y_val[:, i].sum() == 0:
            continue
        per_f1.append(safe_metric(f1_score, y_val[:, i], y_val_pred[:, i]))
        per_prec.append(safe_metric(precision_score, y_val[:, i], y_val_pred[:, i]))
        per_rec.append(safe_metric(recall_score, y_val[:, i], y_val_pred[:, i]))
        per_auc.append(roc_auc_score(y_val[:, i], y_val_prob[:, i]))

    macro_f1 = np.mean(per_f1)
    macro_precision = np.mean(per_prec)
    macro_recall = np.mean(per_rec)
    macro_auc = np.mean(per_auc)
    subset_acc = accuracy_score(y_val, y_val_pred)
    hamming_loss = (y_val != y_val_pred).mean()
    total_pos_true = int(y_val.sum())
    total_pos_pred = int(y_val_pred.sum())

    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_precision:.4f}      │")
    print(f"  │ Macro Recall:     {macro_recall:.4f}      │")
    print(f"  │ Macro AUC:        {macro_auc:.4f}      │")
    print(f"  │ Subset Accuracy:  {subset_acc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming_loss:.4f}      │")
    print(f"  │ 预测/真实:        {total_pos_pred}/{total_pos_true}      │")
    print(f"  └─────────────────────────────┘")

    print("[5/5] 保存结果...")
    dump({"models": models, "label_cols": label_cols}, MODEL_PATH, compress=3)
    print(f"  模型: {MODEL_PATH}")

    submission = test[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    metrics = {
        "model": "protbert-cls",
        "features": "ProtBERT CLS embedding (1024d, fp16 parquet)",
        "classifier": "LogisticRegression (Binary Relevance)",
        "hyperparams": {
            "solver": "liblinear", "max_iter": MAX_ITER,
            "n_jobs": N_JOBS, "backend": "threading",
            "class_weight": "balanced", "min_positive_samples": MIN_POS,
            "val_size": VAL_SIZE, "random_state": RANDOM_STATE,
        },
        "train_samples": len(idx_train), "val_samples": len(idx_val),
        "n_labels_total": len(label_cols), "n_labels_trained": len(models),
        "n_labels_skipped": skipped,
        "train_time_seconds": round(train_time, 1),
        "total_time_seconds": round(time.time() - t_start, 1),
        "val_macro_f1": round(macro_f1, 6),
        "val_macro_precision": round(macro_precision, 6),
        "val_macro_recall": round(macro_recall, 6),
        "val_macro_auc": round(macro_auc, 6),
        "val_subset_accuracy": round(subset_acc, 6),
        "val_hamming_loss": round(hamming_loss, 6),
        "val_total_positive_true": total_pos_true,
        "val_total_positive_pred": total_pos_pred,
    }
    with Path(METRICS_PATH).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  指标: {METRICS_PATH}")

    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
