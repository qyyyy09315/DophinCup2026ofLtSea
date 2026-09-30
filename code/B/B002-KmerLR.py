
"""
basl-KmerLR — 基线模型：氨基酸 k-mer TF-IDF + 逻辑回归逐标签分类器
================================================================
特征: k=3 氨基酸三肽 → HashingVectorizer → TF-IDF (8192维, 稀疏)
模型: SGDClassifier(loss='log_loss') ≈ 逻辑回归, class_weight='balanced'
策略: Binary Relevance — 每个标签独立训练

输出:
  submit/submission_KmerLR.csv    — 测试集预测
  rel/KmerLR_val_metrics.json     — 验证集指标
  models/basl-KmerLR.joblib       — 完整模型 (向量器+TFIDF+500×SGD)
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ── 常量 ──────────────────────────────────────────────
KMER_K = 3
HASH_FEATURES = 8192
MAX_ITER = 1000
MIN_POS = 10
RANDOM_STATE = 42
VAL_SIZE = 0.2

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{SUB_DIR}/submission_KmerLR.csv"
METRICS_PATH = f"{REL_DIR}/KmerLR_val_metrics.json"
MODEL_PATH = f"{MODEL_DIR}/basl-KmerLR.joblib"


def safe_metric(fn, y_true, y_pred):
    if y_true.sum() == 0:
        return np.nan
    return fn(y_true, y_pred, zero_division=0)


def main():
    t_start = time.time()

    # 1. 读取数据
    print("=" * 60)
    print("[1/6] 读取数据...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train.columns if c.startswith("label_")]
    print(f"  训练集: {len(train)} 条, 测试集: {len(test)} 条")
    print(f"  标签数: {len(label_cols)}")

    # 2. k-mer Hashing + TF-IDF
    print(f"[2/6] 提取 k={KMER_K} TF-IDF 特征 (Hashing, {HASH_FEATURES}维)...")
    hv = HashingVectorizer(
        analyzer="char",
        ngram_range=(KMER_K, KMER_K),
        n_features=HASH_FEATURES,
        norm=None,
        alternate_sign=False,
        dtype=np.float32,
    )
    tfidf = TfidfTransformer(sublinear_tf=True)
    X_counts = hv.fit_transform(train["sequence"])
    X_train_full = tfidf.fit_transform(X_counts)
    X_test = tfidf.transform(hv.transform(test["sequence"]))
    sparsity = 1 - X_train_full.nnz / (X_train_full.shape[0] * X_train_full.shape[1])
    print(f"  特征维度: {X_train_full.shape[1]}, 稀疏度: {sparsity:.1%}")

    # 3. 划分训练/验证集
    print("[3/6] 划分训练/验证集 (80/20)...")
    y_full = train[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train)), test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    X_tr, X_val = X_train_full[idx_train], X_train_full[idx_val]
    y_tr, y_val = y_full[idx_train], y_full[idx_val]
    print(f"  训练折: {len(idx_train)} 条, 验证折: {len(idx_val)} 条")

    # 4. 逐标签训练 SGD (log_loss ≈ LR)
    print("[4/6] 逐标签训练 SGDClassifier (log_loss)...")
    y_val_pred = np.zeros_like(y_val, dtype=np.int8)
    y_test_pred = np.zeros((len(test), len(label_cols)), dtype=np.int8)
    models = {}
    skipped = 0
    t_train = time.time()

    for i, col in enumerate(label_cols):
        y_col = y_tr[:, i]

        if y_col.sum() < MIN_POS:
            y_val_pred[:, i] = 0
            y_test_pred[:, i] = 0
            skipped += 1
        else:
            sgd = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=MAX_ITER,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            sgd.fit(X_tr, y_col)
            models[col] = sgd
            y_val_pred[:, i] = sgd.predict(X_val)
            y_test_pred[:, i] = sgd.predict(X_test)

        if (i + 1) % 100 == 0 or i == 0:
            el = time.time() - t_train
            eta = el / (i + 1) * (len(label_cols) - i - 1) if i > 0 else 0
            print(f"  [{i+1}/{len(label_cols)}] 耗时={el:.0f}s, 预计剩余={eta:.0f}s, 已跳过={skipped}")

    train_time = time.time() - t_train
    n_trained = len(models)
    print(f"  训练完成, 耗时: {train_time:.0f}s, 训练: {n_trained}, 跳过: {skipped}")

    # 5. 保存模型
    print("[5/6] 保存模型...")
    bundle = {
        "vectorizer": hv,
        "tfidf": tfidf,
        "models": models,
        "label_cols": label_cols,
        "hyperparams": {
            "kmer_k": KMER_K,
            "hash_features": HASH_FEATURES,
            "max_iter": MAX_ITER,
            "class_weight": "balanced",
            "loss": "log_loss",
            "alpha": 1e-4,
        },
    }
    dump(bundle, MODEL_PATH, compress=3)
    model_size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
    print(f"  模型已保存: {MODEL_PATH} ({model_size_mb:.1f} MB)")

    # 6. 评估 + 保存结果
    print("[6/6] 验证集评估...")
    per_label_f1, per_label_precision, per_label_recall = [], [], []
    for i in range(len(label_cols)):
        f1 = safe_metric(f1_score, y_val[:, i], y_val_pred[:, i])
        prec = safe_metric(precision_score, y_val[:, i], y_val_pred[:, i])
        rec = safe_metric(recall_score, y_val[:, i], y_val_pred[:, i])
        if not np.isnan(f1):
            per_label_f1.append(f1)
        if not np.isnan(prec):
            per_label_precision.append(prec)
        if not np.isnan(rec):
            per_label_recall.append(rec)

    macro_f1 = np.mean(per_label_f1)
    macro_precision = np.mean(per_label_precision)
    macro_recall = np.mean(per_label_recall)
    subset_acc = accuracy_score(y_val, y_val_pred)
    hamming_loss = (y_val != y_val_pred).mean()
    total_pos_true = int(y_val.sum())
    total_pos_pred = int(y_val_pred.sum())

    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_precision:.4f}      │")
    print(f"  │ Macro Recall:     {macro_recall:.4f}      │")
    print(f"  │ Subset Accuracy:  {subset_acc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming_loss:.4f}      │")
    print(f"  │ 验证集正样本数:    {total_pos_true}        │")
    print(f"  │ 预测正样本数:      {total_pos_pred}        │")
    print(f"  └─────────────────────────────┘")

    submission = test[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交文件: {SUB_PATH} ({len(submission)} 行)")

    metrics = {
        "model": "basl-KmerLR",
        "features": f"k={KMER_K} hashed TF-IDF, {HASH_FEATURES} dims",
        "classifier": "SGDClassifier (log_loss ≈ LR, Binary Relevance)",
        "hyperparams": {
            "kmer_k": KMER_K,
            "hash_features": HASH_FEATURES,
            "max_iter": MAX_ITER,
            "loss": "log_loss",
            "alpha": 1e-4,
            "penalty": "l2",
            "class_weight": "balanced",
            "min_positive_samples": MIN_POS,
            "val_size": VAL_SIZE,
            "random_state": RANDOM_STATE,
        },
        "train_samples": len(idx_train),
        "val_samples": len(idx_val),
        "n_labels_total": len(label_cols),
        "n_labels_trained": n_trained,
        "n_labels_skipped": skipped,
        "train_time_seconds": round(train_time, 1),
        "total_time_seconds": round(time.time() - t_start, 1),
        "val_macro_f1": round(macro_f1, 6),
        "val_macro_precision": round(macro_precision, 6),
        "val_macro_recall": round(macro_recall, 6),
        "val_subset_accuracy": round(subset_acc, 6),
        "val_hamming_loss": round(hamming_loss, 6),
        "val_total_positive_true": total_pos_true,
        "val_total_positive_pred": total_pos_pred,
        "val_pred_positive_ratio": round(y_val_pred.mean(), 6),
    }
    with Path(METRICS_PATH).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  指标文件: {METRICS_PATH}")

    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
