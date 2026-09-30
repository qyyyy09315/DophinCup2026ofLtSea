
"""
basl-AARF — 基线模型：氨基酸频率 + 随机森林逐标签分类器
==========================================================
特征: 20种标准氨基酸在序列中的频率 + log1p(序列长度) = 21维
模型: RandomForest (n_estimators=50, max_depth=12, class_weight='balanced')
策略: Binary Relevance — 每个标签独立训练一个二分类器

输出:
  submit/submission_AARF.csv    — 测试集预测，可直接提交
  rel/AARF_val_metrics.json     — 验证集指标 (macro-F1, per-label F1 等)
  rel/AARF_val_pred.csv         — 验证集预测详情 (protein_id + 真实标签 + 预测标签)
"""

import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ── 常量 ──────────────────────────────────────────────
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
N_ESTIMATORS = 50
MAX_DEPTH = 12
MIN_POS = 10          # 训练集中正样本少于该数的标签跳过
RANDOM_STATE = 42
VAL_SIZE = 0.2        # 验证集比例
N_JOBS = -1

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{SUB_DIR}/submission_AARF.csv"
METRICS_PATH = f"{REL_DIR}/AARF_val_metrics.json"
VAL_PRED_PATH = f"{REL_DIR}/AARF_val_pred.csv"

# ── 工具函数 ──────────────────────────────────────────

def extract_aa_features(sequences):
    """从氨基酸序列提取特征: 20维AA频率 + 1维 log1p(长度)"""
    n = len(sequences)
    n_aa = len(AMINO_ACIDS)
    feats = np.zeros((n, n_aa + 1), dtype=np.float32)
    aa_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    for i, seq in enumerate(sequences):
        L = len(seq)
        feats[i, -1] = np.log1p(L)
        if L == 0:
            continue
        for c in seq:
            if c in aa_idx:
                feats[i, aa_idx[c]] += 1
        feats[i, :n_aa] /= L
    return feats


def safe_f1(y_true, y_pred):
    """安全的 F1 计算，全零预测时返回 0"""
    if y_true.sum() == 0:
        return np.nan  # 验证集中该标签无正样本，不计入 macro-F1
    return f1_score(y_true, y_pred, zero_division=0)


def safe_precision(y_true, y_pred):
    if y_true.sum() == 0:
        return np.nan
    return precision_score(y_true, y_pred, zero_division=0)


def safe_recall(y_true, y_pred):
    if y_true.sum() == 0:
        return np.nan
    return recall_score(y_true, y_pred, zero_division=0)


# ── 主流程 ────────────────────────────────────────────

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

    # 2. 特征提取
    print("[2/6] 提取氨基酸频率特征...")
    X_train_full = extract_aa_features(train["sequence"])
    X_test = extract_aa_features(test["sequence"])
    print(f"  特征维度: {X_train_full.shape[1]} (20 AA频率 + 1 长度)")

    # 3. 划分训练/验证集 (stratified split 在多标签场景不太可行，随机分)
    print("[3/6] 划分训练/验证集 (80/20)...")
    y_full = train[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train)), test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    X_tr, X_val = X_train_full[idx_train], X_train_full[idx_val]
    y_tr, y_val = y_full[idx_train], y_full[idx_val]
    print(f"  训练折: {len(idx_train)} 条, 验证折: {len(idx_val)} 条")

    # 4. 逐标签训练
    print("[4/6] 逐标签训练 RandomForest...")
    y_val_pred = np.zeros_like(y_val, dtype=np.int8)
    y_test_pred = np.zeros((len(test), len(label_cols)), dtype=np.int8)
    skipped_labels = []

    t_train = time.time()
    for i, col in enumerate(label_cols):
        y_col = y_tr[:, i]

        if y_col.sum() < MIN_POS:
            skipped_labels.append(col)
            if (i + 1) % 100 == 0 or i == 0:
                el = time.time() - t_train
                eta = el / (i + 1) * (len(label_cols) - i - 1) if i > 0 else 0
                print(f"  [{i+1}/{len(label_cols)}] 耗时={el:.0f}s, 预计剩余={eta:.0f}s")
            continue

        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        rf.fit(X_tr, y_col)

        y_val_pred[:, i] = rf.predict(X_val)
        y_test_pred[:, i] = rf.predict(X_test)

        if (i + 1) % 100 == 0 or i == 0:
            el = time.time() - t_train
            eta = el / (i + 1) * (len(label_cols) - i - 1) if i > 0 else 0
            print(f"  [{i+1}/{len(label_cols)}] 耗时={el:.0f}s, 预计剩余={eta:.0f}s")

    train_time = time.time() - t_train
    print(f"  训练完成, 总耗时: {train_time:.0f}s")
    if skipped_labels:
        print(f"  跳过标签 ({len(skipped_labels)}个正样本不足): {skipped_labels[:5]}..."
              if len(skipped_labels) > 5 else f"  跳过标签: {skipped_labels}")

    # 5. 验证集评估
    print("[5/6] 验证集评估...")
    per_label_f1 = []
    per_label_precision = []
    per_label_recall = []
    for i, col in enumerate(label_cols):
        f1 = safe_f1(y_val[:, i], y_val_pred[:, i])
        prec = safe_precision(y_val[:, i], y_val_pred[:, i])
        rec = safe_recall(y_val[:, i], y_val_pred[:, i])
        if not np.isnan(f1):
            per_label_f1.append(f1)
        if not np.isnan(prec):
            per_label_precision.append(prec)
        if not np.isnan(rec):
            per_label_recall.append(rec)

    macro_f1 = np.mean(per_label_f1)
    macro_precision = np.mean(per_label_precision)
    macro_recall = np.mean(per_label_recall)

    # 子集准确率 (exact match ratio)
    subset_acc = accuracy_score(y_val, y_val_pred)

    # Hamming Loss
    hamming_loss = (y_val != y_val_pred).mean()

    total_pos_pred = y_val_pred.sum()
    total_pos_true = y_val.sum()

    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_precision:.4f}      │")
    print(f"  │ Macro Recall:     {macro_recall:.4f}      │")
    print(f"  │ Subset Accuracy:  {subset_acc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming_loss:.4f}      │")
    print(f"  │ 验证集正样本数:    {total_pos_true}        │")
    print(f"  │ 预测正样本数:      {total_pos_pred}        │")
    print(f"  └─────────────────────────────┘")

    # 6. 保存结果
    print("[6/6] 保存结果...")

    # 提交文件
    submission = test[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交文件: {SUB_PATH} ({len(submission)} 行)")

    # 验证集指标 JSON
    metrics = {
        "model": "basl-AARF",
        "features": "20 AA frequency + log1p(seq_len)",
        "classifier": "RandomForest (Binary Relevance)",
        "hyperparams": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "class_weight": "balanced",
            "min_positive_samples": MIN_POS,
            "val_size": VAL_SIZE,
            "random_state": RANDOM_STATE,
        },
        "train_samples": len(idx_train),
        "val_samples": len(idx_val),
        "n_labels_total": len(label_cols),
        "n_labels_skipped": len(skipped_labels),
        "train_time_seconds": round(train_time, 1),
        "total_time_seconds": round(time.time() - t_start, 1),
        "val_macro_f1": round(macro_f1, 6),
        "val_macro_precision": round(macro_precision, 6),
        "val_macro_recall": round(macro_recall, 6),
        "val_subset_accuracy": round(subset_acc, 6),
        "val_hamming_loss": round(hamming_loss, 6),
        "val_total_positive_true": int(total_pos_true),
        "val_total_positive_pred": int(total_pos_pred),
        "val_pred_positive_ratio": round(y_val_pred.mean(), 6),
    }
    with Path(METRICS_PATH).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  指标文件: {METRICS_PATH}")

    # 验证集预测详情 (前500条采样，避免文件过大)
    val_protein_ids = train.iloc[idx_val]["protein_id"].values
    val_detail_df = pd.DataFrame({"protein_id": val_protein_ids})
    sample_n = min(500, len(idx_val))
    val_detail_df = val_detail_df.iloc[:sample_n].copy()
    for i, col in enumerate(label_cols):
        val_detail_df[f"{col}_true"] = y_val[:sample_n, i]
        val_detail_df[f"{col}_pred"] = y_val_pred[:sample_n, i]
    val_detail_df.to_csv(VAL_PRED_PATH, index=False)
    print(f"  验证集采样: {VAL_PRED_PATH} ({sample_n} 条, 前500行)")

    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")
    print(f"提交文件: {SUB_PATH}")
    print(f"指标文件: {METRICS_PATH}")


if __name__ == "__main__":
    main()
