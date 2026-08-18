
"""
CK001-ensemble — C002b + 序列相似性特征 概率集成
==================================================
方案:
  - 相似性特征 (DeepGOPlus 式): k=3 k-mer hashing (2^18 桶) → L2 归一化
    → cosine 最近邻 (top-3, 邻居空间=训练集) → 邻居标签平均 = sim_prob
  - 分析实验 (sim-feature-analysis.py) 验证: sim 单独 F1 0.3053;
    与 C002b 集成 F1 0.3556 (w_cnn=0.70) — 达到第一名 (35 分) 水平
  - 正式版: 提取验证集+测试集 sim_prob (缓存 rel/CK001_sim_probs.npz,
    重跑秒级), 验证集权重网格 + 阈值搜索, 输出测试集提交
输出:
  submit/submission_CK001_ensemble.csv
  rel/CK001_ensemble_val_metrics.json
  rel/CK001_sim_probs.npz
  models/ck001-ensemble.json
"""

import json
import os
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{SUB_DIR}/submission_CK001_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/CK001_ensemble_val_metrics.json"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/ck001-ensemble.json"
SIM_PROBS_PATH = f"{REL_DIR}/CK001_sim_probs.npz"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"

K = 3
N_FEATURES = 2 ** 18
TOP_K = 3
BLOCK = 500
VAL_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD_STEPS = 50
WEIGHT_STEPS = 11


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


def extract_sim_probs(seqs_train, y_train, seqs_query, query_names, X_train_norm):
    """分块最近邻 → 邻居标签平均 (sim_prob)。"""
    n_q = len(seqs_query)
    neigh_idx = np.zeros((n_q, TOP_K), dtype=np.int64)
    vec = HashingVectorizer(analyzer="char", ngram_range=(K, K),
                            n_features=N_FEATURES, dtype=np.float32)
    X_q = normalize(sparse.csr_matrix(vec.transform(seqs_query)))
    for s in tqdm(range(0, n_q, BLOCK), desc=f"最近邻 ({query_names})"):
        e = min(s + BLOCK, n_q)
        sim = (X_q[s:e] @ X_train_norm.T).toarray()
        neigh_idx[s:e] = np.argpartition(-sim, TOP_K, axis=1)[:, :TOP_K]
    return y_train[neigh_idx.ravel()].reshape(n_q, TOP_K, -1).astype(np.float32).mean(axis=1)


def main():
    t0 = time.time()
    print("=" * 60)

    print("[1/7] 读取数据并划分训练/验证集...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    seqs = train_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    print(f"  训练: {len(idx_train)}, 验证: {len(idx_val)}, 测试: {len(test_df)}")

    if os.path.exists(SIM_PROBS_PATH):
        print("[2/7] 加载缓存的 sim 概率...")
        sp = np.load(SIM_PROBS_PATH)
        sim_val, sim_test = sp["sim_val"], sp["sim_test"]
    else:
        print("[2/7] k-mer hashing 与最近邻 (首次运行, ~13 分钟, 结果缓存)...")
        vec = HashingVectorizer(analyzer="char", ngram_range=(K, K),
                                n_features=N_FEATURES, dtype=np.float32)
        X_train = normalize(sparse.csr_matrix(vec.transform(seqs[idx_train])))
        sim_val = extract_sim_probs(seqs[idx_train], y_train, seqs[idx_val],
                                    "val", X_train)
        sim_test = extract_sim_probs(seqs[idx_train], y_train,
                                     test_df["sequence"].values, "test", X_train)
        os.makedirs(REL_DIR, exist_ok=True)
        np.savez_compressed(SIM_PROBS_PATH, sim_val=sim_val, sim_test=sim_test)
        print(f"  sim 概率已缓存: {SIM_PROBS_PATH}")
    print(f"  sim_val={sim_val.shape}, sim_test={sim_test.shape}")

    print("[3/7] 读取 C002b 概率 (TC003 缓存)...")
    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, val_true = vp["cnn"], vp["y_true"]
    cnn_test = tp["cnn"]
    print(f"  cnn_val={cnn_val.shape}, cnn_test={cnn_test.shape}")

    print("[4/7] 权重网格搜索 (固定阈值 0.5 Macro F1)...")
    weights = np.linspace(0.0, 1.0, WEIGHT_STEPS)
    best_w, best_f1_w = 0.5, -1.0
    for w in weights:
        ens = w * cnn_val + (1.0 - w) * sim_val
        f1 = macro_f1_fixed_threshold(ens, val_true)
        if f1 > best_f1_w:
            best_f1_w, best_w = f1, w
    print(f"  最优权重: w_cnn={best_w:.2f}, w_sim={1.0 - best_w:.2f}, F1(0.5)={best_f1_w:.4f}")

    print("[5/7] 集成概率阈值搜索与评估...")
    ens_val = best_w * cnn_val + (1.0 - best_w) * sim_val
    opt_thresholds = find_optimal_thresholds(ens_val, val_true)
    print(f"  阈值: mean={opt_thresholds.mean():.3f}, <0.5: {(opt_thresholds < 0.5).sum()}/{len(label_cols)}")

    y_val_pred, macro_f1, macro_prec, macro_rec = evaluate_at_thresholds(
        ens_val, val_true, opt_thresholds)
    auc = macro_auc(val_true, ens_val)
    subset_acc = accuracy_score(val_true, y_val_pred)
    hamming = (val_true != y_val_pred).mean()
    total_pos_true = int(val_true.sum())
    total_pos_pred = int(y_val_pred.sum())
    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_prec:.4f}      │")
    print(f"  │ Macro Recall:     {macro_rec:.4f}      │")
    print(f"  │ Macro AUC:        {auc:.4f}      │")
    print(f"  │ Subset Accuracy:  {subset_acc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming:.4f}      │")
    print(f"  │ 预测/真实:        {total_pos_pred}/{total_pos_true}      │")
    print(f"  └─────────────────────────────┘")

    print("[6/7] 测试集预测并保存提交...")
    ens_test = best_w * cnn_test + (1.0 - best_w) * sim_test
    y_test_pred = predict_with_thresholds(ens_test, opt_thresholds)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    print("[7/7] 保存配置与指标...")
    config = {
        "model": "ck001-ensemble",
        "strategy": "C002b probs + sequence similarity (DeepGOPlus style)",
        "sim_params": {"k": K, "n_features": N_FEATURES, "top_k": TOP_K, "metric": "cosine"},
        "weight_selection_metric": "macro_f1 @ threshold=0.5",
        "best_w": {"cnn": round(float(best_w), 4), "sim": round(float(1.0 - best_w), 4)},
        "thresholds": opt_thresholds.tolist(),
        "label_cols": label_cols,
    }
    with open(ENSEMBLE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": "ck001-ensemble",
        "features": "ensemble: cnn-c002b + k-mer(k=3) similarity top-3 neighbor labels",
        "best_w": {"cnn": round(float(best_w), 4), "sim": round(float(1.0 - best_w), 4)},
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
        "total_time_seconds": round(time.time() - t0, 1),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  配置: {ENSEMBLE_CONFIG_PATH}, 指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
