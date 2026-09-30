
"""
sim-feature-analysis — 序列相似性特征分析 (DeepGOPlus 式)
============================================================
目的: 验证蛋白质功能预测中"序列相似性特征"的收益
  (DeepGOPlus 核心组合 = CNN + sequence similarity, CAFA3 中
   相似性部分贡献巨大; 本实验为快速原型, 验证是否值得正式引入)
方案:
  - 特征: k=3 k-mer hashing (2^18 桶) → L2 归一化 → cosine 相似度
  - 邻居: 验证集每条序列在训练集 (91036) 中取 top-3 最近邻
  - sim_prob: 邻居标签平均 (DeepGOPlus 式标签传播)
  - 评估:
    a. sim_prob 单独: 阈值搜索后 Macro F1 / AUC
    b. 与 C002b 概率 (TC003 缓存) 加权集成: 权重网格 + 阈值搜索
  - 分析性实验, 不写腾讯文档, 只记录工作日志
输出:
  rel/sim_feature_analysis.json
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
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
CNN_VAL_PROBS = f"{REL_DIR}/TC003_val_probs.npz"

K = 3
N_FEATURES = 2 ** 18
TOP_K = 3
BLOCK = 500
VAL_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD_STEPS = 50
WEIGHT_STEPS = 11

OUT_PATH = f"{REL_DIR}/sim_feature_analysis.json"


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


def main():
    t0 = time.time()
    print("=" * 60)
    train_df = pd.read_csv(TRAIN_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    seqs = train_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    print(f"[0] train={len(idx_train)}, val={len(idx_val)}, labels={len(label_cols)}")

    print("[1/6] k-mer hashing (k=3, 2^18 桶)...")
    vec = HashingVectorizer(analyzer="char", ngram_range=(K, K),
                            n_features=N_FEATURES, dtype=np.float32)
    X = vec.transform(np.concatenate([seqs[idx_train], seqs[idx_val]]))
    X_train = normalize(sparse.csr_matrix(X[:len(idx_train)]))
    X_val = normalize(sparse.csr_matrix(X[len(idx_train):]))
    print(f"    稀疏矩阵: {X_train.shape}, 非零率 {X_train.nnz / np.prod(X_train.shape):.5f}")

    print(f"[2/6] 分块最近邻 (top-{TOP_K}, cosine)...")
    n_val = X_val.shape[0]
    neigh_idx = np.zeros((n_val, TOP_K), dtype=np.int64)
    for s in tqdm(range(0, n_val, BLOCK), desc="最近邻"):
        e = min(s + BLOCK, n_val)
        sim = (X_val[s:e] @ X_train.T).toarray()
        neigh_idx[s:e] = np.argpartition(-sim, TOP_K, axis=1)[:, :TOP_K]
    mean_sim = np.zeros(n_val)
    for i in range(n_val):
        mean_sim[i] = (X_val[i] @ X_train[neigh_idx[i]].T).toarray()[0].mean()
    print(f"    平均邻居相似度: {mean_sim.mean():.4f} (p10={np.percentile(mean_sim, 10):.4f}, p90={np.percentile(mean_sim, 90):.4f})")

    print("[3/6] sim_prob = 邻居标签平均...")
    sim_prob = y_train[neigh_idx.ravel()].reshape(n_val, TOP_K, -1).astype(np.float32).mean(axis=1)
    print(f"    sim_prob: {sim_prob.shape}, 密度 {sim_prob.mean():.4f}")

    print("[4/6] sim_prob 单独评估...")
    thr_sim = find_optimal_thresholds(sim_prob, y_val)
    _, f1_sim, prec_sim, rec_sim = evaluate_at_thresholds(sim_prob, y_val, thr_sim)
    auc_sim = macro_auc(y_val, sim_prob)
    print(f"    Macro F1={f1_sim:.4f} | Prec={prec_sim:.4f} | Rec={rec_sim:.4f} | AUC={auc_sim:.4f}")

    print("[5/6] 与 C002b 概率加权集成...")
    cnn_val = np.load(CNN_VAL_PROBS)["cnn"]
    weights = np.linspace(0.0, 1.0, WEIGHT_STEPS)
    best_w, best_f1_w = 0.5, -1.0
    for w in weights:
        ens = w * cnn_val + (1.0 - w) * sim_prob
        f1 = macro_f1_fixed_threshold(ens, y_val)
        if f1 > best_f1_w:
            best_f1_w, best_w = f1, w
    print(f"    最优 w_cnn={best_w:.2f}, F1(0.5)={best_f1_w:.4f}")
    ens_val = best_w * cnn_val + (1.0 - best_w) * sim_prob
    thr_ens = find_optimal_thresholds(ens_val, y_val)
    _, f1_ens, prec_ens, rec_ens = evaluate_at_thresholds(ens_val, y_val, thr_ens)
    auc_ens = macro_auc(y_val, ens_val)
    print(f"    Macro F1={f1_ens:.4f} | Prec={prec_ens:.4f} | Rec={rec_ens:.4f} | AUC={auc_ens:.4f}")

    print("[6/6] 保存分析结果...")
    result = {
        "experiment": "sim-feature-analysis (DeepGOPlus style)",
        "params": {"k": K, "n_features": N_FEATURES, "top_k": TOP_K, "metric": "cosine"},
        "neighbor_sim_stats": {
            "mean": round(float(mean_sim.mean()), 4),
            "p10": round(float(np.percentile(mean_sim, 10)), 4),
            "p90": round(float(np.percentile(mean_sim, 90)), 4),
        },
        "sim_alone": {
            "macro_f1": round(f1_sim, 6), "precision": round(prec_sim, 6),
            "recall": round(rec_sim, 6), "auc": round(auc_sim, 6),
        },
        "ensemble_with_cnn": {
            "best_w_cnn": round(float(best_w), 4),
            "macro_f1": round(f1_ens, 6), "precision": round(prec_ens, 6),
            "recall": round(rec_ens, 6), "auc": round(auc_ens, 6),
        },
        "baseline_tc003e": {"macro_f1": 0.2846, "auc": 0.7969},
        "total_time_seconds": round(time.time() - t0, 1),
    }
    with Path(OUT_PATH).open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"    结果: {OUT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
