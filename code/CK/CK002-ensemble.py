
"""
CK002-ensemble — 序列相似性特征调优 + C002b 集成
==================================================
方案:
  - CK001 基线: k=3, top-3 等权, F1 0.3556
  - 调优扫描 (24 配置): k ∈ {3,4,5,6} × top_k ∈ {1,3,5} ×
    聚合 ∈ {等权平均, 相似度加权}
  - 邻居索引/相似度按 k 缓存 (rel/CK002_neigh_k{k}.npz), 重跑秒级
  - 选权指标: 与 C002b 集成 (w 网格 11 档) 的固定阈值 0.5 Macro F1
  - 最优配置: 完整阈值搜索 (50 档) + 测试集提交
输出:
  submit/submission_CK002_ensemble.csv
  rel/CK002_ensemble_val_metrics.json
  rel/CK002_scan_report.json
  rel/CK002_neigh_k{k}.npz (邻居索引/相似度缓存)
  models/ck002-ensemble.json
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
SUB_PATH = f"{SUB_DIR}/submission_CK002_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/CK002_ensemble_val_metrics.json"
SCAN_PATH = f"{REL_DIR}/CK002_scan_report.json"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/ck002-ensemble.json"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"

N_FEATURES = 2 ** 20     # k=6 时 6-mer 空间巨大, 加大桶数减少碰撞
K_LIST = [3, 4, 5, 6]
TOP_K_LIST = [1, 3, 5]
MAX_TOP_K = 5
AGG_MODES = ["mean", "sim_weighted"]
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


def get_neighbors(seqs_train, seqs_query, query_name, k, cache_path):
    """top-5 邻居索引与相似度 (按 k 缓存)。"""
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        print(f"  k={k} {query_name} 邻居读缓存")
        return d[f"idx_{query_name}"], d[f"sim_{query_name}"]
    vec = HashingVectorizer(analyzer="char", ngram_range=(k, k),
                            n_features=N_FEATURES, dtype=np.float32)
    X_train = normalize(sparse.csr_matrix(vec.transform(seqs_train)))
    X_q = normalize(sparse.csr_matrix(vec.transform(seqs_query)))
    n_q = X_q.shape[0]
    idx = np.zeros((n_q, MAX_TOP_K), dtype=np.int64)
    sims = np.zeros((n_q, MAX_TOP_K), dtype=np.float32)
    for s in tqdm(range(0, n_q, BLOCK), desc=f"最近邻 k={k} ({query_name})"):
        e = min(s + BLOCK, n_q)
        sim = (X_q[s:e] @ X_train.T).toarray()
        part = np.argpartition(-sim, MAX_TOP_K, axis=1)[:, :MAX_TOP_K]
        idx[s:e] = part
        sims[s:e] = np.take_along_axis(sim, part, axis=1)
    return idx, sims


def build_sim_prob(neigh_idx, neigh_sim, y_train, top_k, mode):
    """从邻居索引重建 sim_prob (top_k + 聚合方式可任意组合)。"""
    n = neigh_idx.shape[0]
    labels = y_train[neigh_idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    if mode == "mean":
        return labels.mean(axis=1)
    w = neigh_sim[:, :top_k]  # (n, top_k)
    w = np.maximum(w, 1e-7)
    return (labels * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]


def main():
    t0 = time.time()
    print("=" * 60)

    print("[1/6] 读取数据与划分...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    seqs = train_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    seqs_train = seqs[idx_train]
    print(f"  训练: {len(idx_train)}, 验证: {len(idx_val)}, 测试: {len(test_df)}")

    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, val_true = vp["cnn"], vp["y_true"]
    cnn_test = tp["cnn"]

    print("[2/6] 提取各 k 的邻居索引/相似度 (首次 ~50 分钟, 按 k 缓存)...")
    neigh = {}
    for k in K_LIST:
        cache = f"{REL_DIR}/CK002_neigh_k{k}.npz"
        iv, sv = get_neighbors(seqs_train, seqs[idx_val], "val", k, cache)
        it, st = get_neighbors(seqs_train, test_df["sequence"].values, "test", k, cache)
        if not os.path.exists(cache):
            os.makedirs(REL_DIR, exist_ok=True)
            np.savez_compressed(cache, idx_val=iv, sim_val=sv, idx_test=it, sim_test=st)
            print(f"  k={k} 邻居缓存已保存: {cache}")
        neigh[k] = (iv, sv, it, st)

    print("[3/6] 配置扫描 (24 组合, 固定阈值 0.5 F1 选权)...")
    scan_rows = []
    best_cfg, best_f1_scan = None, -1.0
    for k in K_LIST:
        iv, sv, it, st = neigh[k]
        for top_k in TOP_K_LIST:
            for mode in AGG_MODES:
                sim_val = build_sim_prob(iv, sv, y_train, top_k, mode)
                f1_sim = macro_f1_fixed_threshold(sim_val, val_true)
                best_w, best_f1 = 0.5, -1.0
                for w in np.linspace(0.0, 1.0, WEIGHT_STEPS):
                    f1 = macro_f1_fixed_threshold(w * cnn_val + (1.0 - w) * sim_val, val_true)
                    if f1 > best_f1:
                        best_f1, best_w = f1, w
                scan_rows.append({"k": k, "top_k": top_k, "agg": mode,
                                  "f1_sim": round(f1_sim, 4), "w_cnn": round(float(best_w), 2),
                                  "f1_ens": round(best_f1, 4)})
                print(f"  k={k} top={top_k} agg={mode:11s} | sim F1={f1_sim:.4f} | "
                      f"w_cnn={best_w:.2f} | ens F1={best_f1:.4f}")
                if best_f1 > best_f1_scan:
                    best_f1_scan, best_cfg = best_f1, (k, top_k, mode, float(best_w))
    k_best, top_best, agg_best, w_best = best_cfg
    print(f"\n  最优配置: k={k_best}, top_k={top_best}, agg={agg_best}, w_cnn={w_best:.2f}, F1(0.5)={best_f1_scan:.4f}")

    print("[4/6] 最优配置完整评估...")
    iv, sv, it, st = neigh[k_best]
    sim_val = build_sim_prob(iv, sv, y_train, top_best, agg_best)
    sim_test = build_sim_prob(it, st, y_train, top_best, agg_best)
    ens_val = w_best * cnn_val + (1.0 - w_best) * sim_val
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

    print("[5/6] 测试集预测并保存提交...")
    ens_test = w_best * cnn_test + (1.0 - w_best) * sim_test
    y_test_pred = predict_with_thresholds(ens_test, opt_thresholds)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    print("[6/6] 保存报告/配置/指标...")
    with open(SCAN_PATH, "w", encoding="utf-8") as f:
        json.dump({"scan": scan_rows, "best": {
            "k": k_best, "top_k": top_best, "agg": agg_best,
            "w_cnn": round(w_best, 4), "f1_05": round(best_f1_scan, 6)}},
            f, ensure_ascii=False, indent=2)

    config = {
        "model": "ck002-ensemble",
        "best_config": {"k": k_best, "top_k": top_best, "agg": agg_best},
        "w_cnn": round(float(w_best), 4),
        "thresholds": opt_thresholds.tolist(),
        "label_cols": label_cols,
    }
    with open(ENSEMBLE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": "ck002-ensemble",
        "best_config": {"k": k_best, "top_k": top_best, "agg": agg_best},
        "best_w_cnn": round(float(w_best), 4),
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
    print(f"  扫描报告: {SCAN_PATH}")
    print(f"  配置: {ENSEMBLE_CONFIG_PATH}, 指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
