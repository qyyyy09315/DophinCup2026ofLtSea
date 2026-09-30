
"""
TCK003-ensemble — k=6 k-mer 直接特征模型 + 五路集成 (方向 C)
=============================================================
方案:
  - kmer 特征: k=6 hashing (2^20 桶, 与 CK 系列一致) → L2 归一化稀疏矩阵
  - 分类器: 逐标签 SGDClassifier(log_loss) (B002 高配版: k=3→6, 8192→2^20),
    ThreadPool 6 并行
  - 前置检验: kmer_prob 与 sim_prob 的按标签 Pearson 相关 (若 >0.9 视为冗余)
  - 集成: T001a + T002 + C002b + sim + kmer 五路权重网格 (1001 组合,
    固定阈值 0.5 F1 选权) → 100 档阈值 → 提交
输出:
  submit/submission_TCK003_ensemble.csv
  rel/TCK003_ensemble_val_metrics.json
  rel/TCK003_kmer_probs.npz
  models/tck003-ensemble.json
"""

import json
import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{SUB_DIR}/submission_TCK003_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/TCK003_ensemble_val_metrics.json"
KMER_PROBS_PATH = f"{REL_DIR}/TCK003_kmer_probs.npz"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/tck003-ensemble.json"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
NEIGH_K6_PATH = f"{REL_DIR}/CK002_neigh_k6.npz"
NEIGH_FULL_PATH = f"{REL_DIR}/TCK002a_neigh_full.npz"
T001A_PROBS_PATH = f"{REL_DIR}/TCK001_t001a_probs.npz"
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"
TCK002_CONFIG_PATH = f"{MODEL_DIR}/tck002-ensemble.json"

K = 6
N_FEATURES = 2 ** 20
TOP_K = 3
AGG = "sim_weighted"
N_JOBS = 6
SGD_PARAMS = dict(loss="log_loss", alpha=1e-4, max_iter=50, tol=1e-4,
                  random_state=42)
WEIGHT_STEPS = 0.1
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


def train_label(i, X, y):
    clf = SGDClassifier(**SGD_PARAMS)
    clf.fit(X, y[:, i])
    return i, clf


def main():
    t0 = time.time()
    print("=" * 60)

    print("[1/7] 读取数据与划分...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    seqs = train_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    print(f"  训练: {len(idx_train)}, 验证: {len(idx_val)}, 测试: {len(test_df)}")

    if os.path.exists(KMER_PROBS_PATH):
        print("[2/7] kmer 概率读缓存...")
        d = np.load(KMER_PROBS_PATH)
        kmer_val, kmer_test = d["kmer_val"], d["kmer_test"]
    else:
        print("[2/7] k=6 hashing 特征与逐标签 SGD 训练...")
        vec = HashingVectorizer(analyzer="char", ngram_range=(K, K),
                                n_features=N_FEATURES, dtype=np.float32)
        X_tr = normalize(sparse.csr_matrix(vec.transform(seqs[idx_train])))
        X_val = normalize(sparse.csr_matrix(vec.transform(seqs[idx_val])))
        X_te = normalize(sparse.csr_matrix(vec.transform(test_df["sequence"].values)))
        print(f"  特征矩阵: {X_tr.shape}, nnz/行 {X_tr.nnz / X_tr.shape[0]:.0f}")
        clfs = {}
        with ThreadPoolExecutor(max_workers=N_JOBS) as ex:
            for i, clf in tqdm(ex.map(lambda i: train_label(i, X_tr, y_train),
                                      range(len(label_cols))),
                               total=len(label_cols), desc="SGD 逐标签"):
                clfs[i] = clf
        kmer_val = np.zeros((X_val.shape[0], len(label_cols)), dtype=np.float32)
        kmer_test = np.zeros((X_te.shape[0], len(label_cols)), dtype=np.float32)
        for i, clf in clfs.items():
            kmer_val[:, i] = clf.predict_proba(X_val)[:, 1]
            kmer_test[:, i] = clf.predict_proba(X_te)[:, 1]
        os.makedirs(REL_DIR, exist_ok=True)
        np.savez_compressed(KMER_PROBS_PATH, kmer_val=kmer_val, kmer_test=kmer_test)
        print(f"  kmer 概率已缓存: val={kmer_val.shape}, test={kmer_test.shape}")
    print(f"  kmer_val 密度: {kmer_val.mean():.4f}")

    print("[3/7] kmer 独立评估 + 与 sim 相关性...")
    f1_kmer = macro_f1_fixed_threshold(kmer_val, y_val)
    auc_kmer = macro_auc(y_val, kmer_val)
    print(f"  kmer 独立: F1(0.5)={f1_kmer:.4f}, AUC={auc_kmer:.4f}")
    d6 = np.load(NEIGH_K6_PATH)
    sim_val_ref = build_sim_prob(d6["idx_val"], d6["sim_val"], y_train, TOP_K, AGG)
    corrs = []
    for i in range(len(label_cols)):
        c = np.corrcoef(kmer_val[:, i], sim_val_ref[:, i])[0, 1]
        if not np.isnan(c):
            corrs.append(c)
    corr_mean = float(np.mean(corrs))
    print(f"  与 sim 按标签 Pearson 相关: mean={corr_mean:.4f} "
          f"(>0.9 视为冗余)")

    print("[4/7] 读取四路概率并组装五路...")
    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, val_true = vp["cnn"], vp["y_true"]
    cnn_test = tp["cnn"]
    sim_test = build_sim_prob(np.load(NEIGH_FULL_PATH)["idx_test"],
                              np.load(NEIGH_FULL_PATH)["sim_test"], y_full, TOP_K, AGG)
    t1 = np.load(T001A_PROBS_PATH)
    t001a_val, t001a_test = t1["t001a_val"], t1["t001a_test"]
    t2v = np.load(T002_VAL_PROBS_PATH)
    t2t = np.load(T002_TEST_PROBS_PATH)
    t002_val, t002_test = t2v["t002"], t2t["t002"]
    probs_val = {"t001a": t001a_val, "t002": t002_val, "cnn": cnn_val,
                 "sim": sim_val_ref, "kmer": kmer_val}
    probs_test = {"t001a": t001a_test, "t002": t002_test, "cnn": cnn_test,
                  "sim": sim_test, "kmer": kmer_test}

    print("[5/7] 五路权重网格搜索 (1001 组合, 固定阈值 0.5 F1)...")
    grid = np.arange(0.0, 1.0 + 1e-9, WEIGHT_STEPS)
    names = list(probs_val.keys())
    best_w, best_f1 = None, -1.0
    n_combo = 0
    for wt1 in grid:
        for wt2 in grid:
            for wc in grid:
                for ws in grid:
                    wk = 1.0 - wt1 - wt2 - wc - ws
                    if wk < -1e-9:
                        continue
                    n_combo += 1
                    ens = (wt1 * probs_val["t001a"] + wt2 * probs_val["t002"]
                           + wc * probs_val["cnn"] + ws * probs_val["sim"]
                           + wk * probs_val["kmer"])
                    f1 = macro_f1_fixed_threshold(ens, val_true)
                    if f1 > best_f1:
                        best_f1, best_w = f1, (wt1, wt2, wc, ws, wk)
    wt1, wt2, wc, ws, wk = best_w
    print(f"  共 {n_combo} 组合; 最优 w: t001a={wt1:.1f}, t002={wt2:.1f}, "
          f"cnn={wc:.1f}, sim={ws:.1f}, kmer={wk:.1f}, F1(0.5)={best_f1:.4f}")

    print("[6/7] 集成阈值搜索与评估...")
    ens_val = (wt1 * probs_val["t001a"] + wt2 * probs_val["t002"]
               + wc * probs_val["cnn"] + ws * probs_val["sim"] + wk * probs_val["kmer"])
    ens_test = (wt1 * probs_test["t001a"] + wt2 * probs_test["t002"]
                + wc * probs_test["cnn"] + ws * probs_test["sim"] + wk * probs_test["kmer"])
    opt_thresholds = find_optimal_thresholds(ens_val, val_true)
    print(f"  阈值: mean={opt_thresholds.mean():.3f}, <0.5: {(opt_thresholds < 0.5).sum()}/{len(label_cols)}")
    y_val_pred, macro_f1, macro_prec, macro_rec = evaluate_at_thresholds(
        ens_val, val_true, opt_thresholds)
    auc = macro_auc(val_true, ens_val)
    hamming = (val_true != y_val_pred).mean()
    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_prec:.4f}      │")
    print(f"  │ Macro Recall:     {macro_rec:.4f}      │")
    print(f"  │ Macro AUC:        {auc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming:.4f}      │")
    print(f"  │ 预测/真实:        {int(y_val_pred.sum())}/{int(val_true.sum())}      │")
    print(f"  └─────────────────────────────┘")

    print("[7/7] 测试集提交与保存...")
    y_test_pred = predict_with_thresholds(ens_test, opt_thresholds)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    config = {
        "model": "tck003-ensemble",
        "kmer_model": "k=6 hashing SGD per-label",
        "corr_kmer_sim": round(corr_mean, 4),
        "best_w": {"t001a": round(float(wt1), 4), "t002": round(float(wt2), 4),
                   "cnn": round(float(wc), 4), "sim": round(float(ws), 4),
                   "kmer": round(float(wk), 4)},
        "thresholds": opt_thresholds.tolist(),
        "label_cols": label_cols,
    }
    with Path(ENSEMBLE_CONFIG_PATH).open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": "tck003-ensemble",
        "kmer_alone": {"f1_05": round(f1_kmer, 6), "auc": round(auc_kmer, 6)},
        "corr_kmer_sim": round(corr_mean, 4),
        "best_w": {"t001a": round(float(wt1), 4), "t002": round(float(wt2), 4),
                   "cnn": round(float(wc), 4), "sim": round(float(ws), 4),
                   "kmer": round(float(wk), 4)},
        "val_macro_f1": round(macro_f1, 6),
        "val_macro_precision": round(macro_prec, 6),
        "val_macro_recall": round(macro_rec, 6),
        "val_macro_auc": round(auc, 6),
        "val_hamming_loss": round(hamming, 6),
        "val_total_positive_true": int(val_true.sum()),
        "val_total_positive_pred": int(y_val_pred.sum()),
        "val_pred_positive_ratio": round(y_val_pred.mean(), 6),
        "total_time_seconds": round(time.time() - t0, 1),
    }
    with Path(METRICS_PATH).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  配置: {ENSEMBLE_CONFIG_PATH}, 指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
