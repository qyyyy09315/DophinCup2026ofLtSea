"""
I003-classifier-comparison2 — 第二层分类器对照实验（第二轮：Boosting + 朴素贝叶斯）
=====================================================================================
方案:
  - 特征: 与 I001 完全一致 (cnn/t001a/t002/kmer_prob/r001_prob/ksim = 3000 维)
    → StandardScaler + PCA 200 维 (fit 部分拟合, 防泄漏)
  - 对照分类器 (逐标签, 验证集 60/40 划分与 I001/R002 一致):
    XGBoost / AdaBoost / CatBoost / GaussianNB
  - 评估: 固定阈值 0.5 F1 + 100 档阈值 Macro F1 / AUC
  - 与 I001 对照: LR 0.3788 / LDA 0.4043 / TCK002a 概率加权 0.4236
  - 最优分类器 → 生成提交文件
文献:
  - XGBoost: Chen & Guestrin, KDD 2016, arXiv:1603.02754
  - CatBoost: Prokhorenkova et al., NeurIPS 2018
  - AdaBoost: Freund & Schapire, JCSS 55(1):119-139, 1997, doi:10.1006/jcss.1997.1504
  - GaussianNB: 经典概率分类器 (教科书方法, 不列文献)
输出:
  rel/I003_report.json
  submit/submission_I003_best.csv (若最优超过 LDA 对照 0.4043)
"""

import json
import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xgboost as xgb
from catboost import CatBoostClassifier
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
REPORT_PATH = f"{REL_DIR}/I003_report.json"
SUB_PATH = f"{SUB_DIR}/submission_I003_best.csv"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
NEIGH_K6_PATH = f"{REL_DIR}/CK002_neigh_k6.npz"
NEIGH_FULL_PATH = f"{REL_DIR}/TCK002a_neigh_full.npz"
T001A_PROBS_PATH = f"{REL_DIR}/TCK001_t001a_probs.npz"
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"
KMER_PROBS_PATH = f"{REL_DIR}/TCK003_kmer_probs.npz"
R001_PROBS_PATH = f"{REL_DIR}/TCK004_r001_probs.npz"

PCA_DIM = 200
N_JOBS = 6
THRESHOLD_STEPS = 100
VAL_SIZE = 0.2
RANDOM_STATE = 42

# I001 对照基准 (100 档阈值, eval 子集)
I001_REF = {"LogisticRegression": 0.3788, "LDA": 0.4043}
TCK002A_F1 = 0.4236

CLASSIFIERS = {
    "XGBoost": lambda: xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        tree_method="hist", n_jobs=1, verbosity=0, random_state=0),
    "AdaBoost": lambda: AdaBoostClassifier(n_estimators=50, random_state=0),
    "CatBoost": lambda: CatBoostClassifier(
        iterations=100, depth=6, learning_rate=0.1,
        verbose=0, thread_count=1, random_seed=0),
    "GaussianNB": lambda: GaussianNB(),
}


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
    print("  搜索逐标签最优阈值 (100 档)...")
    n_labels = y_true.shape[1]
    best_thresholds = np.full(n_labels, 0.5, dtype=np.float32)
    candidates = np.linspace(0.01, 0.99, THRESHOLD_STEPS)
    for i in tqdm(range(n_labels), desc="    阈值搜索", unit="label"):
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


def build_sim_prob(neigh_idx, neigh_sim, y_space, top_k):
    n = neigh_idx.shape[0]
    labels = y_space[neigh_idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    w = np.maximum(neigh_sim[:, :top_k], 1e-7)
    return (labels * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]


def fit_predict(clf_factory, X_tr, y_tr, X_va, X_te, n_labels, name):
    pv = np.zeros((X_va.shape[0], n_labels), dtype=np.float32)
    pt = np.zeros((X_te.shape[0], n_labels), dtype=np.float32)

    def work(i):
        try:
            clf = clf_factory()
            clf.fit(X_tr, y_tr[:, i])
            if hasattr(clf, "predict_proba"):
                return i, clf.predict_proba(X_va)[:, 1], clf.predict_proba(X_te)[:, 1]
            return i, clf.decision_function(X_va), clf.decision_function(X_te)
        except Exception as e:
            print(f"  标签 {i} 失败 ({name}): {e}")
            return i, np.zeros(X_va.shape[0], dtype=np.float32), np.zeros(X_te.shape[0], dtype=np.float32)

    with ThreadPoolExecutor(max_workers=N_JOBS) as ex:
        for i, v, t in tqdm(ex.map(work, range(n_labels)), total=n_labels,
                            desc=f"  {name}", unit="label"):
            pv[:, i] = v
            pt[:, i] = t
    return pv, pt


def main():
    t0 = time.time()
    print("=" * 60)

    print("[1/6] 读取数据与概率特征...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, cnn_test = vp["cnn"], tp["cnn"]
    t1 = np.load(T001A_PROBS_PATH)
    t001a_val, t001a_test = t1["t001a_val"], t1["t001a_test"]
    t2v = np.load(T002_VAL_PROBS_PATH)
    t2t = np.load(T002_TEST_PROBS_PATH)
    t002_val, t002_test = t2v["t002"], t2t["t002"]
    km = np.load(KMER_PROBS_PATH)
    kmer_val, kmer_test = km["kmer_val"], km["kmer_test"]
    rp = np.load(R001_PROBS_PATH)
    r001_val, r001_test = rp["r001_val"], rp["r001_test"]
    d6 = np.load(NEIGH_K6_PATH)
    ksim_val = build_sim_prob(d6["idx_val"], d6["sim_val"], y_train, 3)
    dfull = np.load(NEIGH_FULL_PATH)
    ksim_test = build_sim_prob(dfull["idx_test"], dfull["sim_test"], y_full, 3)

    X_val = np.hstack([cnn_val, t001a_val, t002_val, kmer_val, r001_val, ksim_val])
    X_test = np.hstack([cnn_test, t001a_test, t002_test, kmer_test, r001_test, ksim_test])
    print(f"  特征: {X_val.shape}")

    print("[2/6] 划分 + 标准化 + PCA...")
    rng = np.random.RandomState(0)
    fit_idx = rng.choice(len(idx_val), int(len(idx_val) * 0.6), replace=False)
    eval_idx = np.setdiff1d(np.arange(len(idx_val)), fit_idx)
    sc = StandardScaler()
    sc.fit(np.vstack([X_val, X_test]))
    Xs = sc.transform(X_val).astype(np.float32)
    Xt = sc.transform(X_test).astype(np.float32)
    pca = PCA(n_components=PCA_DIM, random_state=0)
    pca.fit(Xs[fit_idx])
    Xp = pca.transform(Xs).astype(np.float32)
    Xpt = pca.transform(Xt).astype(np.float32)
    print(f"  PCA {PCA_DIM} 维, 解释方差 {pca.explained_variance_ratio_.sum():.3f}")
    print(f"  I001 对照: LR {I001_REF['LogisticRegression']} / LDA {I001_REF['LDA']} / "
          f"TCK002a {TCK002A_F1}")

    print("[3/6] 逐标签分类器对比 (fit %d / eval %d)..." % (len(fit_idx), len(eval_idx)))
    results = {}
    for name, factory in CLASSIFIERS.items():
        print(f"  --- {name} ---")
        pv, pt = fit_predict(factory, Xp[fit_idx], y_val[fit_idx],
                             Xp[eval_idx], Xpt, len(label_cols), name)
        f1_05 = np.mean([f1_score(y_val[eval_idx][:, i], (pv[:, i] >= 0.5).astype(int),
                                  zero_division=0)
                         for i in range(len(label_cols)) if y_val[eval_idx][:, i].sum() > 0])
        thr = find_optimal_thresholds(pv, y_val[eval_idx])
        _, f1, prec, rec = evaluate_at_thresholds(pv, y_val[eval_idx], thr)
        auc = macro_auc(y_val[eval_idx], pv)
        print(f"    F1(0.5)={f1_05:.4f} | 最优阈值 F1={f1:.4f} | AUC={auc:.4f}")
        results[name] = {"f1_05": round(f1_05, 6), "f1": round(f1, 6),
                         "prec": round(prec, 6), "rec": round(rec, 6),
                         "auc": round(auc, 6), "pv": pv, "pt": pt, "thr": thr}

    print("[4/6] 报告 + 提交...")
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_f1 = results[best_name]["f1"]
    print(f"  最优分类器: {best_name} (F1={best_f1:.4f})")
    print(f"  对照: I001-LDA {I001_REF['LDA']} | TCK002a {TCK002A_F1}"
          f"{'  ← 超过 LDA, 第二层路线有戏!' if best_f1 > I001_REF['LDA'] else ''}")
    report = {k: {kk: vv for kk, vv in v.items() if kk not in ("pv", "pt", "thr")}
              for k, v in results.items()}
    report["i001_reference"] = I001_REF
    report["tck002a_f1"] = TCK002A_F1
    report["best_classifier"] = best_name
    report["pca_dim"] = PCA_DIM
    report["total_time_seconds"] = round(time.time() - t0, 1)
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    r = results[best_name]
    y_test_pred = predict_with_thresholds(r["pt"], r["thr"])
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交 ({best_name}): {SUB_PATH} ({len(submission)} 行)")

    print("[5/6] 报告...")
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
