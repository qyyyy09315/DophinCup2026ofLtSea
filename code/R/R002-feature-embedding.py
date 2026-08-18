"""
R002-feature-embedding — LSTM 中间表示作为特征补充 (对照实验)
==============================================================
方案:
  - 文献依据: TALE (Cao & Shen, Bioinformatics 2021, 10.1093/bioinformatics/
    btab198) 序列-标签联合嵌入; E-SNPs&GO (2022) 序列 embedding 改善注释
  - 核心问题: R001 BiLSTM 的 512 维序列表示作为**特征补充**是否有增益
  - 对照设计 (第二层逐标签 LogisticRegression, L2 正则):
    基线特征 = [cnn, t001a, t002, kmer_prob, r001_prob, ksim] (3000 维)
    增强特征 = 基线 + R001 embedding (3512 维)
    若增强 F1 显著更高 → 表示特征有效
  - 特征标准化 (StandardScaler), 100 档阈值
输出:
  rel/R002_report.json
  submit/submission_R002_embed.csv (若增强有效)
"""

import json
import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
REPORT_PATH = f"{REL_DIR}/R002_report.json"
SUB_PATH = f"{SUB_DIR}/submission_R002_embed.csv"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
NEIGH_K6_PATH = f"{REL_DIR}/CK002_neigh_k6.npz"
NEIGH_FULL_PATH = f"{REL_DIR}/TCK002a_neigh_full.npz"
T001A_PROBS_PATH = f"{REL_DIR}/TCK001_t001a_probs.npz"
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"
KMER_PROBS_PATH = f"{REL_DIR}/TCK003_kmer_probs.npz"
R001_PROBS_PATH = f"{REL_DIR}/TCK004_r001_probs.npz"
EMB_PATH = f"{REL_DIR}/R001_embeddings.npz"

LR_C = 1.0
N_JOBS = 4
PCA_DIM = 200
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


def build_sim_prob(neigh_idx, neigh_sim, y_space, top_k):
    n = neigh_idx.shape[0]
    labels = y_space[neigh_idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    w = np.maximum(neigh_sim[:, :top_k], 1e-7)
    return (labels * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]


def fit_stack(X_tr, y_tr, X_va, X_te, n_labels):
    """逐标签 LR (PCA 降维后, 收敛快), 返回验证/测试概率。"""
    pv = np.zeros((X_va.shape[0], n_labels), dtype=np.float32)
    pt = np.zeros((X_te.shape[0], n_labels), dtype=np.float32)

    def work(i):
        try:
            lr = LogisticRegression(C=LR_C, max_iter=500, tol=1e-3)
            lr.fit(X_tr, y_tr[:, i])
            return i, lr.predict_proba(X_va)[:, 1], lr.predict_proba(X_te)[:, 1]
        except Exception as e:
            print(f"  标签 {i} 失败: {e}")
            return i, np.zeros(X_va.shape[0], dtype=np.float32), np.zeros(X_te.shape[0], dtype=np.float32)

    with ThreadPoolExecutor(max_workers=N_JOBS) as ex:
        for i, v, t in ex.map(work, range(n_labels)):
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

    base_val = np.hstack([cnn_val, t001a_val, t002_val, kmer_val, r001_val, ksim_val])
    base_test = np.hstack([cnn_test, t001a_test, t002_test, kmer_test, r001_test, ksim_test])
    print(f"  基线特征: {base_val.shape}")

    emb = np.load(EMB_PATH)
    emb_val, emb_test = emb["emb_val"], emb["emb_test"]
    enh_val = np.hstack([base_val, emb_val])
    enh_test = np.hstack([base_test, emb_test])
    print(f"  增强特征: {enh_val.shape} (+R001 embedding 512 维)")

    print("[2/6] 特征标准化 + PCA 降维...")
    sc = StandardScaler()
    sc.fit(np.vstack([base_val, base_test]))
    base_val_s = sc.transform(base_val).astype(np.float32)
    base_test_s = sc.transform(base_test).astype(np.float32)
    sc2 = StandardScaler()
    sc2.fit(np.vstack([enh_val, enh_test]))
    enh_val_s = sc2.transform(enh_val).astype(np.float32)
    enh_test_s = sc2.transform(enh_test).astype(np.float32)

    print("[3/6] 验证集内划分 (60% 拟合 / 40% 评估, 防泄漏)...")
    rng = np.random.RandomState(0)
    n_val = len(idx_val)
    fit_idx = rng.choice(n_val, int(n_val * 0.6), replace=False)
    eval_idx = np.setdiff1d(np.arange(n_val), fit_idx)
    print(f"  fit {len(fit_idx)}, eval {len(eval_idx)}")

    # PCA 在拟合部分拟合 (概率特征高度相关, 降维保证 LR 收敛 + 内存可控)
    pca1 = PCA(n_components=PCA_DIM, random_state=0).fit(base_val_s[fit_idx])
    pca2 = PCA(n_components=PCA_DIM, random_state=0).fit(enh_val_s[fit_idx])
    base_val_s = pca1.transform(base_val_s).astype(np.float32)
    base_test_s = pca1.transform(base_test_s).astype(np.float32)
    enh_val_s = pca2.transform(enh_val_s).astype(np.float32)
    enh_test_s = pca2.transform(enh_test_s).astype(np.float32)
    print(f"  PCA {PCA_DIM} 维 (解释方差 {pca1.explained_variance_ratio_.sum():.3f} / "
          f"{pca2.explained_variance_ratio_.sum():.3f})")

    print("[4/6] 逐标签 LR: 基线特征...")
    pv_base, pt_base = fit_stack(base_val_s[fit_idx], y_val[fit_idx],
                                 base_val_s[eval_idx], base_test_s, len(label_cols))
    _, f1_base, _, _ = evaluate_at_thresholds(
        pv_base, y_val[eval_idx], find_optimal_thresholds(pv_base, y_val[eval_idx]))
    auc_base = macro_auc(y_val[eval_idx], pv_base)
    print(f"  基线: F1={f1_base:.4f}, AUC={auc_base:.4f}")

    print("[5/6] 逐标签 LR: 增强特征 (+embedding)...")
    pv_enh, pt_enh = fit_stack(enh_val_s[fit_idx], y_val[fit_idx],
                               enh_val_s[eval_idx], enh_test_s, len(label_cols))
    thr_enh = find_optimal_thresholds(pv_enh, y_val[eval_idx])
    _, f1_enh, prec_enh, rec_enh = evaluate_at_thresholds(pv_enh, y_val[eval_idx], thr_enh)
    auc_enh = macro_auc(y_val[eval_idx], pv_enh)
    print(f"  增强: F1={f1_enh:.4f}, AUC={auc_enh:.4f}")
    print(f"  ΔF1 = {f1_enh - f1_base:+.4f} (增强 - 基线)")

    print("[6/6] 报告 + 提交...")
    report = {
        "baseline_features": "cnn+t001a+t002+kmer_prob+r001_prob+ksim (3000维)",
        "enhanced_features": "baseline + R001 embedding (3512维)",
        "baseline": {"f1": round(f1_base, 6), "auc": round(auc_base, 6)},
        "enhanced": {"f1": round(f1_enh, 6), "auc": round(auc_enh, 6),
                     "precision": round(prec_enh, 6), "recall": round(rec_enh, 6)},
        "delta_f1": round(f1_enh - f1_base, 6),
        "total_time_seconds": round(time.time() - t0, 1),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if f1_enh > f1_base + 0.001:
        y_test_pred = predict_with_thresholds(pt_enh, thr_enh)
        os.makedirs(SUB_DIR, exist_ok=True)
        submission = test_df[["protein_id"]].copy()
        label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
        submission = pd.concat([submission, label_df], axis=1)
        submission.to_csv(SUB_PATH, index=False)
        print(f"  增强有效 → 提交: {SUB_PATH} ({len(submission)} 行)")
    else:
        print("  增强无显著增益 → 不生成提交")
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
