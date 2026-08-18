
"""
TCK002a-ensemble — 测试集全量邻居空间提交优化 (方向 A)
========================================================
方案:
  - 背景: sim 特征是集成主导 (w=0.5), 其质量取决于邻居空间大小
  - 优化: 测试集查询的邻居空间从 91,036 (train 部分) 扩到全量
    113,796 (train + val, 验证集标签可用, 测试集不在其中 → 无泄漏)
  - 验证集评估不变 (邻居空间仍 91,036, 避免泄漏), 权重/阈值复用
    TCK002 的最优 (t001a=0.1, t002=0.2, cnn=0.2, sim=0.5 + 100档阈值)
  - 预期: 验证 F1 不变 (0.4236), 提交质量提升 (更多同源邻居候选)
输出:
  submit/submission_TCK002a_ensemble.csv
  rel/TCK002a_ensemble_val_metrics.json
  rel/TCK002a_neigh_full.npz (测试集全量邻居缓存)
  models/tck002a-ensemble.json
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
SUB_PATH = f"{SUB_DIR}/submission_TCK002a_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/TCK002a_ensemble_val_metrics.json"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/tck002a-ensemble.json"
NEIGH_FULL_PATH = f"{REL_DIR}/TCK002a_neigh_full.npz"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
NEIGH_K6_PATH = f"{REL_DIR}/CK002_neigh_k6.npz"       # 验证集邻居 (91,036 空间)
T001A_PROBS_PATH = f"{REL_DIR}/TCK001_t001a_probs.npz"
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"
TCK002_CONFIG_PATH = f"{MODEL_DIR}/tck002-ensemble.json"  # 权重/阈值复用

K = 6
TOP_K = 3
AGG = "sim_weighted"
MAX_TOP_K = 5
N_FEATURES = 2 ** 20
BLOCK = 500
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


def build_sim_prob(neigh_idx, neigh_sim, y_train, top_k, mode):
    n = neigh_idx.shape[0]
    labels = y_train[neigh_idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    if mode == "mean":
        return labels.mean(axis=1)
    w = np.maximum(neigh_sim[:, :top_k], 1e-7)
    return (labels * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]


def main():
    t0 = time.time()
    print("=" * 60)

    print("[1/6] 读取数据与 TCK002 配置...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    seqs = train_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train = y_full[idx_train]
    y_all = y_full  # 全量标签 (train + val) 作为测试集邻居标签空间

    with open(TCK002_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    w = cfg["best_w"]
    thr = np.array(cfg["thresholds"], dtype=np.float32)
    print(f"  复用 TCK002: w={w}, 阈值 mean={thr.mean():.3f}")

    print("[2/6] 测试集全量邻居搜索 (空间 113,796)...")
    if os.path.exists(NEIGH_FULL_PATH):
        d = np.load(NEIGH_FULL_PATH)
        idx_full, sim_full = d["idx_test"], d["sim_test"]
        print(f"  读缓存: idx={idx_full.shape}, sim={sim_full.shape}")
    else:
        vec = HashingVectorizer(analyzer="char", ngram_range=(K, K),
                                n_features=N_FEATURES, dtype=np.float32)
        X_all = normalize(sparse.csr_matrix(vec.transform(seqs)))     # 全量 113,796
        X_test = normalize(sparse.csr_matrix(vec.transform(test_df["sequence"].values)))
        n_q = X_test.shape[0]
        idx_full = np.zeros((n_q, MAX_TOP_K), dtype=np.int64)
        sim_full = np.zeros((n_q, MAX_TOP_K), dtype=np.float32)
        for s in tqdm(range(0, n_q, BLOCK), desc="全量最近邻 (test)"):
            e = min(s + BLOCK, n_q)
            sim = (X_test[s:e] @ X_all.T).toarray()
            part = np.argpartition(-sim, MAX_TOP_K, axis=1)[:, :MAX_TOP_K]
            idx_full[s:e] = part
            sim_full[s:e] = np.take_along_axis(sim, part, axis=1)
        os.makedirs(REL_DIR, exist_ok=True)
        np.savez_compressed(NEIGH_FULL_PATH, idx_test=idx_full, sim_test=sim_full)
        print(f"  全量邻居缓存已保存: {NEIGH_FULL_PATH}")
    print(f"  平均 top-1 相似度: {sim_full[:, 0].mean():.4f} "
          f"(对比原空间: {np.load(NEIGH_K6_PATH)['sim_test'][:, 0].mean():.4f})")

    print("[3/6] 重建测试集 sim_prob (全量空间, top-3 加权)...")
    sim_test = build_sim_prob(idx_full, sim_full, y_all, TOP_K, AGG)
    print(f"  sim_test_full: {sim_test.shape}, 密度 {sim_test.mean():.4f}")

    print("[4/6] 组装提交概率 (权重复用 TCK002)...")
    t1 = np.load(T001A_PROBS_PATH)
    t001a_test = t1["t001a_test"]
    t2t = np.load(T002_TEST_PROBS_PATH)
    t002_test = t2t["t002"]
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_test = tp["cnn"]
    ens_test = (w["t001a"] * t001a_test + w["t002"] * t002_test
                + w["cnn"] * cnn_test + w["sim"] * sim_test)
    y_test_pred = np.zeros_like(ens_test, dtype=np.int8)
    for i, t in enumerate(thr):
        y_test_pred[:, i] = (ens_test[:, i] >= t).astype(np.int8)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行), 预测正例 {y_test_pred.sum():,}")

    print("[5/6] 验证集评估 (不变, 复现 TCK002)...")
    vp = np.load(TC003_VAL_PROBS_PATH)
    cnn_val, val_true = vp["cnn"], vp["y_true"]
    d6 = np.load(NEIGH_K6_PATH)
    sim_val = build_sim_prob(d6["idx_val"], d6["sim_val"], y_train, TOP_K, AGG)
    t1v = np.load(T001A_PROBS_PATH)
    t001a_val = t1v["t001a_val"]
    t2v = np.load(T002_VAL_PROBS_PATH)
    t002_val = t2v["t002"]
    ens_val = (w["t001a"] * t001a_val + w["t002"] * t002_val
               + w["cnn"] * cnn_val + w["sim"] * sim_val)
    y_val_pred, macro_f1, macro_prec, macro_rec = evaluate_at_thresholds(ens_val, val_true, thr)
    auc = macro_auc(val_true, ens_val)
    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_prec:.4f}      │")
    print(f"  │ Macro Recall:     {macro_rec:.4f}      │")
    print(f"  │ Macro AUC:        {auc:.4f}      │")
    print(f"  │ 预测/真实:        {int(y_val_pred.sum())}/{int(val_true.sum())}      │")
    print(f"  └─────────────────────────────┘")

    print("[6/6] 保存配置与指标...")
    config = {
        "model": "tck002a-ensemble",
        "note": "test-set neighbor space expanded to full 113,796 (no leakage); val unchanged",
        "base_weights": w,
        "thresholds": thr.tolist(),
        "label_cols": label_cols,
    }
    with open(ENSEMBLE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": "tck002a-ensemble",
        "test_neighbor_space": 113796,
        "val_neighbor_space": 91036,
        "top1_sim_full_space": round(float(sim_full[:, 0].mean()), 4),
        "top1_sim_old_space": round(float(np.load(NEIGH_K6_PATH)["sim_test"][:, 0].mean()), 4),
        "base_w": w,
        "val_macro_f1": round(macro_f1, 6),
        "val_macro_precision": round(macro_prec, 6),
        "val_macro_recall": round(macro_rec, 6),
        "val_macro_auc": round(auc, 6),
        "val_total_positive_true": int(val_true.sum()),
        "val_total_positive_pred": int(y_val_pred.sum()),
        "test_pred_positive": int(y_test_pred.sum()),
        "total_time_seconds": round(time.time() - t0, 1),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  配置: {ENSEMBLE_CONFIG_PATH}, 指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
