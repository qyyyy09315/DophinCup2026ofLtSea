"""
I002-cascade — 预测反馈的邻居重加权 (级联集成, ProtIR 式)
===========================================================
方案:
  - round1: TCK002a 集成概率 (T001a+T002+C002b+sim 四路加权, 全量邻居空间)
  - 反馈: 用 round1 软预测对邻居相似度重加权
      s'_ij = s_ij × (1 + γ·match_i_j), match = <p1_i, y_j>/<p1_i,1>
    (当前预测为 1 的标签在邻居 j 中出现的比例 → 邻居证据加权)
  - round2: 重加权邻居标签聚合 (sim') → 与原始概率按 TCK002a 权重组合
  - γ ∈ {0.2, 0.5, 1.0, 2.0} 网格 → 最优 γ 做 100 档阈值 → 提交
  - 与 TCK002b (全局共现) 区别: 邻居层面局部证据, 稀疏不过密
输出:
  submit/submission_I002_cascade.csv
  rel/I002_cascade_report.json
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
REPORT_PATH = f"{REL_DIR}/I002_cascade_report.json"
SUB_PATH = f"{SUB_DIR}/submission_I002_cascade.csv"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
NEIGH_K6_PATH = f"{REL_DIR}/CK002_neigh_k6.npz"          # val 邻居 (91,036 空间)
NEIGH_FULL_PATH = f"{REL_DIR}/TCK002a_neigh_full.npz"    # test 邻居 (113,796 空间)
T001A_PROBS_PATH = f"{REL_DIR}/TCK001_t001a_probs.npz"
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"
TCK002_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

TOP_K = 3
GAMMA_GRID = [0.2, 0.5, 1.0, 2.0]
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


def aggregate_weighted(neigh_idx, neigh_sim, y_space, p_soft, gamma, top_k):
    """预测反馈重加权的邻居标签聚合。
    match_i_j = <p_i, y_j> / <p_i, 1> (预测为 1 的标签在邻居 j 中的覆盖率)
    s'_ij = s_ij * (1 + gamma * match_i_j)
    """
    n = neigh_idx.shape[0]
    k = top_k
    idx = neigh_idx[:, :k]
    sim = neigh_sim[:, :k]
    y_nb = y_space[idx.ravel()].reshape(n, k, -1).astype(np.float32)
    # match: (n, k)  — psum 已是 (n,1) 列向量, 不能再套 [:, None] (会变成 (n,1,1),
    # 导致除法把 match 广播成 (n,n,k), w 变 4 维
    psum = p_soft.sum(axis=1, keepdims=True).clip(min=1e-7)
    match = (p_soft[:, None, :] * y_nb).sum(axis=-1) / psum
    w = np.maximum(sim * (1.0 + gamma * match), 1e-7)
    return (y_nb * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]


def main():
    t0 = time.time()
    print("=" * 60)

    print("[1/6] 读取数据与 round1 概率...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    with open(TCK002_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    w = cfg["best_w"]
    print(f"  TCK002 权重: {w}")

    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, cnn_test = vp["cnn"], tp["cnn"]
    t1 = np.load(T001A_PROBS_PATH)
    t001a_val, t001a_test = t1["t001a_val"], t1["t001a_test"]
    t2v = np.load(T002_VAL_PROBS_PATH)
    t2t = np.load(T002_TEST_PROBS_PATH)
    t002_val, t002_test = t2v["t002"], t2t["t002"]
    d6 = np.load(NEIGH_K6_PATH)
    dfull = np.load(NEIGH_FULL_PATH)

    def sim_plain(idx, sim, y_space):
        y_nb = y_space[idx[:, :TOP_K].ravel()].reshape(len(idx), TOP_K, -1).astype(np.float32)
        ww = np.maximum(sim[:, :TOP_K], 1e-7)
        return (y_nb * ww[:, :, None]).sum(axis=1) / ww.sum(axis=1)[:, None]

    sim_val0 = sim_plain(d6["idx_val"], d6["sim_val"], y_train)
    sim_test0 = sim_plain(dfull["idx_test"], dfull["sim_test"], y_full)

    def ensemble(sim_v, sim_t):
        ev = (w["t001a"] * t001a_val + w["t002"] * t002_val
              + w["cnn"] * cnn_val + w["sim"] * sim_v)
        if sim_t is None:  # γ 网格阶段只评估验证集
            return ev, None
        et = (w["t001a"] * t001a_test + w["t002"] * t002_test
              + w["cnn"] * cnn_test + w["sim"] * sim_t)
        return ev, et

    p1_val, p1_test = ensemble(sim_val0, sim_test0)
    f1_round1 = macro_f1_fixed_threshold(p1_val, y_val)
    print(f"  round1: F1(0.5)={f1_round1:.4f} (应≈0.4136)")

    print("[2/6] γ 网格 (预测反馈邻居重加权)...")
    results = {}
    for gamma in GAMMA_GRID:
        sim_v2 = aggregate_weighted(d6["idx_val"], d6["sim_val"], y_train,
                                    p1_val, gamma, TOP_K)
        p2_val, _ = ensemble(sim_v2, None)
        f1 = macro_f1_fixed_threshold(p2_val, y_val)
        results[gamma] = f1
        print(f"  γ={gamma}: F1(0.5)={f1:.4f}")

    best_gamma = max(results, key=lambda g: results[g])
    print(f"  最优 γ={best_gamma}")

    print("[3/6] 最优 γ 完整评估...")
    sim_v2 = aggregate_weighted(d6["idx_val"], d6["sim_val"], y_train,
                                p1_val, best_gamma, TOP_K)
    sim_t2 = aggregate_weighted(dfull["idx_test"], dfull["sim_test"], y_full,
                                p1_test, best_gamma, TOP_K)
    p2_val, p2_test = ensemble(sim_v2, sim_t2)
    thr = find_optimal_thresholds(p2_val, y_val)
    _, f1, prec, rec = evaluate_at_thresholds(p2_val, y_val, thr)
    auc = macro_auc(y_val, p2_val)
    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {f1:.4f}      │")
    print(f"  │ Macro Precision:  {prec:.4f}      │")
    print(f"  │ Macro Recall:     {rec:.4f}      │")
    print(f"  │ Macro AUC:        {auc:.4f}      │")
    print(f"  └─────────────────────────────┘")
    print(f"  round2 vs round1: ΔF1 = {f1 - f1_round1:+.4f}")

    print("[4/6] 测试集提交...")
    y_test_pred = predict_with_thresholds(p2_test, thr)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    print("[5/6] 保存报告...")
    report = {
        "gamma_grid": results,
        "best_gamma": best_gamma,
        "round1_f1_05": round(f1_round1, 6),
        "round2": {"f1": round(f1, 6), "prec": round(prec, 6),
                   "rec": round(rec, 6), "auc": round(auc, 6)},
        "delta_f1": round(f1 - f1_round1, 6),
        "total_time_seconds": round(time.time() - t0, 1),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
