
"""
CK002a-ensemble — 集成策略稳健性检验 (CK002 衍生)
==================================================
目的: 检验 CK002 集成策略 (w 网格/阈值/选权指标) 的稳健性,
      确认 0.4199 的配置选择是否最优、平台是否平坦
检验内容 (相似性特征固定为 CK002 最优: k=6, top-3, 相似度加权):
  A. w 网格细化: 0.1 → 0.05 步长 F1 选权 → 50 档阈值
  B. 阈值细化:   w 0.1 网格 F1 选权 → 100 档阈值
  C. AUC 选权:   w 0.1 网格按 Macro AUC 选权 → 50 档阈值 (对比 F1 选权)
  D. 敏感性:     最终 w 邻域 ±0.1 的固定阈值 F1 曲线 (平台宽度)
提交: 三条线中验证 F1 最高的配置
输出:
  submit/submission_CK002a_ensemble.csv
  rel/CK002a_ensemble_val_metrics.json
  rel/CK002a_analysis.json
  models/ck002a-ensemble.json
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{SUB_DIR}/submission_CK002a_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/CK002a_ensemble_val_metrics.json"
ANALYSIS_PATH = f"{REL_DIR}/CK002a_analysis.json"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/ck002a-ensemble.json"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
NEIGH_K6_PATH = f"{REL_DIR}/CK002_neigh_k6.npz"

K = 6
TOP_K = 3
AGG = "sim_weighted"
THRESHOLD_STEPS = 50
THRESHOLD_STEPS_FINE = 100
WEIGHT_STEPS = 11
WEIGHT_STEPS_FINE = 21
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


def find_optimal_thresholds(y_prob, y_true, n_steps=THRESHOLD_STEPS):
    print(f"  搜索逐标签最优阈值 ({n_steps} 档)...")
    n_labels = y_true.shape[1]
    best_thresholds = np.full(n_labels, 0.5, dtype=np.float32)
    candidates = np.linspace(0.01, 0.99, n_steps)
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


def main():
    t0 = time.time()
    print("=" * 60)

    print("[1/6] 读取数据与概率...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)), test_size=VAL_SIZE,
                                          random_state=RANDOM_STATE)
    y_train = y_full[idx_train]

    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, val_true = vp["cnn"], vp["y_true"]
    cnn_test = tp["cnn"]

    print("[2/6] 构建 sim_prob (k=6, top-3, 相似度加权, 复用 CK002 缓存)...")
    d = np.load(NEIGH_K6_PATH)
    sim_val = build_sim_prob(d["idx_val"], d["sim_val"], y_train, TOP_K, AGG)
    sim_test = build_sim_prob(d["idx_test"], d["sim_test"], y_train, TOP_K, AGG)
    print(f"  sim_val={sim_val.shape}, sim_test={sim_test.shape}")

    analysis = {}

    print("[3/6] 检验 A: w 网格细化 (0.05 步长 F1 选权)...")
    best_w_f, best_f1_f = 0.5, -1.0
    for w in np.linspace(0.0, 1.0, WEIGHT_STEPS_FINE):
        f1 = macro_f1_fixed_threshold(w * cnn_val + (1.0 - w) * sim_val, val_true)
        if f1 > best_f1_f:
            best_f1_f, best_w_f = f1, w
    print(f"  w_fine={best_w_f:.2f}, F1(0.5)={best_f1_f:.4f}")
    thr_fine = find_optimal_thresholds(best_w_f * cnn_val + (1.0 - best_w_f) * sim_val,
                                       val_true, THRESHOLD_STEPS)
    _, f1_a, prec_a, rec_a = evaluate_at_thresholds(
        best_w_f * cnn_val + (1.0 - best_w_f) * sim_val, val_true, thr_fine)
    analysis["A_w_fine"] = {"w": round(float(best_w_f), 4), "f1_05": round(best_f1_f, 6),
                            "final_f1": round(f1_a, 6)}
    print(f"  A 最终: F1={f1_a:.4f}")

    print("[4/6] 检验 B: 阈值 100 档...")
    ens_c = 0.5 * cnn_val + 0.5 * sim_val  # 占位, 下面用 CK002 复现的 w
    best_w_b, best_f1_b = 0.5, -1.0
    for w in np.linspace(0.0, 1.0, WEIGHT_STEPS):
        f1 = macro_f1_fixed_threshold(w * cnn_val + (1.0 - w) * sim_val, val_true)
        if f1 > best_f1_b:
            best_f1_b, best_w_b = f1, w
    thr_b = find_optimal_thresholds(best_w_b * cnn_val + (1.0 - best_w_b) * sim_val,
                                    val_true, THRESHOLD_STEPS_FINE)
    _, f1_b, _, _ = evaluate_at_thresholds(
        best_w_b * cnn_val + (1.0 - best_w_b) * sim_val, val_true, thr_b)
    analysis["B_threshold_fine"] = {"w": round(float(best_w_b), 4), "final_f1": round(f1_b, 6)}
    print(f"  B 最终: F1={f1_b:.4f}")

    print("[5/6] 检验 C: AUC 选权 + D: 敏感性...")
    best_w_auc, best_auc = 0.5, -1.0
    for w in np.linspace(0.0, 1.0, WEIGHT_STEPS):
        auc = macro_auc(val_true, w * cnn_val + (1.0 - w) * sim_val)
        if auc > best_auc:
            best_auc, best_w_auc = auc, w
    print(f"  AUC 选权: w_auc={best_w_auc:.2f}, AUC={best_auc:.4f}")
    thr_c = find_optimal_thresholds(best_w_auc * cnn_val + (1.0 - best_w_auc) * sim_val,
                                    val_true, THRESHOLD_STEPS)
    _, f1_c, _, _ = evaluate_at_thresholds(
        best_w_auc * cnn_val + (1.0 - best_w_auc) * sim_val, val_true, thr_c)
    analysis["C_auc_weighting"] = {"w": round(float(best_w_auc), 4), "auc": round(best_auc, 6),
                                   "final_f1": round(f1_c, 6)}
    print(f"  C 最终: F1={f1_c:.4f}")

    w_ref = best_w_b  # F1 选权参考点
    sens = {}
    for dw in [-0.20, -0.10, 0.0, 0.10, 0.20]:
        w = round(min(max(w_ref + dw, 0.0), 1.0), 2)
        sens[f"w={w:.2f}"] = round(macro_f1_fixed_threshold(
            w * cnn_val + (1.0 - w) * sim_val, val_true), 4)
    analysis["D_sensitivity"] = sens
    print(f"  D 敏感性 (±0.2): {sens}")

    print("[6/6] 选择最优配置并提交...")
    candidates = {"A": (f1_a, best_w_f, thr_fine), "B": (f1_b, best_w_b, thr_b),
                  "C": (f1_c, best_w_auc, thr_c)}
    best_key = max(candidates, key=lambda k: candidates[k][0])
    f1_best, w_best, thr_best = candidates[best_key]
    print(f"  最优线: {best_key} (w={w_best:.2f}, F1={f1_best:.4f})")

    ens_test = w_best * cnn_test + (1.0 - w_best) * sim_test
    y_test_pred = predict_with_thresholds(ens_test, thr_best)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    ens_val = w_best * cnn_val + (1.0 - w_best) * sim_val
    y_val_pred, macro_f1, macro_prec, macro_rec = evaluate_at_thresholds(
        ens_val, val_true, thr_best)
    auc = macro_auc(val_true, ens_val)
    hamming = (val_true != y_val_pred).mean()
    total_pos_true = int(val_true.sum())
    total_pos_pred = int(y_val_pred.sum())
    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_prec:.4f}      │")
    print(f"  │ Macro Recall:     {macro_rec:.4f}      │")
    print(f"  │ Macro AUC:        {auc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming:.4f}      │")
    print(f"  │ 预测/真实:        {total_pos_pred}/{total_pos_true}      │")
    print(f"  └─────────────────────────────┘")

    with Path(ANALYSIS_PATH).open("w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    config = {
        "model": "ck002a-ensemble",
        "strategy": f"best of A(w fine)/B(thr fine)/C(auc weighting): {best_key}",
        "w_cnn": round(float(w_best), 4),
        "thresholds": thr_best.tolist(),
        "label_cols": label_cols,
    }
    with Path(ENSEMBLE_CONFIG_PATH).open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": "ck002a-ensemble",
        "best_line": best_key,
        "analysis": analysis,
        "best_w_cnn": round(float(w_best), 4),
        "val_macro_f1": round(macro_f1, 6),
        "val_macro_precision": round(macro_prec, 6),
        "val_macro_recall": round(macro_rec, 6),
        "val_macro_auc": round(auc, 6),
        "val_hamming_loss": round(hamming, 6),
        "val_total_positive_true": total_pos_true,
        "val_total_positive_pred": total_pos_pred,
        "val_pred_positive_ratio": round(y_val_pred.mean(), 6),
        "total_time_seconds": round(time.time() - t0, 1),
    }
    with Path(METRICS_PATH).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  分析: {ANALYSIS_PATH}, 配置: {ENSEMBLE_CONFIG_PATH}, 指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
