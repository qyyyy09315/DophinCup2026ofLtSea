
"""
TC003a-ensemble — TC003 阈值稳健化
====================================
方案:
  - 复用 TC003 概率缓存 (rel/TC003_val_probs.npz / TC003_test_probs.npz)
    与最优权重 w (models/tc003-ensemble.json), 不重新推理/训练
  - 背景: 本地验证 F1 越高, 提交得分降幅越大 —— 峰值阈值对验证集过拟合
  - 稳健阈值: 对每个标签取 F1 >= 0.99*F1_max 的平坦区间中点,
    替代峰值阈值 (峰值点对验证集噪声敏感, 换测试集泛化损失大)
  - 输出对比: 峰值阈值 vs 稳健阈值的本地 Macro F1
  - 提交文件使用稳健阈值
输出:
  submit/submission_TC003a_ensemble.csv
  rel/TC003a_ensemble_val_metrics.json
  models/tc003a-ensemble.json
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{SUB_DIR}/submission_TC003a_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/TC003a_ensemble_val_metrics.json"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/tc003a-ensemble.json"

VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
TC003_CONFIG_PATH = f"{MODEL_DIR}/tc003-ensemble.json"

THRESHOLD_STEPS = 50
PLATEAU_RATIO = 0.99  # F1 >= ratio * max 视为平坦区间
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


def find_thresholds(y_prob, y_true, mode="peak"):
    """逐标签阈值搜索。mode=peak: 峰值最优; mode=robust: F1 平坦区间中点。"""
    n_labels = y_true.shape[1]
    best_thresholds = np.full(n_labels, 0.5, dtype=np.float32)
    candidates = np.linspace(0.01, 0.99, THRESHOLD_STEPS)
    for i in range(n_labels):
        yt = y_true[:, i]
        if yt.sum() == 0:
            continue
        f1s = np.array([f1_score(yt, (y_prob[:, i] >= t).astype(int), zero_division=0)
                        for t in candidates])
        if mode == "peak":
            best_thresholds[i] = candidates[int(np.argmax(f1s))]
        else:  # robust: 平坦区间中点
            plateau = np.where(f1s >= PLATEAU_RATIO * f1s.max())[0]
            best_thresholds[i] = candidates[int(round(np.mean([plateau[0], plateau[-1]])))]
    return best_thresholds


def predict_with_thresholds(y_prob, thresholds):
    preds = np.zeros_like(y_prob, dtype=np.int8)
    for i, t in enumerate(thresholds):
        preds[:, i] = (y_prob[:, i] >= t).astype(np.int8)
    return preds


def main():
    t_start = time.time()
    print("=" * 60)

    print("[1/5] 读取 TC003 概率缓存与最优权重...")
    if not (os.path.exists(VAL_PROBS_PATH) and os.path.exists(TEST_PROBS_PATH)):
        raise FileNotFoundError(f"缺少 TC003 概率缓存, 请先运行 TC003-ensemble.py")
    vp = np.load(VAL_PROBS_PATH)
    tp = np.load(TEST_PROBS_PATH)
    cnn_val, tf_val, y_true = vp["cnn"], vp["transformer"], vp["y_true"]
    cnn_test, tf_test = tp["cnn"], tp["transformer"]
    with open(TC003_CONFIG_PATH, "r", encoding="utf-8") as f:
        tc003_cfg = json.load(f)
    w = float(tc003_cfg["best_w_cnn"])
    print(f"  缓存: val={cnn_val.shape}, test={cnn_test.shape}, w_cnn={w:.2f} (来自 tc003-ensemble.json)")

    print("[2/5] 概率加权集成...")
    ens_val = w * cnn_val + (1.0 - w) * tf_val
    ens_test = w * cnn_test + (1.0 - w) * tf_test

    print("[3/5] 峰值阈值 vs 稳健阈值搜索...")
    thr_peak = find_thresholds(ens_val, y_true, mode="peak")
    thr_robust = find_thresholds(ens_val, y_true, mode="robust")
    print(f"  峰值阈值: mean={thr_peak.mean():.3f}, <0.5: {(thr_peak < 0.5).sum()}/{y_true.shape[1]}")
    print(f"  稳健阈值: mean={thr_robust.mean():.3f}, <0.5: {(thr_robust < 0.5).sum()}/{y_true.shape[1]}")
    diff = np.abs(thr_robust - thr_peak)
    print(f"  阈值位移: mean={diff.mean():.4f}, max={diff.max():.4f}, >0.05 的标签数: {(diff > 0.05).sum()}")

    print("[4/5] 对比评估...")
    y_peak, f1_peak, prec_peak, rec_peak = evaluate_at_thresholds(ens_val, y_true, thr_peak)
    y_robust, f1_robust, prec_robust, rec_robust = evaluate_at_thresholds(ens_val, y_true, thr_robust)
    auc = macro_auc(y_true, ens_val)  # AUC 与阈值无关
    print(f"  ┌─────────────────────────────┐")
    print(f"  │             峰值阈值   稳健阈值 │")
    print(f"  │ Macro F1:   {f1_peak:.4f}     {f1_robust:.4f}   │")
    print(f"  │ Precision:  {prec_peak:.4f}     {prec_robust:.4f}   │")
    print(f"  │ Recall:     {rec_peak:.4f}     {rec_robust:.4f}   │")
    print(f"  │ Macro AUC:        {auc:.4f}        │")
    print(f"  │ 预测/真实:  {int(y_peak.sum())}/{int(y_true.sum())}  {int(y_robust.sum())}/{int(y_true.sum())}   │")
    print(f"  └─────────────────────────────┘")
    print(f"  F1 差异: {f1_robust - f1_peak:+.4f} (稳健 - 峰值)")

    print("[5/5] 生成稳健阈值提交并保存...")
    os.makedirs(SUB_DIR, exist_ok=True)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in pd.read_csv(f"{DATA_DIR}/train.csv", nrows=1).columns if c.startswith("label_")]
    y_test_pred = predict_with_thresholds(ens_test, thr_robust)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    hamming_robust = (y_true != y_robust).mean()
    config = {
        "model": "tc003a-ensemble",
        "threshold_strategy": "robust plateau midpoint (F1 >= 0.99*max)",
        "base_ensemble": "TC003 (C002b + Transformer, w_cnn=0.80)",
        "w_cnn": round(w, 4),
        "thresholds_robust": thr_robust.tolist(),
        "label_cols": label_cols,
    }
    with open(ENSEMBLE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  配置: {ENSEMBLE_CONFIG_PATH}")

    metrics = {
        "model": "tc003a-ensemble",
        "features": "probability ensemble (reuse TC003 cache): cnn-c002b + transformer-cls",
        "threshold_strategy": "robust plateau midpoint",
        "plateau_ratio": PLATEAU_RATIO,
        "w_cnn": round(w, 4),
        "threshold_stats_robust": {
            "min": round(float(thr_robust.min()), 4),
            "max": round(float(thr_robust.max()), 4),
            "mean": round(float(thr_robust.mean()), 4),
            "n_below_05": int((thr_robust < 0.5).sum()),
        },
        "threshold_shift_vs_peak": {
            "mean": round(float(diff.mean()), 4),
            "max": round(float(diff.max()), 4),
            "n_gt_005": int((diff > 0.05).sum()),
        },
        "val_macro_f1_peak": round(f1_peak, 6),
        "val_macro_f1_robust": round(f1_robust, 6),
        "val_macro_precision_robust": round(prec_robust, 6),
        "val_macro_recall_robust": round(rec_robust, 6),
        "val_macro_auc": round(auc, 6),
        "val_hamming_loss_robust": round(hamming_robust, 6),
        "val_total_positive_true": int(y_true.sum()),
        "val_total_positive_pred_robust": int(y_robust.sum()),
        "val_pred_positive_ratio_robust": round(y_robust.mean(), 6),
        "total_time_seconds": round(time.time() - t_start, 1),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
