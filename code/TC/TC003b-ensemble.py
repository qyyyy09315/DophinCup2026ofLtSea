
"""
TC003b-ensemble — 三模型概率加权集成 (C002b + B007 + T002)
============================================================
方案:
  - 对照 TC003 (两模型): 增加 T002 (RoPE Transformer) 为第三个模型
  - 概率来源: C002b/B007 复用 TC003 缓存; T002 提取一次并缓存
    (rel/T002_val_probs.npz / T002_test_probs.npz, TC004 等可复用)
  - 权重网格: w1+w2+w3=1, 步长 0.1 (66 组合), 选权指标 = 固定阈值
    0.5 的 Macro F1 (numpy 向量化, 避免 66 次全阈值搜索)
  - 选定权重后做逐标签阈值搜索 (50 档), 输出测试集提交
输出:
  submit/submission_TC003b_ensemble.csv
  rel/TC003b_ensemble_val_metrics.json
  models/tc003b-ensemble.json
"""

import importlib.util
import json
import os
import time
import numpy as np
import pandas as pd
import torch
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
SUB_PATH = f"{SUB_DIR}/submission_TC003b_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/TC003b_ensemble_val_metrics.json"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/tc003b-ensemble.json"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"
T002_MODEL_PATH = f"{MODEL_DIR}/t002-rope.pt"

WEIGHT_STEPS = 0.1
THRESHOLD_STEPS = 50
VAL_SIZE = 0.2
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SCRIPTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(path):
    for sub in ("B", "C", "T", "TC", "CK", "TCK", "tools"):
        p = os.path.join(SCRIPTS_ROOT, sub, path)
        if os.path.exists(p):
            break
    else:
        raise FileNotFoundError(f"脚本 {path} 未找到 (搜索 code/ 各系列)")
    spec = importlib.util.spec_from_file_location(os.path.splitext(path)[0], p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def macro_auc(y_true, y_prob):
    aucs = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, y_prob[:, i]))
    return float(np.mean(aucs))


def macro_f1_fixed_threshold(ens, y_true, t=0.5):
    """固定阈值的 Macro F1 (numpy 快速版, 用于权重选权)。"""
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


def get_t002_probs(sequences_val, y_val, sequences_test):
    """提取并缓存 T002 概率 (仅首次运行推理, 之后读缓存)。"""
    if os.path.exists(T002_VAL_PROBS_PATH) and os.path.exists(T002_TEST_PROBS_PATH):
        vp = np.load(T002_VAL_PROBS_PATH)
        tp = np.load(T002_TEST_PROBS_PATH)
        print(f"  T002 概率读缓存: val={vp['t002'].shape}, test={tp['t002'].shape}")
        return vp["t002"], tp["t002"]
    print("  提取 T002 概率 (首次运行)...")
    mod = load_module("T002-rope.py")
    model = mod.TransformerClassifier(n_labels=y_val.shape[1]).to(DEVICE)
    ckpt = torch.load(T002_MODEL_PATH, map_location=DEVICE, weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    val_ds = mod.ProteinDataset(sequences_val, y_val, mod.MAX_LEN)
    test_ds = mod.ProteinDataset(sequences_test, np.zeros((len(sequences_test), y_val.shape[1]), dtype=np.int8), mod.MAX_LEN)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False, pin_memory=True, num_workers=0)
    val_p, _ = mod.get_probs(model, val_loader)
    test_p, _ = mod.get_probs(model, test_loader)
    os.makedirs(REL_DIR, exist_ok=True)
    np.savez_compressed(T002_VAL_PROBS_PATH, t002=val_p)
    np.savez_compressed(T002_TEST_PROBS_PATH, t002=test_p)
    print(f"  T002 概率已缓存: val={val_p.shape}, test={test_p.shape}")
    return val_p, test_p


def main():
    t_start = time.time()
    print("=" * 60)

    print("[1/6] 读取数据与概率...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    sequences = train_df["sequence"].values
    _, idx_val = train_test_split(np.arange(len(train_df)), test_size=VAL_SIZE, random_state=RANDOM_STATE)

    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, tf_val, y_true = vp["cnn"], vp["transformer"], vp["y_true"]
    cnn_test, tf_test = tp["cnn"], tp["transformer"]
    t002_val, t002_test = get_t002_probs(sequences[idx_val], y_full[idx_val], test_df["sequence"].values)
    print(f"  C002b val={cnn_val.shape}, B007 val={tf_val.shape}, T002 val={t002_val.shape}")

    print("[2/6] 三模型权重网格搜索 (固定阈值 0.5 Macro F1)...")
    grid = np.arange(0.0, 1.0 + 1e-9, WEIGHT_STEPS)
    best_w, best_f1 = None, -1.0
    n_combo = 0
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            n_combo += 1
            ens = w1 * cnn_val + w2 * tf_val + w3 * t002_val
            f1 = macro_f1_fixed_threshold(ens, y_true)
            if f1 > best_f1:
                best_f1, best_w = f1, (w1, w2, w3)
    print(f"  共 {n_combo} 组合; 最优 w: cnn={best_w[0]:.1f}, b007={best_w[1]:.1f}, t002={best_w[2]:.1f}, F1(0.5)={best_f1:.4f}")

    print("[3/6] 集成概率阈值搜索...")
    ens_val = best_w[0] * cnn_val + best_w[1] * tf_val + best_w[2] * t002_val
    ens_test = best_w[0] * cnn_test + best_w[1] * tf_test + best_w[2] * t002_test
    opt_thresholds = find_optimal_thresholds(ens_val, y_true)
    print(f"  阈值: mean={opt_thresholds.mean():.3f}, <0.5: {(opt_thresholds < 0.5).sum()}/{len(label_cols)}")

    print("[4/6] 评估...")
    y_val_pred, macro_f1, macro_prec, macro_rec = evaluate_at_thresholds(ens_val, y_true, opt_thresholds)
    auc = macro_auc(y_true, ens_val)
    subset_acc = accuracy_score(y_true, y_val_pred)
    hamming = (y_true != y_val_pred).mean()
    total_pos_true = int(y_true.sum())
    total_pos_pred = int(y_val_pred.sum())
    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_prec:.4f}      │")
    print(f"  │ Macro Recall:     {macro_rec:.4f}      │")
    print(f"  │ Macro AUC:        {auc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming:.4f}      │")
    print(f"  │ 预测/真实:        {total_pos_pred}/{total_pos_true}      │")
    print(f"  └─────────────────────────────┘")

    print("[5/6] 测试集预测并保存提交...")
    os.makedirs(SUB_DIR, exist_ok=True)
    y_test_pred = predict_with_thresholds(ens_test, opt_thresholds)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    print("[6/6] 保存配置与指标...")
    config = {
        "model": "tc003b-ensemble",
        "weight_selection_metric": "macro_f1 @ threshold=0.5",
        "models": ["cnn-c002b (C002b)", "transformer-cls (B007)", "t002-rope (T002)"],
        "best_w": {"cnn": round(float(best_w[0]), 4), "b007": round(float(best_w[1]), 4),
                   "t002": round(float(best_w[2]), 4)},
        "best_weight_f1_05": round(float(best_f1), 6),
        "thresholds": opt_thresholds.tolist(),
        "label_cols": label_cols,
    }
    with Path(ENSEMBLE_CONFIG_PATH).open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": "tc003b-ensemble",
        "features": "3-model probability ensemble: cnn-c002b + transformer-cls + t002-rope",
        "weight_selection_metric": "macro_f1 @ threshold=0.5 (66 combos)",
        "best_w": {"cnn": round(float(best_w[0]), 4), "b007": round(float(best_w[1]), 4),
                   "t002": round(float(best_w[2]), 4)},
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
        "total_time_seconds": round(time.time() - t_start, 1),
    }
    with Path(METRICS_PATH).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  配置: {ENSEMBLE_CONFIG_PATH}, 指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
