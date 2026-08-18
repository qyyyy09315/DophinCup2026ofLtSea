
"""
tc004-ensemble — CNN-c002b + T002 (RoPE Transformer) 概率加权集成
==================================================================
方案:
  - 分别加载 C002b (cnn-c002b) 和 T002 (t002-rope) 的概率
  - C002b 概率复用 TC003 缓存 (rel/TC003_val_probs.npz 的 cnn 键,
    同一模型同一切分, 无需重新推理); 无缓存时自动重新提取
  - T002 (RoPE 位置编码) 替代 TC003 中的 B007 (可学习位置编码),
    T002 单模型 AUC 0.7791 高于 B007 的 0.7567
  - 验证集网格搜索权重 w: p_ens = w * p_cnn + (1-w) * p_t002
  - 用验证集 Macro F1 选择最优 w（比赛评分标准）
  - 对集成概率做逐标签阈值网格搜索，输出测试集提交
输出:
  submit/submission_TC004_ensemble.csv
  rel/TC004_ensemble_val_metrics.json
  rel/TC004_val_probs.npz
  rel/TC004_test_probs.npz
  models/tc004-ensemble.json
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
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{SUB_DIR}/submission_TC004_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/TC004_ensemble_val_metrics.json"
VAL_PROBS_PATH = f"{REL_DIR}/TC004_val_probs.npz"
TEST_PROBS_PATH = f"{REL_DIR}/TC004_test_probs.npz"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/tc004-ensemble.json"

# C002b 概率复用 TC003 缓存 (同一模型同一验证切分)
TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
# T002 概率复用 TC003b 提取的共享缓存 (无需重新推理)
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"

CNN_MODEL_PATH = f"{MODEL_DIR}/cnn-c002b.pt"
T002_MODEL_PATH = f"{MODEL_DIR}/t002-rope.pt"

WEIGHT_STEPS = 11
THRESHOLD_STEPS = 50
VAL_SIZE = 0.2
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_model_state(model, path):
    model = model.to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model.eval()


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


def find_optimal_thresholds(y_prob, y_true, quiet=False):
    if not quiet:
        print("  搜索逐标签最优阈值...")
    n_labels = y_true.shape[1]
    best_thresholds = np.full(n_labels, 0.5, dtype=np.float32)
    candidates = np.linspace(0.01, 0.99, THRESHOLD_STEPS)
    iterator = (range(n_labels) if quiet else tqdm(
        range(n_labels), desc="阈值搜索", unit="label",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"))
    for i in iterator:
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
    if not quiet:
        iterator.close()
    return best_thresholds


def macro_f1_with_optimal_thresholds(y_prob, y_true, quiet=True):
    thresholds = find_optimal_thresholds(y_prob, y_true, quiet=quiet)
    _, f1, _, _ = evaluate_at_thresholds(y_prob, y_true, thresholds)
    return f1


def predict_with_thresholds(y_prob, thresholds):
    preds = np.zeros_like(y_prob, dtype=np.int8)
    for i, t in enumerate(thresholds):
        preds[:, i] = (y_prob[:, i] >= t).astype(np.int8)
    return preds

def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("[1/7] 读取数据并划分训练/验证集...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    sequences = train_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=VAL_SIZE, random_state=RANDOM_STATE)
    print(f"  训练: {len(idx_train)}, 验证: {len(idx_val)}, 测试: {len(test_df)}, 标签: {len(label_cols)}")

    if os.path.exists(VAL_PROBS_PATH) and os.path.exists(TEST_PROBS_PATH):
        print("[2/7] 加载缓存的验证/测试概率 (TC004)...")
        vp = np.load(VAL_PROBS_PATH)
        tp = np.load(TEST_PROBS_PATH)
        cnn_val, t002_val, val_true = vp["cnn"], vp["t002"], vp["y_true"]
        cnn_test, t002_test = tp["cnn"], tp["t002"]
    else:
        # C002b 概率: 优先复用 TC003 缓存 (同一模型同一验证切分)
        if os.path.exists(TC003_VAL_PROBS_PATH) and os.path.exists(TC003_TEST_PROBS_PATH):
            print("[2/7] C002b 概率复用 TC003 缓存...")
            vp = np.load(TC003_VAL_PROBS_PATH)
            tp = np.load(TC003_TEST_PROBS_PATH)
            cnn_val, val_true = vp["cnn"], vp["y_true"]
            cnn_test = tp["cnn"]
            print(f"  C002b val: {cnn_val.shape}, test: {cnn_test.shape} (来自 TC003 缓存)")
        else:
            print("[2/7] 加载 CNN-c002b 并提取概率 (无 TC003 缓存)...")
            cnn_mod = load_module("C002b-cnn.py")
            cnn_model = load_model_state(
                cnn_mod.MultiScaleCNNv3(n_labels=len(label_cols), filters=cnn_mod.FILTERS),
                CNN_MODEL_PATH)
            cnn_val_ds = cnn_mod.ProteinDataset(
                sequences[idx_val], y_full[idx_val], cnn_mod.MAX_LEN)
            cnn_test_ds = cnn_mod.ProteinDataset(
                test_df["sequence"].values,
                np.zeros((len(test_df), len(label_cols)), dtype=np.int8), cnn_mod.MAX_LEN)
            cnn_val_loader = torch.utils.data.DataLoader(
                cnn_val_ds, batch_size=cnn_mod.BATCH_SIZE, shuffle=False,
                collate_fn=cnn_mod.collate_onehot, pin_memory=True, num_workers=0)
            cnn_test_loader = torch.utils.data.DataLoader(
                cnn_test_ds, batch_size=cnn_mod.BATCH_SIZE, shuffle=False,
                collate_fn=cnn_mod.collate_onehot, pin_memory=True, num_workers=0)
            cnn_val, val_true = cnn_mod.get_probs(cnn_model, cnn_val_loader)
            cnn_test, _ = cnn_mod.get_probs(cnn_model, cnn_test_loader)
            print(f"  C002b val: {cnn_val.shape}, test: {cnn_test.shape}")

        print("[3/7] 加载 T002 (RoPE Transformer) 并提取概率...")
        if os.path.exists(T002_VAL_PROBS_PATH) and os.path.exists(T002_TEST_PROBS_PATH):
            print("  T002 概率读共享缓存 (TC003b 提取)...")
            vp2 = np.load(T002_VAL_PROBS_PATH)
            tp2 = np.load(T002_TEST_PROBS_PATH)
            t002_val = vp2["t002"]
            t002_test = tp2["t002"]
        else:
            print("  提取 T002 概率 (首次运行)...")
            t002_mod = load_module("T002-rope.py")
            t002_model = load_model_state(
                t002_mod.TransformerClassifier(n_labels=len(label_cols)), T002_MODEL_PATH)
            t002_val_ds = t002_mod.ProteinDataset(
                sequences[idx_val], y_full[idx_val], t002_mod.MAX_LEN)
            t002_test_ds = t002_mod.ProteinDataset(
                test_df["sequence"].values,
                np.zeros((len(test_df), len(label_cols)), dtype=np.int8), t002_mod.MAX_LEN)
            t002_val_loader = torch.utils.data.DataLoader(
                t002_val_ds, batch_size=512, shuffle=False,
                pin_memory=True, num_workers=0)
            t002_test_loader = torch.utils.data.DataLoader(
                t002_test_ds, batch_size=512, shuffle=False,
                pin_memory=True, num_workers=0)
            t002_val, _ = t002_mod.get_probs(t002_model, t002_val_loader)
            t002_test, _ = t002_mod.get_probs(t002_model, t002_test_loader)
        print(f"  T002 val: {t002_val.shape}, test: {t002_test.shape}")

        os.makedirs(REL_DIR, exist_ok=True)
        np.savez_compressed(VAL_PROBS_PATH, cnn=cnn_val, t002=t002_val, y_true=val_true)
        np.savez_compressed(TEST_PROBS_PATH, cnn=cnn_test, t002=t002_test)
        print(f"  概率缓存已保存 (TC004)")

    print("[4/7] 权重网格搜索 (固定阈值 0.5 Macro F1, 快速选权)...")
    weights = np.linspace(0.0, 1.0, WEIGHT_STEPS)
    best_w, best_f1_w = 0.5, -1.0
    pbar = tqdm(weights, desc="权重搜索", unit="w",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for w in pbar:
        ens = w * cnn_val + (1.0 - w) * t002_val
        f1 = macro_f1_fixed_threshold(ens, val_true)
        if f1 > best_f1_w:
            best_f1_w, best_w = f1, w
        pbar.set_postfix(f1=f"{f1:.4f}")
    pbar.close()
    print(f"  最优权重: w_cnn={best_w:.2f}, w_t002={1.0 - best_w:.2f}, F1(0.5)={best_f1_w:.4f}")

    print("[5/7] 集成概率阈值搜索...")
    ens_val = best_w * cnn_val + (1.0 - best_w) * t002_val
    opt_thresholds = find_optimal_thresholds(ens_val, val_true)
    print(f"  阈值: mean={opt_thresholds.mean():.3f}, <0.5: {(opt_thresholds < 0.5).sum()}/{len(label_cols)}")

    y_val_pred, macro_f1, macro_prec, macro_rec = evaluate_at_thresholds(
        ens_val, val_true, opt_thresholds)
    macro_auc_final = macro_auc(val_true, ens_val)
    subset_acc = accuracy_score(val_true, y_val_pred)
    hamming_loss = (val_true != y_val_pred).mean()
    total_pos_true = int(val_true.sum())
    total_pos_pred = int(y_val_pred.sum())

    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_prec:.4f}      │")
    print(f"  │ Macro Recall:     {macro_rec:.4f}      │")
    print(f"  │ Macro AUC:        {macro_auc_final:.4f}      │")
    print(f"  │ Subset Accuracy:  {subset_acc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming_loss:.4f}      │")
    print(f"  │ 预测/真实:        {total_pos_pred}/{total_pos_true}      │")
    print(f"  └─────────────────────────────┘")

    print("[6/7] 测试集集成预测并保存...")
    ens_test = best_w * cnn_test + (1.0 - best_w) * t002_test
    y_test_pred = predict_with_thresholds(ens_test, opt_thresholds)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    print("[7/7] 保存集成配置与指标...")
    config = {
        "model": "tc004-ensemble",
        "weight_selection_metric": "macro_f1 @ threshold=0.5",
        "best_weight_macro_f1": round(float(best_f1_w), 6),
        "models": ["cnn-c002b (C002b)", "t002-rope (T002, RoPE)"],
        "best_w_cnn": round(float(best_w), 4),
        "best_w_t002": round(float(1.0 - best_w), 4),
        "thresholds": opt_thresholds.tolist(),
        "label_cols": label_cols,
    }
    with open(ENSEMBLE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  配置: {ENSEMBLE_CONFIG_PATH}")

    metrics = {
        "model": "tc004-ensemble",
        "weight_selection_metric": "macro_f1 @ threshold=0.5",
        "best_weight_macro_f1": round(float(best_f1_w), 6),
        "features": "probability ensemble: cnn-c002b + t002-rope",
        "best_w": {
            "cnn": round(float(best_w), 4),
            "t002": round(float(1.0 - best_w), 4),
        },
        "threshold_stats": {
            "min": round(float(opt_thresholds.min()), 4),
            "max": round(float(opt_thresholds.max()), 4),
            "mean": round(float(opt_thresholds.mean()), 4),
            "n_below_05": int((opt_thresholds < 0.5).sum()),
        },
        "val_samples": len(idx_val), "test_samples": len(test_df),
        "n_labels_total": len(label_cols),
        "total_time_seconds": round(time.time() - t_start, 1),
        "val_macro_f1": round(macro_f1, 6),
        "val_macro_precision": round(macro_prec, 6),
        "val_macro_recall": round(macro_rec, 6),
        "val_macro_auc": round(macro_auc_final, 6),
        "val_subset_accuracy": round(subset_acc, 6),
        "val_hamming_loss": round(hamming_loss, 6),
        "val_total_positive_true": total_pos_true,
        "val_total_positive_pred": total_pos_pred,
        "val_pred_positive_ratio": round(y_val_pred.mean(), 6),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
