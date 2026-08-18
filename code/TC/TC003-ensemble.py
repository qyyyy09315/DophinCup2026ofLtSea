
"""
tc003-ensemble — CNN-c002b + Transformer 概率加权集成
===========================================================
方案:
  - 分别加载 C002b (cnn-c002b) 和 B007 (transformer-cls)
  - 在验证集上网格搜索权重 w: p_ens = w * p_cnn + (1-w) * p_tf
  - 用验证集 Macro F1 选择最优 w（比赛评分标准）
  - 对集成概率做逐标签阈值网格搜索，输出测试集提交
输出:
  submit/submission_TC003_ensemble.csv
  rel/TC003_ensemble_val_metrics.json
  rel/TC003_val_probs.npz
  rel/TC003_test_probs.npz
  models/tc003-ensemble.json
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
SUB_PATH = f"{SUB_DIR}/submission_TC003_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/TC003_ensemble_val_metrics.json"
VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/tc003-ensemble.json"

CNN_MODEL_PATH = f"{MODEL_DIR}/cnn-c002b.pt"
TRANSFORMER_MODEL_PATH = f"{MODEL_DIR}/transformer-cls.pt"

WEIGHT_STEPS = 11
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
        print("[2/7] 加载缓存的验证/测试概率...")
        vp = np.load(VAL_PROBS_PATH)
        tp = np.load(TEST_PROBS_PATH)
        cnn_val, tf_val, val_true = vp["cnn"], vp["transformer"], vp["y_true"]
        cnn_test, tf_test = tp["cnn"], tp["transformer"]
    else:
        print("[2/7] 加载 CNN-c002b 并提取概率...")
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
        print(f"  CNN val: {cnn_val.shape}, test: {cnn_test.shape}")

        print("[3/7] 加载 Transformer 并提取概率...")
        tf_mod = load_module("B007-transformer-cls.py")
        tf_model = load_model_state(
            tf_mod.TransformerClassifier(n_labels=len(label_cols)), TRANSFORMER_MODEL_PATH)
        tf_val_ds = tf_mod.ProteinDataset(
            sequences[idx_val], y_full[idx_val], tf_mod.MAX_LEN)
        tf_test_ds = tf_mod.ProteinDataset(
            test_df["sequence"].values,
            np.zeros((len(test_df), len(label_cols)), dtype=np.int8), tf_mod.MAX_LEN)
        tf_val_loader = torch.utils.data.DataLoader(
            tf_val_ds, batch_size=tf_mod.BATCH_SIZE, shuffle=False,
            pin_memory=True, num_workers=0)
        tf_test_loader = torch.utils.data.DataLoader(
            tf_test_ds, batch_size=tf_mod.BATCH_SIZE, shuffle=False,
            pin_memory=True, num_workers=0)
        tf_val, _ = tf_mod.get_probs(tf_model, tf_val_loader)
        tf_test, _ = tf_mod.get_probs(tf_model, tf_test_loader)
        print(f"  Transformer val: {tf_val.shape}, test: {tf_test.shape}")

        os.makedirs(REL_DIR, exist_ok=True)
        np.savez_compressed(VAL_PROBS_PATH, cnn=cnn_val, transformer=tf_val, y_true=val_true)
        np.savez_compressed(TEST_PROBS_PATH, cnn=cnn_test, transformer=tf_test)
        print(f"  概率缓存已保存")

    print("[4/7] 权重网格搜索 (验证集 Macro F1)...")
    weights = np.linspace(0.0, 1.0, WEIGHT_STEPS)
    best_w, best_f1_w = 0.5, -1.0
    pbar = tqdm(weights, desc="权重搜索", unit="w",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for w in pbar:
        ens = w * cnn_val + (1.0 - w) * tf_val
        f1 = macro_f1_with_optimal_thresholds(ens, val_true, quiet=True)
        if f1 > best_f1_w:
            best_f1_w, best_w = f1, w
        pbar.set_postfix(f1=f"{f1:.4f}")
    pbar.close()
    print(f"  最优权重: w_cnn={best_w:.2f}, w_tf={1.0 - best_w:.2f}, Macro F1={best_f1_w:.4f}")

    print("[5/7] 集成概率阈值搜索...")
    ens_val = best_w * cnn_val + (1.0 - best_w) * tf_val
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
    ens_test = best_w * cnn_test + (1.0 - best_w) * tf_test
    y_test_pred = predict_with_thresholds(ens_test, opt_thresholds)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    print("[7/7] 保存集成配置与指标...")
    config = {
        "model": "tc003-ensemble",
        "weight_selection_metric": "macro_f1",
        "best_weight_macro_f1": round(float(best_f1_w), 6),
        "models": ["cnn-c002b (C002b)", "transformer-cls (B007)"],
        "best_w_cnn": round(float(best_w), 4),
        "best_w_transformer": round(float(1.0 - best_w), 4),
        "thresholds": opt_thresholds.tolist(),
        "label_cols": label_cols,
    }
    with open(ENSEMBLE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  配置: {ENSEMBLE_CONFIG_PATH}")

    metrics = {
        "model": "tc003-ensemble",
        "weight_selection_metric": "macro_f1",
        "best_weight_macro_f1": round(float(best_f1_w), 6),
        "features": "probability ensemble: cnn-c002b + transformer-cls",
        "best_w": {
            "cnn": round(float(best_w), 4),
            "transformer": round(float(1.0 - best_w), 4),
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
