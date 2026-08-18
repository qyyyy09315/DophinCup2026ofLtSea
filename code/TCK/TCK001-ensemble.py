
"""
TCK001-ensemble — Transformer + CNN + k-mer 相似性特征 三路集成
=================================================================
方案:
  - 三路概率: T001a (RoPE Transformer, F1 0.2426) + C002b (CNN, F1 0.2714)
    + sim (k=6 top-3 相似度加权邻居标签, F1 0.3977)
  - 概率来源: C002b 复用 TC003 缓存; sim 由 CK002 邻居缓存重建;
    T001a 提取一次并缓存 (rel/TCK001_t001a_probs.npz)
  - 权重网格: w1+w2+w3=1, 步长 0.1 (66 组合), 固定阈值 0.5 F1 选权
  - 阈值搜索 100 档 (CK002a 经验), 输出测试集提交
输出:
  submit/submission_TCK001_ensemble.csv
  rel/TCK001_ensemble_val_metrics.json
  rel/TCK001_t001a_probs.npz
  models/tck001-ensemble.json
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
SUB_PATH = f"{SUB_DIR}/submission_TCK001_ensemble.csv"
METRICS_PATH = f"{REL_DIR}/TCK001_ensemble_val_metrics.json"
ENSEMBLE_CONFIG_PATH = f"{MODEL_DIR}/tck001-ensemble.json"
T001A_PROBS_PATH = f"{REL_DIR}/TCK001_t001a_probs.npz"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
NEIGH_K6_PATH = f"{REL_DIR}/CK002_neigh_k6.npz"
T001A_MODEL_PATH = f"{MODEL_DIR}/t001a-rope.pt"

K = 6
TOP_K = 3
AGG = "sim_weighted"
WEIGHT_STEPS = 0.1
THRESHOLD_STEPS = 100
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


def get_t001a_probs(sequences_val, y_val, sequences_test, label_cols):
    """T001a 概率: 缓存或提取。"""
    if os.path.exists(T001A_PROBS_PATH):
        d = np.load(T001A_PROBS_PATH)
        print("  T001a 概率读缓存")
        return d["t001a_val"], d["t001a_test"]
    print("  提取 T001a 概率 (首次运行)...")
    mod = load_module("T001a-rope.py")
    model = mod.TransformerClassifier(n_labels=len(label_cols)).to(DEVICE)
    ckpt = torch.load(T001A_MODEL_PATH, map_location=DEVICE, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    val_ds = mod.ProteinDataset(sequences_val, y_val, mod.MAX_LEN)
    test_ds = mod.ProteinDataset(sequences_test, np.zeros((len(sequences_test), len(label_cols)), dtype=np.int8), mod.MAX_LEN)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=512, shuffle=False,
                                             pin_memory=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=512, shuffle=False,
                                              pin_memory=True, num_workers=0)
    val_p, _ = mod.get_probs(model, val_loader)
    test_p, _ = mod.get_probs(model, test_loader)
    os.makedirs(REL_DIR, exist_ok=True)
    np.savez_compressed(T001A_PROBS_PATH, t001a_val=val_p, t001a_test=test_p)
    print(f"  T001a 概率已缓存: val={val_p.shape}, test={test_p.shape}")
    return val_p, test_p


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
    y_train = y_full[idx_train]
    print(f"  训练: {len(idx_train)}, 验证: {len(idx_val)}, 测试: {len(test_df)}")

    print("[2/7] 三路概率准备...")
    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, val_true = vp["cnn"], vp["y_true"]
    cnn_test = tp["cnn"]
    print(f"  C002b: val={cnn_val.shape} (TC003 缓存)")

    d = np.load(NEIGH_K6_PATH)
    sim_val = build_sim_prob(d["idx_val"], d["sim_val"], y_train, TOP_K, AGG)
    sim_test = build_sim_prob(d["idx_test"], d["sim_test"], y_train, TOP_K, AGG)
    print(f"  sim: val={sim_val.shape} (k=6 top-3 相似度加权, CK002 缓存重建)")

    t001a_val, t001a_test = get_t001a_probs(
        seqs[idx_val], y_full[idx_val], test_df["sequence"].values, label_cols)
    print(f"  T001a: val={t001a_val.shape}")

    print("[3/7] 三权重网格搜索 (固定阈值 0.5 F1, 66 组合)...")
    grid = np.arange(0.0, 1.0 + 1e-9, WEIGHT_STEPS)
    best_w, best_f1 = None, -1.0
    n_combo = 0
    for wt in grid:
        for wc in grid:
            ws = 1.0 - wt - wc
            if ws < -1e-9:
                continue
            n_combo += 1
            ens = wt * t001a_val + wc * cnn_val + ws * sim_val
            f1 = macro_f1_fixed_threshold(ens, val_true)
            if f1 > best_f1:
                best_f1, best_w = f1, (wt, wc, ws)
    wt, wc, ws = best_w
    print(f"  共 {n_combo} 组合; 最优 w: t001a={wt:.1f}, cnn={wc:.1f}, sim={ws:.1f}, F1(0.5)={best_f1:.4f}")

    print("[4/7] 集成概率阈值搜索...")
    ens_val = wt * t001a_val + wc * cnn_val + ws * sim_val
    ens_test = wt * t001a_test + wc * cnn_test + ws * sim_test
    opt_thresholds = find_optimal_thresholds(ens_val, val_true)
    print(f"  阈值: mean={opt_thresholds.mean():.3f}, <0.5: {(opt_thresholds < 0.5).sum()}/{len(label_cols)}")

    print("[5/7] 评估...")
    y_val_pred, macro_f1, macro_prec, macro_rec = evaluate_at_thresholds(
        ens_val, val_true, opt_thresholds)
    auc = macro_auc(val_true, ens_val)
    subset_acc = accuracy_score(val_true, y_val_pred)
    hamming = (val_true != y_val_pred).mean()
    total_pos_true = int(val_true.sum())
    total_pos_pred = int(y_val_pred.sum())
    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_prec:.4f}      │")
    print(f"  │ Macro Recall:     {macro_rec:.4f}      │")
    print(f"  │ Macro AUC:        {auc:.4f}      │")
    print(f"  │ Subset Accuracy:  {subset_acc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming:.4f}      │")
    print(f"  │ 预测/真实:        {total_pos_pred}/{total_pos_true}      │")
    print(f"  └─────────────────────────────┘")

    print("[6/7] 测试集预测并保存提交...")
    y_test_pred = predict_with_thresholds(ens_test, opt_thresholds)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    print("[7/7] 保存配置与指标...")
    config = {
        "model": "tck001-ensemble",
        "models": ["t001a-rope (T001a)", "cnn-c002b (C002b)", "kmer-sim k=6 top-3 weighted"],
        "best_w": {"t001a": round(float(wt), 4), "cnn": round(float(wc), 4),
                   "sim": round(float(ws), 4)},
        "weight_selection_metric": "macro_f1 @ threshold=0.5",
        "thresholds": opt_thresholds.tolist(),
        "label_cols": label_cols,
    }
    with open(ENSEMBLE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metrics = {
        "model": "tck001-ensemble",
        "features": "3-way ensemble: t001a-rope + cnn-c002b + kmer similarity",
        "best_w": {"t001a": round(float(wt), 4), "cnn": round(float(wc), 4),
                   "sim": round(float(ws), 4)},
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
        "total_time_seconds": round(time.time() - t0, 1),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  配置: {ENSEMBLE_CONFIG_PATH}, 指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
