"""
esm2-head — ESM-2 平均池化嵌入 + GPU 联合线性分类头
=====================================================
输入: rel/train_esm2_embeddings.parquet, rel/test_esm2_embeddings.parquet
模型: Linear(640, 500), BCEWithLogitsLoss + pos_weight
后处理: 验证集逐标签阈值网格搜索 (F1 最大化)
输出:
  submit/submission_ESM2_head.csv  — 测试集预测
  rel/ESM2_head_val_metrics.json   — 验证集指标
  models/esm2-head.pt              — 线性头权重 + 阈值
"""

import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

REL_DIR = "rel"
SUB_DIR = "submit"
MODEL_DIR = "models"
TRAIN_EMB_PATH = f"{REL_DIR}/train_esm2_embeddings.parquet"
TEST_EMB_PATH = f"{REL_DIR}/test_esm2_embeddings.parquet"
SUB_PATH = f"{SUB_DIR}/submission_ESM2_head.csv"
METRICS_PATH = f"{REL_DIR}/ESM2_head_val_metrics.json"
MODEL_PATH = f"{MODEL_DIR}/esm2-head.pt"

EMB_DIM = 640
EMB_COLS = [f"emb_{i}" for i in range(EMB_DIM)]
RANDOM_STATE = 42
VAL_SIZE = 0.2
BATCH_SIZE = 2048
N_EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 6
LR_PATIENCE = 3
LR_FACTOR = 0.5
POS_WEIGHT_CLIP = 50.0
THRESHOLD_STEPS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx]),
                torch.from_numpy(self.y[idx].astype(np.float32)))


def collate(batch):
    X = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    return X, y


@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    probs, labels = [], []
    for X, y in loader:
        X = X.to(DEVICE)
        probs.append(torch.sigmoid(model(X)).cpu().numpy())
        labels.append(y.numpy())
    return np.vstack(probs), np.vstack(labels).astype(int)


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
    best_th = np.full(n_labels, 0.5, dtype=np.float32)
    candidates = np.linspace(0.01, 0.99, THRESHOLD_STEPS)
    pbar = tqdm(range(n_labels), desc="阈值搜索", unit="label",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for i in pbar:
        yt = y_true[:, i]
        if yt.sum() == 0:
            continue
        best_f1, best_t = 0, 0.5
        for t in candidates:
            pred = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(yt, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_th[i] = best_t
    pbar.close()
    return best_th


@torch.no_grad()
def predict_with_thresholds(model, loader, thresholds):
    model.eval()
    preds = []
    for X, _ in loader:
        X = X.to(DEVICE)
        probs = torch.sigmoid(model(X)).cpu().numpy()
        batch_pred = np.zeros_like(probs, dtype=np.int8)
        for i, t in enumerate(thresholds):
            batch_pred[:, i] = (probs[:, i] >= t).astype(np.int8)
        preds.append(batch_pred)
    return np.vstack(preds)


def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")
    print("=" * 60)

    print("[1/6] 读取 ESM-2 嵌入 parquet...")
    train = pd.read_parquet(TRAIN_EMB_PATH)
    test = pd.read_parquet(TEST_EMB_PATH)
    label_cols = [c for c in train.columns if c.startswith("label_")]
    X_full = train[EMB_COLS].values.astype(np.float32)
    y_full = train[label_cols].values.astype(np.int8)
    X_test = test[EMB_COLS].values.astype(np.float32)
    print(f"  train: {X_full.shape}, test: {X_test.shape}, labels: {len(label_cols)}")

    print("[2/6] 划分训练/验证集 (80/20)...")
    idx_train, idx_val = train_test_split(
        np.arange(len(train)), test_size=VAL_SIZE, random_state=RANDOM_STATE)
    X_tr, X_val = X_full[idx_train], X_full[idx_val]
    y_tr, y_val = y_full[idx_train], y_full[idx_val]

    train_ds = EmbedDataset(X_tr, y_tr)
    val_ds = EmbedDataset(X_val, y_val)
    test_ds = EmbedDataset(X_test, np.zeros((len(X_test), len(label_cols)), dtype=np.int8))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,
                             pin_memory=True, num_workers=0)

    print("[3/6] 构建线性分类头...")
    model = nn.Linear(EMB_DIM, len(label_cols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    pos = y_tr.sum(axis=0).astype(np.float32)
    neg = len(y_tr) - pos
    pos_weight = np.ones(len(label_cols), dtype=np.float32)
    mask = pos > 0
    pos_weight[mask] = np.clip(neg[mask] / pos[mask], 1.0, POS_WEIGHT_CLIP)
    pw_tensor = torch.tensor(pos_weight, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE)

    print("[4/6] GPU 联合训练...")
    best_f1, best_epoch, patience_counter = 0, 0, 0
    for epoch in range(1, N_EPOCHS + 1):
        t_ep = time.time()
        model.train()
        total_loss = 0
        pbar_ep = tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}", unit="batch", leave=False)
        for X, y in pbar_ep:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            pbar_ep.set_postfix(loss=loss.item())
        pbar_ep.close()
        train_loss = total_loss / len(train_ds)

        val_prob, val_true = get_probs(model, val_loader)
        _, val_f1, _, _ = evaluate_at_thresholds(
            val_prob, val_true, np.full(len(label_cols), 0.5))
        scheduler.step(val_f1)
        elapsed = time.time() - t_ep
        print(f"  Epoch {epoch:2d}/{N_EPOCHS} | loss={train_loss:.4f} | "
              f"val_f1(0.5)={val_f1:.4f} | {elapsed:.0f}s")

        if val_f1 > best_f1:
            best_f1, best_epoch, patience_counter = val_f1, epoch, 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break
    print(f"  最佳 epoch: {best_epoch}, val_f1(0.5)={best_f1:.4f}")

    print("[5/6] 加载最佳模型 + 阈值网格搜索...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    val_prob, val_true = get_probs(model, val_loader)
    opt_thresholds = find_optimal_thresholds(val_prob, val_true)
    print(f"  阈值: mean={opt_thresholds.mean():.3f}, <0.5: {(opt_thresholds < 0.5).sum()}/{len(label_cols)}")

    y_val_pred, macro_f1, macro_prec, macro_rec = evaluate_at_thresholds(
        val_prob, val_true, opt_thresholds)
    macro_auc = np.mean([roc_auc_score(val_true[:, i], val_prob[:, i])
                         for i in range(len(label_cols)) if val_true[:, i].sum() > 0])
    subset_acc = accuracy_score(val_true, y_val_pred)
    hamming_loss = (val_true != y_val_pred).mean()
    total_pos_true = int(val_true.sum())
    total_pos_pred = int(y_val_pred.sum())

    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {macro_f1:.4f}      │")
    print(f"  │ Macro Precision:  {macro_prec:.4f}      │")
    print(f"  │ Macro Recall:     {macro_rec:.4f}      │")
    print(f"  │ Macro AUC:        {macro_auc:.4f}      │")
    print(f"  │ Subset Accuracy:  {subset_acc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming_loss:.4f}      │")
    print(f"  │ 预测/真实:        {total_pos_pred}/{total_pos_true}      │")
    print(f"  └─────────────────────────────┘")

    print("[6/6] 保存结果...")
    torch.save({"state_dict": model.state_dict(), "thresholds": opt_thresholds,
                "label_cols": label_cols}, MODEL_PATH)

    y_test_pred = predict_with_thresholds(model, test_loader, opt_thresholds)
    submission = test[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    metrics = {
        "model": "esm2-head",
        "features": "ESM-2 150M mean-pooled embedding (640d)",
        "classifier": "Linear head (GPU joint training, BCE + pos_weight)",
        "hyperparams": {
            "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "epochs": N_EPOCHS, "patience": PATIENCE,
            "lr_patience": LR_PATIENCE, "lr_factor": LR_FACTOR,
            "pos_weight_clip": POS_WEIGHT_CLIP, "threshold_steps": THRESHOLD_STEPS,
            "best_epoch": best_epoch, "val_size": VAL_SIZE, "random_state": RANDOM_STATE,
        },
        "threshold_stats": {
            "min": round(float(opt_thresholds.min()), 4),
            "max": round(float(opt_thresholds.max()), 4),
            "mean": round(float(opt_thresholds.mean()), 4),
            "n_below_05": int((opt_thresholds < 0.5).sum()),
        },
        "train_samples": len(idx_train), "val_samples": len(idx_val),
        "n_labels_total": len(label_cols), "n_params": n_params,
        "train_time_seconds": round(time.time() - t_start, 1),
        "val_macro_f1": round(macro_f1, 6),
        "val_macro_precision": round(macro_prec, 6),
        "val_macro_recall": round(macro_rec, 6),
        "val_macro_auc": round(macro_auc, 6),
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
