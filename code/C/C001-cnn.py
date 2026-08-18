
"""
cnn-c001 — 多尺度 CNN 增强版 (C001)
=====================================
相对 B003/B003a 的关键改动:
  - 卷积分支从 3 路扩展到 4 路 (kernel 3/5/7/9)
  - 每路加入 SE 通道注意力
  - 池化从 GlobalMax 改为 max + mean 双池化拼接
  - 损失加入 pos_weight
  - 保留逐标签阈值网格搜索
用法:
  python cnn-c001.py               # 正常训练
  python cnn-c001.py --eval-only   # 加载已保存模型直接评估
输出:
  submit/submission_CNN_C001.csv
  rel/CNN_C001_val_metrics.json
  models/cnn-c001.pt
"""

import contextlib
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
SUB_PATH = f"{SUB_DIR}/submission_CNN_C001.csv"
METRICS_PATH = f"{REL_DIR}/CNN_C001_val_metrics.json"
MODEL_PATH = f"{MODEL_DIR}/cnn-c001.pt"

MAX_LEN = 1500
BATCH_SIZE = 256
N_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10
LR_PATIENCE = 4
LR_FACTOR = 0.5
FILTERS = 128
KERNELS = (3, 5, 7, 9)
DROPOUT = 0.5
POS_WEIGHT_CLIP = 50.0
THRESHOLD_STEPS = 50
VAL_SIZE = 0.2
RANDOM_STATE = 42

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
N_CHANNELS = len(AMINO_ACIDS) + 1
PAD_IDX = len(AMINO_ACIDS)
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

EVAL_ONLY = "--eval-only" in sys.argv
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def autocast_ctx():
    if DEVICE.type == "cuda":
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()


def _build_translate_table():
    table = bytearray([PAD_IDX] * 256)
    for aa, idx in AA_TO_IDX.items():
        table[ord(aa)] = idx
    return bytes(table)


_TRANSLATE_TABLE = _build_translate_table()


def encode_sequences(sequences, max_len):
    n = len(sequences)
    out = np.full((n, max_len), PAD_IDX, dtype=np.uint8)
    for i, seq in enumerate(sequences):
        raw = seq.encode("ascii")[:max_len]
        out[i, :len(raw)] = np.frombuffer(raw.translate(_TRANSLATE_TABLE), dtype=np.uint8)
    return out


class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, max_len):
        self.seqs = encode_sequences(sequences, max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.seqs[idx]),
                torch.tensor(self.labels[idx], dtype=torch.float32))


def collate_onehot(batch):
    seqs = torch.stack([item[0] for item in batch]).long()
    labels = torch.stack([item[1] for item in batch])
    one_hot = F.one_hot(seqs, num_classes=N_CHANNELS).permute(0, 2, 1).float()
    return one_hot, labels


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x.mean(dim=2))
        return x * w.unsqueeze(-1)


class MultiScaleCNNv2(nn.Module):
    def __init__(self, n_labels=500, filters=FILTERS, kernels=KERNELS, dropout=DROPOUT):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            branch = nn.Sequential(
                nn.Conv1d(N_CHANNELS, filters, k, padding=k // 2),
                nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
                nn.Conv1d(filters, filters, k, padding=k // 2),
                nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
                SEBlock(filters),
            )
            self.branches.append(branch)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(kernels) * 2, n_labels)

    def forward(self, x):
        feats = []
        for branch in self.branches:
            h = branch(x)
            feats.append(h.max(dim=2)[0])
            feats.append(h.mean(dim=2))
        return self.fc(self.dropout(torch.cat(feats, dim=1)))


@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        with autocast_ctx():
            logits = model(x)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.vstack(all_probs), np.vstack(all_labels).astype(int)


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
        best_thresholds[i] = best_t
    pbar.close()
    return best_thresholds


@torch.no_grad()
def predict_with_thresholds(model, loader, thresholds):
    model.eval()
    preds = []
    for x, _ in loader:
        x = x.to(DEVICE)
        with autocast_ctx():
            logits = model(x)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        batch_preds = np.zeros_like(probs, dtype=np.int8)
        for i, t in enumerate(thresholds):
            batch_preds[:, i] = (probs[:, i] >= t).astype(np.int8)
        preds.append(batch_preds)
    return np.vstack(preds)

def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    n_samples = 0
    pbar = tqdm(loader, desc="Epoch", unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    pbar.close()
    return total_loss / max(n_samples, 1)


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)


def main():
    t_start = time.time()
    print(f"Device: {DEVICE}, eval_only: {EVAL_ONLY}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
    print("=" * 60)

    print("[1/8] 读取数据...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    print(f"  训练集: {len(train_df)}, 测试集: {len(test_df)}, 标签: {len(label_cols)}")

    print("[2/8] 划分训练/验证集...")
    y_full = train_df[label_cols].values.astype(np.int8)
    sequences = train_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=VAL_SIZE, random_state=RANDOM_STATE)
    train_ds = ProteinDataset(sequences[idx_train], y_full[idx_train], MAX_LEN)
    val_ds = ProteinDataset(sequences[idx_val], y_full[idx_val], MAX_LEN)
    test_ds = ProteinDataset(
        test_df["sequence"].values,
        np.zeros((len(test_df), len(label_cols)), dtype=np.int8), MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_onehot, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_onehot, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_onehot, pin_memory=True, num_workers=0)
    print(f"  训练: {len(train_ds)}, 验证: {len(val_ds)}")

    print("[3/8] 构建 MultiScaleCNNv2 (C001)...")
    model = MultiScaleCNNv2(n_labels=len(label_cols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}, kernels={KERNELS}, filters={FILTERS}, max+mean pooling, SE")

    pos = y_full[idx_train].sum(axis=0).astype(np.float32)
    neg = len(idx_train) - pos
    pos_weight = np.ones(len(label_cols), dtype=np.float32)
    mask = pos > 0
    pos_weight[mask] = np.clip(neg[mask] / pos[mask], 1.0, POS_WEIGHT_CLIP)
    pw_tensor = torch.tensor(pos_weight, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    if EVAL_ONLY:
        print("[4/8] 加载已保存模型并评估...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"未找到 {MODEL_PATH}")
        load_checkpoint(model, MODEL_PATH)
        best_epoch = None
        best_f1_05 = None
    else:
        print("[4/8] 训练...")
        best_f1, best_epoch, patience_counter = 0, 0, 0
        for epoch in range(1, N_EPOCHS + 1):
            t_ep = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
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
        load_checkpoint(model, MODEL_PATH)
        best_f1_05 = best_f1

    print("[5/8] 阈值搜索与最终评估...")
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

    print("[6/8] 测试集预测并保存提交...")
    os.makedirs(SUB_DIR, exist_ok=True)
    y_test_pred = predict_with_thresholds(model, test_loader, opt_thresholds)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    print("[7/8] 保存模型与指标...")
    torch.save({"state_dict": model.state_dict(), "thresholds": opt_thresholds,
                "label_cols": label_cols}, MODEL_PATH)
    print(f"  模型: {MODEL_PATH}")

    metrics = {
        "model": "cnn-c001",
        "eval_only": EVAL_ONLY,
        "features": "one-hot encoding (max_len=1500)",
        "classifier": "MultiScaleCNNv2 (4 kernels + SE + max/mean pooling)",
        "hyperparams": {
            "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "epochs": N_EPOCHS, "patience": PATIENCE,
            "lr_patience": LR_PATIENCE, "lr_factor": LR_FACTOR,
            "filters": FILTERS, "kernels": list(KERNELS), "dropout": DROPOUT,
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
