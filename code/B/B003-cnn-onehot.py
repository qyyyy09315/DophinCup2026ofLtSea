
"""
cnn-onehot — 基线模型：one-hot 编码 + 多尺度 1D-CNN
====================================================
输入: 氨基酸序列 → one-hot (21通道: 20AA + 1pad) → 固定长度 1500
模型: 三路并行 1D-CNN (kernel=3/5/7) → GlobalMaxPool → FC → 500 输出
训练: BCEWithLogitsLoss, AdamW, ReduceLROnPlateau, EarlyStopping

输出:
  submit/submission_CNN.csv        — 测试集预测
  rel/CNN_val_metrics.json         — 验证集指标
  models/cnn-onehot.pt             — 模型权重
"""

import json
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ── 常量 ──────────────────────────────────────────────
MAX_LEN = 1500
BATCH_SIZE = 64
N_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10
LR_PATIENCE = 4
LR_FACTOR = 0.5
RANDOM_STATE = 42
VAL_SIZE = 0.2
THRESHOLD = 0.5

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
N_CHANNELS = len(AMINO_ACIDS) + 1
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = len(AMINO_ACIDS)

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{SUB_DIR}/submission_CNN.csv"
METRICS_PATH = f"{REL_DIR}/CNN_val_metrics.json"
MODEL_PATH = f"{MODEL_DIR}/cnn-onehot.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 数据集 ────────────────────────────────────────────

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, max_len):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        indices = [AA_TO_IDX.get(aa, PAD_IDX) for aa in seq[:self.max_len]]
        n = len(indices)
        if n < self.max_len:
            indices += [PAD_IDX] * (self.max_len - n)
        seq_tensor = torch.tensor(indices, dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq_tensor, label_tensor


def collate_onehot(batch):
    seqs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    one_hot = F.one_hot(seqs, num_classes=N_CHANNELS).permute(0, 2, 1).float()
    return one_hot, labels


# ── 模型 ──────────────────────────────────────────────

class MultiScaleCNN(nn.Module):
    """三路并行 1D-CNN, 不同感受野捕捉不同尺度的 motif"""

    def __init__(self, n_labels=500, filters=64):
        super().__init__()
        self.branch_a = nn.Sequential(
            nn.Conv1d(N_CHANNELS, filters, 3, padding=1),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, 3, padding=1),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
        )
        self.branch_b = nn.Sequential(
            nn.Conv1d(N_CHANNELS, filters, 5, padding=2),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, 5, padding=2),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
        )
        self.branch_c = nn.Sequential(
            nn.Conv1d(N_CHANNELS, filters, 7, padding=3),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, 7, padding=3),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(filters * 3, n_labels)

    def forward(self, x):
        f1 = self.branch_a(x).max(dim=2)[0]
        f2 = self.branch_b(x).max(dim=2)[0]
        f3 = self.branch_c(x).max(dim=2)[0]
        f = torch.cat([f1, f2, f3], dim=1)
        f = self.dropout(f)
        return self.fc(f)


# ── 评估 ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader):
    """返回 f1, prec, rec, auc, y_pred_binary, y_true, y_prob"""
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_labels).astype(int)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    per_f1, per_prec, per_rec, per_auc = [], [], [], []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() == 0:
            continue
        per_f1.append(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        per_prec.append(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
        per_rec.append(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
        per_auc.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    return (np.mean(per_f1), np.mean(per_prec), np.mean(per_rec),
            np.mean(per_auc), y_pred, y_true, y_prob)


# ── 预测 ──────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds = []
    for x, _ in loader:
        x = x.to(DEVICE)
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds.append((probs >= THRESHOLD).cpu().numpy().astype(np.int8))
    return np.vstack(preds)


# ── 训练 ──────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


# ── 主流程 ────────────────────────────────────────────

def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")

    # 1. 读取数据
    print("=" * 60)
    print("[1/7] 读取数据...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    print(f"  训练集: {len(train_df)} 条, 测试集: {len(test_df)} 条")
    print(f"  标签数: {len(label_cols)}, 序列截断长度: {MAX_LEN}")

    # 2. 划分数据
    print("[2/7] 划分训练/验证集 (80/20)...")
    y_full = train_df[label_cols].values.astype(np.int8)
    sequences = train_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    train_ds = ProteinDataset(sequences[idx_train], y_full[idx_train], MAX_LEN)
    val_ds = ProteinDataset(sequences[idx_val], y_full[idx_val], MAX_LEN)
    test_ds = ProteinDataset(
        test_df["sequence"].values,
        np.zeros((len(test_df), len(label_cols)), dtype=np.int8),
        MAX_LEN,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_onehot, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_onehot, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_onehot, pin_memory=True, num_workers=2)
    print(f"  训练折: {len(train_ds)} 条, 验证折: {len(val_ds)} 条")

    # 3. 构建模型
    print("[3/7] 构建 MultiScaleCNN...")
    model = MultiScaleCNN(n_labels=len(label_cols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {n_params:,}")

    # 4. 训练
    print("[4/7] 训练...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE, verbose=True
    )

    best_f1 = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, N_EPOCHS + 1):
        t_ep = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_f1, val_prec, val_rec, val_auc, _, _, _ = evaluate(model, val_loader)
        scheduler.step(val_f1)

        elapsed = time.time() - t_ep
        print(f"  Epoch {epoch:2d}/{N_EPOCHS} | loss={train_loss:.4f} | "
              f"f1={val_f1:.4f} | prec={val_prec:.4f} | rec={val_rec:.4f} | auc={val_auc:.4f} | "
              f"{elapsed:.0f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"  最佳 epoch: {best_epoch}, val_f1={best_f1:.4f}")

    # 5. 加载最佳模型
    print("[5/7] 加载最佳模型...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

    # 6. 验证集最终评估
    print("[6/7] 验证集最终评估...")
    val_f1, val_prec, val_rec, val_auc, val_pred, val_true, _ = evaluate(model, val_loader)
    subset_acc = accuracy_score(val_true, val_pred)
    hamming_loss = (val_true != val_pred).mean()
    total_pos_true = int(val_true.sum())
    total_pos_pred = int(val_pred.sum())

    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {val_f1:.4f}      │")
    print(f"  │ Macro Precision:  {val_prec:.4f}      │")
    print(f"  │ Macro Recall:     {val_rec:.4f}      │")
    print(f"  │ Macro AUC:        {val_auc:.4f}      │")
    print(f"  │ Subset Accuracy:  {subset_acc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming_loss:.4f}      │")
    print(f"  │ 验证集正样本数:    {total_pos_true}        │")
    print(f"  │ 预测正样本数:      {total_pos_pred}        │")
    print(f"  └─────────────────────────────┘")

    # 7. 预测 + 保存
    print("[7/7] 预测测试集并保存...")
    y_test_pred = predict(model, test_loader)

    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交文件: {SUB_PATH} ({len(submission)} 行)")

    metrics = {
        "model": "cnn-onehot",
        "features": f"one-hot ({N_CHANNELS}ch), max_len={MAX_LEN}",
        "classifier": "MultiScaleCNN (k=3/5/7, 64f each, GlobalMaxPool)",
        "hyperparams": {
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "epochs": N_EPOCHS,
            "patience": PATIENCE,
            "lr_patience": LR_PATIENCE,
            "lr_factor": LR_FACTOR,
            "threshold": THRESHOLD,
            "best_epoch": best_epoch,
            "val_size": VAL_SIZE,
            "random_state": RANDOM_STATE,
        },
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "n_labels_total": len(label_cols),
        "n_params": n_params,
        "train_time_seconds": round(time.time() - t_start, 1),
        "val_macro_f1": round(val_f1, 6),
        "val_macro_precision": round(val_prec, 6),
        "val_macro_recall": round(val_rec, 6),
        "val_macro_auc": round(val_auc, 6),
        "val_subset_accuracy": round(subset_acc, 6),
        "val_hamming_loss": round(hamming_loss, 6),
        "val_total_positive_true": total_pos_true,
        "val_total_positive_pred": total_pos_pred,
        "val_pred_positive_ratio": round(val_pred.mean(), 6),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  指标文件: {METRICS_PATH}")

    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
