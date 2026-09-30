
"""
cnn-threshopt — CNN 优化：增大容量 + 逐标签阈值网格搜索
=========================================================
关键改动:
  - 滤波器 64→128
  - 训练后逐标签阈值网格搜索 (验证集 F1 最大化)
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
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

MAX_LEN = 1500
BATCH_SIZE = 256       # 64→256: 提高显存占用和GPU利用率
N_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10
LR_PATIENCE = 4
LR_FACTOR = 0.5
FILTERS = 128
THRESHOLD_STEPS = 50
RANDOM_STATE = 42
VAL_SIZE = 0.2

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
N_CHANNELS = len(AMINO_ACIDS) + 1
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = len(AMINO_ACIDS)

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{SUB_DIR}/submission_CNN_threshopt.csv"
METRICS_PATH = f"{REL_DIR}/CNN_threshopt_val_metrics.json"
MODEL_PATH = f"{MODEL_DIR}/cnn-threshopt.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CUDA 加速设置
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True   # 自动选择最优卷积算法
    torch.backends.cudnn.deterministic = False


def _build_translate_table():
    """20 种标准 AA -> 0..19, 其余字符(X/U/O/B等) -> PAD_IDX"""
    table = bytearray([PAD_IDX] * 256)
    for aa, idx in AA_TO_IDX.items():
        table[ord(aa)] = idx
    return bytes(table)


_TRANSLATE_TABLE = _build_translate_table()


def encode_sequences(sequences, max_len):
    """一次性把序列编码为 uint8 索引矩阵 (PAD_IDX 填充)"""
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


class MultiScaleCNN(nn.Module):
    def __init__(self, n_labels=500, filters=128):
        super().__init__()
        self.branch_a = nn.Sequential(
            nn.Conv1d(N_CHANNELS, filters, 3, padding=1),
            nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, 3, padding=1),
            nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
        )
        self.branch_b = nn.Sequential(
            nn.Conv1d(N_CHANNELS, filters, 5, padding=2),
            nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, 5, padding=2),
            nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
        )
        self.branch_c = nn.Sequential(
            nn.Conv1d(N_CHANNELS, filters, 7, padding=3),
            nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, 7, padding=3),
            nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(filters * 3, n_labels)

    def forward(self, x):
        f1 = self.branch_a(x).max(dim=2)[0]
        f2 = self.branch_b(x).max(dim=2)[0]
        f3 = self.branch_c(x).max(dim=2)[0]
        return self.fc(self.dropout(torch.cat([f1, f2, f3], dim=1)))


@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        probs = torch.sigmoid(model(x))
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
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


@torch.no_grad()
def predict_with_thresholds(model, loader, thresholds):
    model.eval()
    preds = []
    for x, _ in loader:
        x = x.to(DEVICE)
        probs = torch.sigmoid(model(x)).cpu().numpy()
        batch_preds = np.zeros_like(probs, dtype=np.int8)
        for i, t in enumerate(thresholds):
            batch_preds[:, i] = (probs[:, i] >= t).astype(np.int8)
        preds.append(batch_preds)
    return np.vstack(preds)


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")

    print("=" * 60)
    print("[1/8] 读取数据...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    print(f"  训练集: {len(train_df)}, 测试集: {len(test_df)}, 标签: {len(label_cols)}")

    print("[2/8] 划分训练/验证集 (80/20)...")
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

    print(f"[3/8] 构建 MultiScaleCNN (filters={FILTERS})...")
    model = MultiScaleCNN(n_labels=len(label_cols), filters=FILTERS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {n_params:,}")

    print("[4/8] 训练...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

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

    print("[5/8] 加载最佳模型...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

    print("[6/8] 验证集概率提取 + 逐标签阈值网格搜索...")
    val_prob, val_true = get_probs(model, val_loader)
    opt_thresholds = find_optimal_thresholds(val_prob, val_true)
    n_below = (opt_thresholds < 0.5).sum()
    print(f"  阈值: min={opt_thresholds.min():.3f}, max={opt_thresholds.max():.3f}, "
          f"mean={opt_thresholds.mean():.3f}, <0.5: {n_below}/{len(opt_thresholds)}")

    print("[7/8] 验证集最终评估 (逐标签最优阈值)...")
    y_val_pred, val_f1, val_prec, val_rec = evaluate_at_thresholds(
        val_prob, val_true, opt_thresholds)
    val_auc = np.mean([roc_auc_score(val_true[:, i], val_prob[:, i])
                       for i in range(val_true.shape[1]) if val_true[:, i].sum() > 0])
    subset_acc = accuracy_score(val_true, y_val_pred)
    hamming_loss = (val_true != y_val_pred).mean()
    total_pos_true = int(val_true.sum())
    total_pos_pred = int(y_val_pred.sum())

    print(f"  ┌─────────────────────────────┐")
    print(f"  │ Macro F1:        {val_f1:.4f}      │")
    print(f"  │ Macro Precision:  {val_prec:.4f}      │")
    print(f"  │ Macro Recall:     {val_rec:.4f}      │")
    print(f"  │ Macro AUC:        {val_auc:.4f}      │")
    print(f"  │ Subset Accuracy:  {subset_acc:.4f}      │")
    print(f"  │ Hamming Loss:     {hamming_loss:.4f}      │")
    print(f"  │ 预测/真实:        {total_pos_pred}/{total_pos_true}        │")
    print(f"  └─────────────────────────────┘")

    torch.save({"state_dict": model.state_dict(), "thresholds": opt_thresholds,
                "label_cols": label_cols}, MODEL_PATH)

    print("[8/8] 预测测试集并保存...")
    y_test_pred = predict_with_thresholds(model, test_loader, opt_thresholds)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    metrics = {
        "model": "cnn-threshopt",
        "key_changes": "filters 64->128, per-label threshold grid search",
        "hyperparams": {
            "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "lr": LR,
            "filters": FILTERS, "threshold_steps": THRESHOLD_STEPS,
            "best_epoch": best_epoch,
        },
        "threshold_stats": {
            "min": round(float(opt_thresholds.min()), 4),
            "max": round(float(opt_thresholds.max()), 4),
            "mean": round(float(opt_thresholds.mean()), 4),
            "n_below_05": int(n_below),
        },
        "train_samples": len(train_ds), "val_samples": len(val_ds),
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
        "val_pred_positive_ratio": round(y_val_pred.mean(), 6),
    }
    with Path(METRICS_PATH).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
