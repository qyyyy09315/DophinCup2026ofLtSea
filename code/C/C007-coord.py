"""
C007-coord — C002b 骨干 + 真实长度归一化的 1D 坐标通道
================================================================
依据:
  - CoordConv: Liu et al., NeurIPS 2018, arXiv:1807.03247
    卷积对绝对坐标不敏感；在输入上拼 x/y 通道即可让网络看见位置。
  - C002b 左对齐 one-hot，pad 已有第 21 通道，但不知道残基在蛋白里的相对位置。
    若按 MAX_LEN=1500 归一化，短蛋白会全部挤在「画布左端」，像全是 N 端。
自己的改法:
  坐标按**真实序列长度**归一化：残基 i 的通道值为 i/(L-1)∈[0,1]，
  pad 为 0。N 端≈0、C 端≈1，与蛋白长短无关。不是视频 TSM 移通道，
  也不是换嵌入（C006 已负）。骨干/ASL/80ep/cosine 同 C002b，只多 1 通道。
  判据：独立 F1(0.5) 超 C002b 0.2714。
用法:
  python code/C/C007-coord.py
  python code/C/C007-coord.py --eval-only
输出:
  models/cnn-c007.pt
  rel/C007_report.json
  rel/C007_val_probs.npz
  submit/submission_C007.csv
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util
from pathlib import Path
_spec = importlib.util.spec_from_file_location(
    "c002b", os.path.join(PROJECT_ROOT, "code", "C", "C002b-cnn.py"))
c002b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c002b)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUB_PATH = os.path.join(SUB_DIR, "submission_C007.csv")
REPORT_PATH = os.path.join(REL_DIR, "C007_report.json")
PROBS_PATH = os.path.join(REL_DIR, "C007_val_probs.npz")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn-c007.pt")

MAX_LEN = 1500
BATCH_SIZE = 256
N_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10
VAL_SIZE = 0.2
RANDOM_STATE = 42
IN_CH = c002b.N_CHANNELS + 1
C002B_F1_05 = 0.2714
DEVICE = c002b.DEVICE


def collate_coord(batch):
    seqs = torch.stack([item[0] for item in batch]).long()
    labels = torch.stack([item[1] for item in batch])
    one_hot = F.one_hot(seqs, num_classes=c002b.N_CHANNELS).permute(0, 2, 1).float()
    mask = seqs != c002b.PAD_IDX
    length = mask.sum(dim=1, keepdim=True).clamp(min=1)
    idx = torch.arange(seqs.size(1), device=seqs.device).unsqueeze(0).expand_as(seqs)
    denom = (length - 1).clamp(min=1).float()
    coord = torch.where(mask, idx.float() / denom, torch.zeros_like(seqs, dtype=torch.float32))
    x = torch.cat([one_hot, coord.unsqueeze(1)], dim=1)
    return x, labels


class MultiScaleCNNCoord(nn.Module):
    """C002b 四路膨胀 CNN，输入 21 one-hot + 1 真实长度坐标。"""

    def __init__(self, n_labels=500, in_ch=IN_CH, filters=c002b.FILTERS,
                 kernels=c002b.KERNELS, dilations=c002b.DILATIONS,
                 dropout=c002b.DROPOUT):
        super().__init__()
        self.branches = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        self.ses = nn.ModuleList()
        for k, d in zip(kernels, dilations):
            pad = d * (k - 1) // 2
            branch = nn.Sequential(
                nn.Conv1d(in_ch, filters, k, padding=pad, dilation=d),
                nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
                nn.Conv1d(filters, filters, k, padding=pad, dilation=d),
                nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
            )
            self.branches.append(branch)
            self.shortcuts.append(nn.Conv1d(in_ch, filters, 1))
            self.ses.append(c002b.SEBlock(filters))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(kernels) * 2, n_labels)

    def forward(self, x):
        feats = []
        for branch, shortcut, se in zip(self.branches, self.shortcuts, self.ses):
            h = branch(x) + shortcut(x)
            h = se(h)
            feats.append(h.max(dim=2)[0])
            feats.append(h.mean(dim=2))
        return self.fc(self.dropout(torch.cat(feats, dim=1)))


def main():
    eval_only = "--eval-only" in sys.argv
    t_start = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REL_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)
    print("=" * 60)
    print(f"C007 coord: in_ch={IN_CH} epochs={N_EPOCHS} "
          f"device={DEVICE} eval_only={eval_only}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    sequences = train_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=VAL_SIZE, random_state=RANDOM_STATE)
    train_ds = c002b.ProteinDataset(sequences[idx_train], y_full[idx_train], MAX_LEN)
    val_ds = c002b.ProteinDataset(sequences[idx_val], y_full[idx_val], MAX_LEN)
    test_ds = c002b.ProteinDataset(
        test_df["sequence"].values,
        np.zeros((len(test_df), len(label_cols)), dtype=np.int8), MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_coord, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_coord, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_coord, pin_memory=True, num_workers=0)
    print(f"  训练 {len(train_ds)} 验证 {len(val_ds)}")

    model = MultiScaleCNNCoord(n_labels=len(label_cols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")
    criterion = c002b.AsymmetricLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=LR / 100)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    best_f1, best_epoch = 0.0, 0
    if eval_only:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(MODEL_PATH)
        c002b.load_checkpoint(model, MODEL_PATH)
        print("  加载已保存模型")
    else:
        patience_counter = 0
        for epoch in range(1, N_EPOCHS + 1):
            t_ep = time.time()
            train_loss = c002b.train_epoch(model, train_loader, optimizer, criterion, scaler)
            val_prob, val_true = c002b.get_probs(model, val_loader)
            _, val_f1, _, _ = c002b.evaluate_at_thresholds(
                val_prob, val_true, np.full(len(label_cols), 0.5))
            scheduler.step()
            print(f"  Epoch {epoch:2d}/{N_EPOCHS} | loss={train_loss:.4f} | "
                  f"val_f1(0.5)={val_f1:.4f} | {time.time() - t_ep:.0f}s")
            if val_f1 > best_f1:
                best_f1, best_epoch, patience_counter = val_f1, epoch, 0
                torch.save({"state_dict": model.state_dict(),
                            "best_f1_05": best_f1, "best_epoch": best_epoch}, MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        print(f"  最佳 epoch {best_epoch} F1(0.5)={best_f1:.4f}")
        c002b.load_checkpoint(model, MODEL_PATH)

    val_prob, val_true = c002b.get_probs(model, val_loader)
    opt_t = c002b.find_optimal_thresholds(val_prob, val_true)
    _, macro_f1, macro_prec, macro_rec = c002b.evaluate_at_thresholds(
        val_prob, val_true, opt_t)
    f1_05 = float(np.mean([
        f1_score(val_true[:, i], (val_prob[:, i] >= 0.5).astype(int), zero_division=0)
        for i in range(len(label_cols)) if val_true[:, i].sum() > 0]))
    macro_auc = float(np.mean([
        roc_auc_score(val_true[:, i], val_prob[:, i])
        for i in range(len(label_cols)) if val_true[:, i].sum() > 0]))
    beat = f1_05 > C002B_F1_05
    print(f"  F1(0.5)={f1_05:.4f} F1(100)={macro_f1:.4f} R={macro_rec:.4f} "
          f"P={macro_prec:.4f} AUC={macro_auc:.4f}")
    print(f"  vs C002b F1(0.5)={C002B_F1_05}: {'提升' if beat else '未超'}")

    y_test = c002b.predict_with_thresholds(model, test_loader, opt_t)
    sub = pd.concat([test_df[["protein_id"]],
                     pd.DataFrame(y_test, columns=label_cols)], axis=1)
    sub.to_csv(SUB_PATH, index=False)
    np.savez_compressed(PROBS_PATH, val=val_prob, y_true=val_true, thresholds=opt_t)
    torch.save({"state_dict": model.state_dict(), "thresholds": opt_t,
                "label_cols": label_cols}, MODEL_PATH)
    report = {
        "in_ch": IN_CH,
        "n_params": n_params,
        "f1_05": round(f1_05, 6),
        "f1_100": round(macro_f1, 6),
        "recall_100": round(macro_rec, 6),
        "precision_100": round(macro_prec, 6),
        "auc": round(macro_auc, 6),
        "best_epoch": best_epoch,
        "beat_c002b": bool(beat),
        "c002b_f1_05": C002B_F1_05,
        "seconds": round(time.time() - t_start),
        "submission": SUB_PATH,
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  提交: {SUB_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
