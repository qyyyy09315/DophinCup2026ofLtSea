
"""
esm2-finetune — ESM-2 150M 微调 + 多标签分类头
================================================
方案:
  - 本地 ESM-2 150M (models/esm2_150m) + Linear(640, 500)
  - 8GB 显存配置: 冻结前 20 层, 微调后 10 层 + 分类头
  - AMP fp16, batch 8 + 梯度累积 4 = 有效 batch 32
  - 均值池化: 对非特殊残基的 last_hidden_state 取平均
  - 训练后逐标签阈值网格搜索 (验证集 F1 最大化)
用法:
  python esm2-finetune.py               # 正常训练
  python esm2-finetune.py --eval-only   # 加载已保存模型直接评估
输入:
  data/train.csv, data/test.csv, models/esm2_150m
输出:
  submit/submission_ESM2_finetune.csv
  rel/ESM2_finetune_val_metrics.json
  models/esm2-finetune.pt
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import EsmTokenizer, EsmModel, get_linear_schedule_with_warmup
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_DIR = "models/esm2_150m"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
TOKEN_CACHE_TRAIN = f"{REL_DIR}/esm2_tokens_train.npz"
TOKEN_CACHE_TEST = f"{REL_DIR}/esm2_tokens_test.npz"
SUB_PATH = f"{SUB_DIR}/submission_ESM2_finetune.csv"
METRICS_PATH = f"{REL_DIR}/ESM2_finetune_val_metrics.json"
MODEL_PATH = "models/esm2-finetune.pt"

MAX_LEN = 512
TOKEN_BATCH = 1024
BATCH_SIZE = 8
GRAD_ACCUM = 4          # 有效 batch = 8 * 4 = 32
N_EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 2
FREEZE_FIRST_LAYERS = 20
DROPOUT = 0.1
POS_WEIGHT_CLIP = 50.0
THRESHOLD_STEPS = 50
VAL_SIZE = 0.2
RANDOM_STATE = 42

EVAL_ONLY = "--eval-only" in sys.argv
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_ctx():
    if DEVICE.type == "cuda":
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()


def tokenize_all(tokenizer, sequences, desc):
    n = len(sequences)
    input_ids = np.zeros((n, MAX_LEN), dtype=np.int32)
    attention_mask = np.zeros((n, MAX_LEN), dtype=np.int8)
    n_batches = (n + TOKEN_BATCH - 1) // TOKEN_BATCH
    pbar = tqdm(total=n_batches, desc=desc, unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for i in range(0, n, TOKEN_BATCH):
        chunk = sequences[i:i + TOKEN_BATCH]
        enc = tokenizer(list(chunk), padding="max_length", max_length=MAX_LEN,
                        truncation=True, return_tensors="np")
        input_ids[i:i + len(chunk)] = enc["input_ids"]
        attention_mask[i:i + len(chunk)] = enc["attention_mask"]
        pbar.update(1)
    pbar.close()
    return input_ids, attention_mask


def load_or_tokenize(tokenizer, sequences, cache_path, desc):
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached["input_ids"].shape[0] == len(sequences) and cached["input_ids"].shape[1] == MAX_LEN:
            print(f"  加载缓存: {cache_path}")
            return (cached["input_ids"].astype(np.int32),
                    cached["attention_mask"].astype(np.int8))
    print(f"  开始分词: {desc}")
    input_ids, attention_mask = tokenize_all(tokenizer, sequences, desc)
    os.makedirs(REL_DIR, exist_ok=True)
    np.savez_compressed(cache_path, input_ids=input_ids, attention_mask=attention_mask)
    print(f"  缓存已保存: {cache_path}")
    return input_ids, attention_mask


class ProteinTokenDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.input_ids[idx]).long(),
                torch.from_numpy(self.attention_mask[idx]).long(),
                torch.from_numpy(self.labels[idx]).float())


def mean_pool(hidden, input_ids, attention_mask):
    valid = (attention_mask.unsqueeze(-1) == 1) & (input_ids.unsqueeze(-1) >= 5)
    denom = valid.float().sum(dim=1).clamp(min=1)
    return (hidden.float() * valid.float()).sum(dim=1) / denom


class Esm2Finetune(nn.Module):
    def __init__(self, model_dir, n_labels, freeze_first_layers=FREEZE_FIRST_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.esm = EsmModel.from_pretrained(model_dir)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.esm.config.hidden_size, n_labels)
        if freeze_first_layers > 0:
            self.esm.requires_grad_(False)
            for layer in self.esm.encoder.layer[freeze_first_layers:]:
                layer.requires_grad_(True)

    def forward(self, input_ids, attention_mask):
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = mean_pool(out, input_ids, attention_mask)
        return self.classifier(self.dropout(pooled))


@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        with autocast_ctx():
            logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
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
    for input_ids, attention_mask, _ in loader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        with autocast_ctx():
            logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        batch_preds = np.zeros_like(probs, dtype=np.int8)
        for i, t in enumerate(thresholds):
            batch_preds[:, i] = (probs[:, i] >= t).astype(np.int8)
        preds.append(batch_preds)
    return np.vstack(preds)

def train_epoch(model, loader, optimizer, criterion, scaler, scheduler):
    model.train()
    total_loss = 0.0
    n_samples = 0
    pbar = tqdm(loader, desc="Epoch", unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for step, (input_ids, attention_mask, labels) in enumerate(pbar, 1):
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)
        with autocast_ctx():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
        scaled_loss = loss / GRAD_ACCUM
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        total_loss += loss.item() * input_ids.size(0)
        n_samples += input_ids.size(0)

        if step % GRAD_ACCUM == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    if len(loader) % GRAD_ACCUM != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    pbar.close()
    return total_loss / max(n_samples, 1)


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
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

    print("[2/8] 加载 ESM-2 和分词器...")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_DIR)

    print("[3/8] 分词并缓存...")
    train_ids, train_mask = load_or_tokenize(
        tokenizer, train_df["sequence"].values, TOKEN_CACHE_TRAIN, "train")
    test_ids, test_mask = load_or_tokenize(
        tokenizer, test_df["sequence"].values, TOKEN_CACHE_TEST, "test")
    print(f"  token 矩阵: train {train_ids.shape}, test {test_ids.shape}")

    print("[4/8] 划分训练/验证集...")
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=VAL_SIZE, random_state=RANDOM_STATE)
    train_ds = ProteinTokenDataset(train_ids[idx_train], train_mask[idx_train], y_full[idx_train])
    val_ds = ProteinTokenDataset(train_ids[idx_val], train_mask[idx_val], y_full[idx_val])
    test_ds = ProteinTokenDataset(
        test_ids, test_mask,
        np.zeros((len(test_df), len(label_cols)), dtype=np.int8))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=True, num_workers=0)
    print(f"  训练: {len(train_ds)}, 验证: {len(val_ds)}")

    print("[5/8] 构建微调模型...")
    model = Esm2Finetune(MODEL_DIR, len(label_cols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {n_params:,}, 可训练: {n_trainable:,}")
    print(f"  冻结前 {FREEZE_FIRST_LAYERS} 层, 微调后 {model.esm.config.num_hidden_layers - FREEZE_FIRST_LAYERS} 层 + 分类头")

    pos = y_full[idx_train].sum(axis=0).astype(np.float32)
    neg = len(idx_train) - pos
    pos_weight = np.ones(len(label_cols), dtype=np.float32)
    mask = pos > 0
    pos_weight[mask] = np.clip(neg[mask] / pos[mask], 1.0, POS_WEIGHT_CLIP)
    pw_tensor = torch.tensor(pos_weight, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = (len(train_loader) + GRAD_ACCUM - 1) // GRAD_ACCUM
    total_steps = steps_per_epoch * N_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    if EVAL_ONLY:
        print("[6/8] 加载已保存模型并评估...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"未找到 {MODEL_PATH}")
        load_checkpoint(model, MODEL_PATH)
        best_epoch = None
        best_f1_05 = None
    else:
        print("[6/8] 微调训练...")
        best_f1, best_epoch, patience_counter = 0, 0, 0
        for epoch in range(1, N_EPOCHS + 1):
            t_ep = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, scheduler)
            val_prob, val_true = get_probs(model, val_loader)
            _, val_f1, _, _ = evaluate_at_thresholds(
                val_prob, val_true, np.full(len(label_cols), 0.5))
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

    print("[7/8] 阈值搜索与最终评估...")
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

    print("[8/8] 保存结果...")
    os.makedirs(SUB_DIR, exist_ok=True)
    y_test_pred = predict_with_thresholds(model, test_loader, opt_thresholds)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    torch.save({"state_dict": model.state_dict(), "thresholds": opt_thresholds,
                "label_cols": label_cols}, MODEL_PATH)
    print(f"  模型: {MODEL_PATH}")

    metrics = {
        "model": "esm2-finetune",
        "eval_only": EVAL_ONLY,
        "features": "ESM-2 token ids (max_len=512)",
        "classifier": "ESM-2 last 10 layers + Linear head",
        "hyperparams": {
            "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
            "epochs": N_EPOCHS, "patience": PATIENCE,
            "freeze_first_layers": FREEZE_FIRST_LAYERS, "dropout": DROPOUT,
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
        "n_labels_total": len(label_cols),
        "n_params": n_params, "n_trainable": n_trainable,
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
    with Path(METRICS_PATH).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  指标: {METRICS_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
