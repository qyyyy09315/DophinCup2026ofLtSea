
"""
T002-rope — 仅 RoPE 位置编码 Transformer（T002，容量同 B007）
================================================================
方案:
  - 在 B007 基础上仅替换位置编码: 可学习 Embedding → RoPE
  - 容量与 B007 完全相同 (d_model 128, 4 层, d_ff 512, batch 128,
    lr 1e-3, 30 epochs, patience 6) → 与 B007 严格可比
  - 目的: 单独验证 RoPE 在蛋白序列任务上是否有收益
    (T001 同时改 RoPE + 扩容, 变量未拆分无法归因)
  - 接口与 B007/T001 一致 (TransformerClassifier / get_probs / ProteinDataset)
用法:
  python T002-rope.py               # 正常训练
  python T002-rope.py --eval-only   # 加载已保存模型直接评估
输出:
  submit/submission_T002.csv
  rel/T002_val_metrics.json
  models/t002-rope.pt
"""

import contextlib
import json
import math
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
SUB_PATH = f"{SUB_DIR}/submission_T002.csv"
METRICS_PATH = f"{REL_DIR}/T002_val_metrics.json"
MODEL_PATH = f"{MODEL_DIR}/t002-rope.pt"

MAX_LEN = 512
BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 1
N_EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 6
LR_PATIENCE = 3
LR_FACTOR = 0.5
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 8
D_FF = 512
DROPOUT = 0.1
ROPE_BASE = 10000.0
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
        return (torch.from_numpy(self.seqs[idx]).long(),
                torch.from_numpy(self.labels[idx]).float())


class RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码 (RoFormer, arXiv:2104.09864)。
    前后半配对旋转 (HF/LLaMA 风格): 维度 i 与 i+dim/2 配对, 角度 theta_i = base^(-2i/dim)。"""
    def __init__(self, dim, max_len, base=ROPE_BASE):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_len, dim//2)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class TAttention(nn.Module):
    """多头自注意力 + RoPE (norm_first 层内使用)。"""
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim, MAX_LEN)

    def forward(self, x, key_padding_mask):
        bsz, seq, _ = x.shape
        q = self.wq(x).view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(seq)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,seq,head_dim//2)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
        # SDPA bool attn_mask: True=参与注意力; key_padding_mask True=pad → 取反
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask.bool()).unsqueeze(1).unsqueeze(2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(bsz, seq, -1)
        return self.out(out)


class TBlock(nn.Module):
    """Transformer 块 (norm_first + gelu，与 B007 一致)。"""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask):
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerClassifier(nn.Module):
    """类名/接口与 B007 保持一致 (transformer-cls.py)，便于集成脚本复用。"""
    def __init__(self, n_labels=500, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT, max_len=MAX_LEN):
        super().__init__()
        self.embed = nn.Embedding(N_CHANNELS, d_model, padding_idx=PAD_IDX)
        self.blocks = nn.ModuleList(
            [TBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_labels)
        self.d_model = d_model

    def forward(self, x, mask):
        h = self.embed(x) * math.sqrt(self.d_model)
        key_padding_mask = ~mask.bool()
        for block in self.blocks:
            h = block(h, key_padding_mask)
        h = self.final_norm(h)
        valid = mask.unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return self.head(pooled)


@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        mask = (x != PAD_IDX).long()
        with autocast_ctx():
            logits = model(x, mask)
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
        mask = (x != PAD_IDX).long()
        with autocast_ctx():
            logits = model(x, mask)
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
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="Epoch", unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for step, (x, y) in enumerate(pbar, 1):
        x = x.to(DEVICE)
        mask = (x != PAD_IDX).long()
        y = y.to(DEVICE)
        with autocast_ctx():
            logits = model(x, mask)
            loss = criterion(logits, y) / GRAD_ACCUM_STEPS
        if scaler is not None:
            scaler.scale(loss).backward()
            if step % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if step % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * x.size(0) * GRAD_ACCUM_STEPS
        n_samples += x.size(0)
        pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}")
    # 处理尾部不足一个累积周期的梯度 (仅当存在未 step 的累积梯度)
    if len(loader) % GRAD_ACCUM_STEPS != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
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

    print("[1/7] 读取数据...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    print(f"  训练集: {len(train_df)}, 测试集: {len(test_df)}, 标签: {len(label_cols)}")

    print("[2/7] 划分训练/验证集...")
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
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=True, num_workers=0)
    print(f"  训练: {len(train_ds)}, 验证: {len(val_ds)}")

    print("[3/7] 构建 RoPE Transformer 分类模型 (T002)...")
    model = TransformerClassifier(n_labels=len(label_cols)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}, d_model={D_MODEL}, layers={N_LAYERS}, "
          f"heads={N_HEADS}, d_ff={D_FF}, RoPE base={ROPE_BASE}")

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
        print("[4/7] 加载已保存模型并评估...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"未找到 {MODEL_PATH}")
        load_checkpoint(model, MODEL_PATH)
        best_epoch = None
        best_f1_05 = None
    else:
        print("[4/7] 训练...")
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

    print("[5/7] 阈值搜索与最终评估...")
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

    print("[6/7] 测试集预测并保存提交...")
    os.makedirs(SUB_DIR, exist_ok=True)
    y_test_pred = predict_with_thresholds(model, test_loader, opt_thresholds)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_test_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    print("[7/7] 保存模型与指标...")
    torch.save({"state_dict": model.state_dict(), "thresholds": opt_thresholds,
                "label_cols": label_cols}, MODEL_PATH)
    print(f"  模型: {MODEL_PATH}")

    metrics = {
        "model": "t002-rope",
        "eval_only": EVAL_ONLY,
        "features": "amino acid index encoding (max_len=512)",
        "classifier": "RoPE TransformerEncoder + mean pooling + Linear head",
        "references": [
            "RoFormer: Su et al., arXiv:2104.09864",
        ],
        "hyperparams": {
            "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "epochs": N_EPOCHS,
            "patience": PATIENCE, "lr_patience": LR_PATIENCE,
            "lr_factor": LR_FACTOR, "d_model": D_MODEL, "n_layers": N_LAYERS,
            "n_heads": N_HEADS, "d_ff": D_FF, "dropout": DROPOUT,
            "rope_base": ROPE_BASE, "pos_weight_clip": POS_WEIGHT_CLIP,
            "threshold_steps": THRESHOLD_STEPS,
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
