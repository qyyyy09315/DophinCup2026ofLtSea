"""
C003c-interlabel-cnn — 候选C: InterLabelGO 式标签依赖损失重训 CNN
==================================================================
依据 (文献深读):
  - InterLabelGO+ (Liu, Zhang, Freddolino. Bioinformatics 40(11):btae655, 2024,
    doi:10.1093/bioinformatics/btae655; CAFA5 顶尖, GitHub QuanEvans/InterLabelGO):
    ESM2+NN + DIAMOND 同源; 核心差异 = 训练损失显式建模「标签不平衡 + 标签间依赖」,
    而非输出层事后传播(我们 TCK002b/ECK002 失败因是后者)
  - 损失机制(读源码确认):
    total = MLCE × protein_centric_softF1 × go_centric_softF1
      · MLCE: 多标签分类交叉熵 (等价 multi-label categorical crossentropy)
      · protein_centric_softF1: 每蛋白的软 F1, 可微, 直接对齐 Macro F1 目标
      · go_centric_softF1: 每个 GO 项的软 F1 (标签级)
      · weight_tensor: GO 项信息含量加权 (我们无 GO IC → 用标签训练频率逆, 处理稀疏)
方案:
  - 结构: 复用 C002b-cnn.py 的网络/数据/80epoch 配置 (逐标签阈值搜索同款)
  - 仅替换损失: ASL → InterLabelLoss (MLCE + 双 soft-F1)
  - weight: w_label = 1/sqrt(pos_freq + eps), 归一化 (稀疏标签高权重)
  - 评估: 等 C002b 的 val F1(0.5)/100档/AUC + 提交文件
  - 对照: C002bbase-asl F1=0.2714; 若 interlabel 独立 F1 提升 → 进 ECK001 集成
用法:
  python code/C/C003c-interlabel-cnn.py [--epochs N] [--tag TAG]
输出:
  models/c003c-interlabel.pt
  rel/C003c_val_metrics.json
  submit/submission_C003c_interlabel.csv
"""

import contextlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SRC = os.path.dirname(os.path.abspath(__file__))
# 加载 C002b 的网络与数据配置 (import 需要规范化路径)
sys.path.insert(0, SRC)
import importlib.util
spec = importlib.util.spec_from_file_location("c002b", os.path.join(SRC, "C002b-cnn.py"))
c002b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c002b)

PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "c003c-interlabel.pt")
REPORT_PATH = f"{REL_DIR}/C003c_val_metrics.json"
SUB_PATH = f"{SUB_DIR}/submission_C003c_interlabel.csv"

# 复用 C002b 的超参/结构 (epochs/lr 可 --tag 覆盖)
N_EPOCHS = c002b.N_EPOCHS
BATCH = c002b.BATCH_SIZE
LR = c002b.LR


# ---------------- InterLabelLoss (移植自 InterLabelGO, 适配匿名标签) ----------------
class InterLabelLoss(nn.Module):
    """MLCE × protein_softF1 × go_softF1。weight = 标签频率逆加权。"""

    def __init__(self, weight=None, device="cuda"):
        super().__init__()
        self.device = device
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32) if weight is not None else None)

    def forward(self, y_pred, y_true, print_loss=False):
        sig = torch.sigmoid(y_pred.float())
        y_true = y_true.float()
        w = self.weight if self.weight is not None else torch.ones(y_pred.shape[1], device=y_pred.device)
        mlce = self.multilabel_cce(y_pred, y_true)
        go_f1 = self.go_centric_f1(sig, y_true, w)
        prot_f1 = self.protein_centric_f1(sig, y_true, w)
        total = mlce * prot_f1 * go_f1
        if print_loss:
            print(f"    MLCE={mlce.item():.4f} goF1={go_f1.item():.4f} protF1={prot_f1.item():.4f} total={total.item():.4f}")
        return total

    def multilabel_cce(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred
        neg = y_pred - y_true * 1e16
        pos = y_pred - (1 - y_true) * 1e16
        z = torch.zeros_like(y_pred[..., :1])
        neg = torch.cat([neg, z], -1); pos = torch.cat([pos, z], -1)
        return torch.mean(torch.logsumexp(neg, -1) + torch.logsumexp(pos, -1))

    def go_centric_f1(self, sig, y_true, w):
        tp = torch.sum(y_true * sig * w, dim=0)
        fp = torch.sum((1 - y_true) * sig * w, dim=0)
        fn = torch.sum(y_true * (1 - sig) * w, dim=0)
        prec = tp / (tp + fp + 1e-16)
        rec = tp / (tp + fn + 1e-16)
        f1 = 2 * prec * rec / (prec + rec + 1e-16)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        return 1 - f1.mean()

    def protein_centric_f1(self, sig, y_true, w):
        tp = torch.sum(y_true * sig * w, dim=1)
        fp = torch.sum((1 - y_true) * sig * w, dim=1)
        fn = torch.sum(y_true * (1 - sig) * w, dim=1)
        prec = tp / (tp + fp + 1e-16)
        rec = tp / (tp + fn + 1e-16)
        f1 = 2 * prec * rec / (prec + rec + 1e-16)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        return 1 - f1.mean()


def build_weight(y_train_pos_freq):
    """标签频率逆加权 (稀疏→高权重), sqrt 压缩 + 归一化。"""
    w = 1.0 / np.sqrt(y_train_pos_freq + 1e-6)
    w = w / w.sum() * len(w)   # 均值为 1
    return w.astype(np.float32)


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total = 0.0; n = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to("cuda"), y.to("cuda")
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * x.size(0); n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, y_true, cands):
    model.eval()
    probs = []
    for x, _ in tqdm(loader, desc="eval", leave=False):
        x = x.to("cuda")
        with torch.cuda.amp.autocast():
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    p = np.concatenate(probs)
    best_t = np.full(y_true.shape[1], 0.5)
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() == 0:
            continue
        m = p[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (m & ytm).sum(0); fp = (m & ~ytm).sum(0); fn = (~m & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_t[i] = cands[f1t.argmax()]
    r100, p100 = [], []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        pred = p[:, i] >= best_t[i]
        tp = int((pred & yt).sum()); fp = int((pred & ~yt).sum()); fn = int((~pred & yt).sum())
        r100.append(tp / (tp + fn) if tp + fn else 0.0)
        p100.append(tp / (tp + fp) if tp + fp else 0.0)
    auc = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        auc.append(roc_auc_score(yt, p[:, i]))
    f1_100 = np.mean([f1_score(y_true[:, i].astype(bool), p[:, i] >= best_t[i], zero_division=0)
                      for i in range(y_true.shape[1]) if y_true[:, i].sum() > 0])
    return {"f1_05": float(np.mean([f1_score(y_true[:, i].astype(bool), p[:, i] >= 0.5, zero_division=0)
                                    for i in range(y_true.shape[1]) if y_true[:, i].sum() > 0])),
            "f1_100": float(f1_100), "recall_100": float(np.mean(r100)),
            "precision_100": float(np.mean(p100)), "auc": float(np.mean(auc)), "probs": p, "best_t": best_t}


def main():
    # 解析 --epochs/--tag
    n_epochs = N_EPOCHS
    tag = ""
    for a in sys.argv:
        if a.startswith("--epochs="):
            n_epochs = int(a.split("=")[1])
        if a.startswith("--tag="):
            tag = "_" + a.split("=")[1]
    lazy = "--lazy" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print(f"C003c InterLabelLoss CNN | epochs={n_epochs} tag={tag or 'default'}")

    # 数据 (复用 C002b 的 prep 逻辑: uint8 one-hot 矩阵 + split)
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    seqs = train_df["sequence"].values
    n_labels = len(label_cols)
    print(f"  标签数: {n_labels}")

    from sklearn.model_selection import train_test_split
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=0.2, random_state=42)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    # 复用 C002b 的 encode_sequences + collate_onehot (通道/截断完全一致, 保证可比)
    cache = f"{REL_DIR}/c003c_tokens_train.npz"
    if lazy and os.path.exists(cache):
        d = np.load(cache)
        X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
        print(f"  token 读缓存 {X_train.shape}/{X_val.shape}/{X_test.shape}")
    else:
        X_train = c002b.encode_sequences([s for s in seqs[idx_train]], c002b.MAX_LEN)
        X_val = c002b.encode_sequences([s for s in seqs[idx_val]], c002b.MAX_LEN)
        X_test = c002b.encode_sequences(test_df["sequence"].values, c002b.MAX_LEN)
        np.savez_compressed(cache, X_train=X_train, X_val=X_val, X_test=X_test)
        print(f"  token: {X_train.shape}/{X_val.shape}/{X_test.shape}")

    tr_l = DataLoader(list(zip(torch.from_numpy(X_train).long(), torch.from_numpy(y_train.astype(np.float32)))),
                      batch_size=BATCH, shuffle=True, num_workers=0,
                      collate_fn=c002b.collate_onehot)
    va_l = DataLoader(list(zip(torch.from_numpy(X_val).long(), torch.from_numpy(y_val.astype(np.float32)))),
                      batch_size=BATCH, shuffle=False, num_workers=0,
                      collate_fn=c002b.collate_onehot)
    te_l = DataLoader(list(zip(torch.from_numpy(X_test).long(), torch.zeros(len(X_test), n_labels))),
                      batch_size=BATCH, shuffle=False, num_workers=0,
                      collate_fn=c002b.collate_onehot)

    # 模型 (C002b 结构)
    model = c002b.MultiScaleCNNv3(n_labels=n_labels).cuda()
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失 + 权重
    pos_freq = y_train.mean(axis=0)
    w = build_weight(pos_freq)
    print(f"  标签频率: min={pos_freq.min():.4f} max={pos_freq.max():.4f} | 权重归一后 mean={w.mean():.2f}")
    criterion = InterLabelLoss(weight=w).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = torch.cuda.amp.GradScaler()
    cands = np.linspace(0.01, 0.99, 100)

    best_f1 = -1
    for ep in range(n_epochs):
        tr_l_ = tr_l  # reuse
        loss = train_epoch(model, tr_l_, optimizer, criterion, scaler)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == n_epochs - 1:
            ev = evaluate(model, va_l, y_val, cands)
            print(f"  Epoch {ep+1:3d}/{n_epochs} | loss={loss:.4f} | F1(0.5)={ev['f1_05']:.4f} F1(100)={ev['f1_100']:.4f} AUC={ev['auc']:.4f}")
            if ev["f1_100"] > best_f1:
                best_f1 = ev["f1_100"]
                torch.save(model.state_dict(), MODEL_PATH)
    print(f"  最佳 F1(100)={best_f1:.4f} 模型: {MODEL_PATH}")

    # 最终评估 (val)
    ev = evaluate(model, va_l, y_val, cands)
    print(f"  最终 val: F1(0.5)={ev['f1_05']:.4f} F1(100)={ev['f1_100']:.4f} "
          f"R={ev['recall_100']:.4f} P={ev['precision_100']:.4f} AUC={ev['auc']:.4f}")

    # test 提交
    ev_test = evaluate(model, te_l, np.zeros((len(X_test), n_labels)), cands)
    y_pred = (ev_test["probs"] >= ev["best_t"]).astype(np.int8)
    os.makedirs(SUB_DIR, exist_ok=True)
    sub = test_df[["protein_id"]].copy()
    sub = pd.concat([sub, pd.DataFrame(y_pred, columns=label_cols)], axis=1)
    sub.to_csv(SUB_PATH, index=False)

    report = {"epochs": n_epochs, "loss": "InterLabelLoss(MLCE*softF1)", "weight": "label_freq_inverse",
              "f1_05": round(ev["f1_05"], 4), "f1_100": round(ev["f1_100"], 4),
              "recall_100": round(ev["recall_100"], 4), "precision_100": round(ev["precision_100"], 4),
              "auc": round(ev["auc"], 4), "best_f1_100": round(best_f1, 4),
              "c002b_asl_baseline_f1_05": 0.2714,
              "submission": SUB_PATH, "seconds": round(time.time() - t0)}
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  提交: {SUB_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
