"""
I062 — 多 PLM 特征扩容 (2176D) + 多种子残差 MLP 集成
=====================================================
依据:
  - Protein Dimension DB (bioRxiv 2025, doi:10.1101/2025.09.29.679219):
    ESM-2 + ProtT5 + Ankh 拼接优于最强单模型 (AUPRC 0.7641 -> 0.8006)。
  - I061 消融: 全局增益几乎全部来自 PLM-MLP (+0.0017)。
改动:
  - 特征: 640 (ESM2) + 1024 (ProtTrans) -> 再拼接 512 (R001 BiLSTM) = 2176D。
  - 多种子: 3 个 seed 独立训练后概率平均（方差约减，抗单次拟合偶然性）。
  - 架构与损失与 I060 一致（残差 MLP + LayerNorm + GELU + ASL）。
用法:
  python code/I/I062-multi-plm-mlp.py
输出:
  rel/multi_plm_mlp_val_probs.npz / _test_probs.npz
  models/multi_plm_mlp_seed{0,1,2}.pt
"""

import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
VAL_OUT = os.path.join(REL_DIR, "multi_plm_mlp_val_probs.npz")
TEST_OUT = os.path.join(REL_DIR, "multi_plm_mlp_test_probs.npz")

N_LABELS = 500
HIDDEN = 1024
DROPOUT = 0.2
EPOCHS = 35
BATCH = 512
LR = 1e-3
SEEDS = [0, 1, 2]


class PlmResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, out_dim=N_LABELS, dropout=DROPOUT):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.b1 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.a1 = nn.GELU()
        self.b2 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.a2 = nn.GELU()
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h0 = self.in_proj(x)
        h1 = self.a1(h0 + self.b1(h0))
        h2 = self.a2(h1 + self.b2(h1))
        return self.head(h2)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-6):
        super().__init__()
        self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits.float()).clamp(self.eps, 1.0 - self.eps)
        pos = -targets * ((1.0 - p) ** self.gp) * torch.log(p)
        p_m = torch.clamp(p - self.clip, min=0.0)
        neg = -(1.0 - targets) * (p_m ** self.gn) * torch.log(1.0 - p_m)
        return (pos + neg).mean()


def search_scut(ens, y):
    cands = np.linspace(0.01, 0.99, 60)
    best = np.full(N_LABELS, 0.5)
    for i in range(N_LABELS):
        yt = y[:, i]
        if yt.sum() == 0:
            continue
        pr = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0); fp = (pr & ~ytm).sum(0); fn = (~pr & ytm).sum(0)
        f1t = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
        best[i] = cands[f1t.argmax()]
    return best


def macro_f1(probs, y, ths):
    preds = (probs >= ths).astype(bool)
    yb = y.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
    p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return float(np.mean(f))


def build_features():
    """构造 2176D 特征: ESM2(640) + ProtTrans(1024) + R001(512)。"""
    print("  Loading parquet embeddings (ESM2 640D + ProtTrans 1024D)...")
    df_esm_tr = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    df_ptb_tr = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
    df_esm_te = pd.read_parquet(os.path.join(REL_DIR, "test_esm2_embeddings.parquet"))
    df_ptb_te = pd.read_parquet(os.path.join(REL_DIR, "test_embeddings.parquet"))

    esm_cols = [c for c in df_esm_tr.columns if c.startswith("emb_")]
    ptb_cols = [c for c in df_ptb_tr.columns if c.startswith("emb_")]
    label_cols = [c for c in df_esm_tr.columns if c.startswith("label_")]

    X_tr = np.concatenate([df_esm_tr[esm_cols].values, df_ptb_tr[ptb_cols].values], axis=1).astype(np.float32)
    X_te = np.concatenate([df_esm_te[esm_cols].values, df_ptb_te[ptb_cols].values], axis=1).astype(np.float32)
    Y = df_esm_tr[label_cols].values.astype(np.float32)

    # 拼接 R001 BiLSTM 512D (需重排到 train.csv 全量顺序)
    print("  Merging R001 BiLSTM 512D embeddings into full-order array...")
    r = np.load(os.path.join(REL_DIR, "R001_embeddings.npz"))
    n_all = len(Y)
    idx_train, idx_val = train_test_split(np.arange(n_all), test_size=0.2, random_state=42)
    X_r = np.zeros((n_all, r["emb_train"].shape[1]), dtype=np.float32)
    X_r[idx_train] = r["emb_train"]
    X_r[idx_val] = r["emb_val"]
    X_te_r = r["emb_test"].astype(np.float32)

    X_tr = np.concatenate([X_tr, X_r], axis=1)
    X_te = np.concatenate([X_te, X_te_r], axis=1)
    print(f"  Combined feature dim: {X_tr.shape[1]} (ESM2 640 + ProtTrans 1024 + R001 512)")
    return X_tr, X_te, Y, idx_train, idx_val, label_cols


def train_one_seed(seed, X_tr, y_tr, X_va, y_va, in_dim):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=BATCH, shuffle=True, pin_memory=True)
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=BATCH, shuffle=False, pin_memory=True)

    model = PlmResidualMLP(in_dim).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    crit = AsymmetricLoss()
    scaler = torch.amp.GradScaler("cuda")

    best_f1, best_probs, best_ep = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.cuda(non_blocking=True), by.cuda(non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(bx), by)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()

        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            model.eval()
            out = []
            with torch.no_grad():
                for bx, _ in va_loader:
                    with torch.amp.autocast("cuda"):
                        lg = model(bx.cuda(non_blocking=True))
                    out.append(torch.sigmoid(lg.float()).cpu().numpy())
            probs = np.concatenate(out, axis=0)
            f1 = macro_f1(probs, y_va, search_scut(probs, y_va))
            if f1 > best_f1:
                best_f1, best_probs, best_ep = f1, probs, ep + 1
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"multi_plm_mlp_seed{seed}.pt"))
            print(f"    seed{seed} ep{ep+1:02d} | F1={f1:.4f} (best {best_f1:.4f}@{best_ep})", flush=True)
    return best_f1, best_probs


def predict_test(seed, X_te, in_dim):
    model = PlmResidualMLP(in_dim).cuda()
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"multi_plm_mlp_seed{seed}.pt")))
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_te)), batch_size=BATCH, shuffle=False)
    out = []
    with torch.no_grad():
        for (bx,) in loader:
            with torch.amp.autocast("cuda"):
                lg = model(bx.cuda(non_blocking=True))
            out.append(torch.sigmoid(lg.float()).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    t0 = time.time()
    print("=" * 70)
    print("  I062: Multi-PLM 2176D Features + Multi-Seed Residual MLP Ensemble")
    print("=" * 70)

    X_tr_all, X_te_all, Y_all, idx_train, idx_val, label_cols = build_features()
    in_dim = X_tr_all.shape[1]
    X_tr, X_va = X_tr_all[idx_train], X_tr_all[idx_val]
    y_tr, y_va = Y_all[idx_train], Y_all[idx_val]

    print(f"\n  Train {X_tr.shape} | Val {X_va.shape} | Test {X_te_all.shape}")

    seed_val_probs, seed_test_probs, seed_f1s = [], [], []
    for seed in SEEDS:
        print(f"\n--- Training seed {seed} ---")
        f1, probs = train_one_seed(seed, X_tr, y_tr, X_va, y_va, in_dim)
        seed_f1s.append(f1)
        seed_val_probs.append(probs)
        auc = roc_auc_score(y_va, probs, average="macro")
        print(f"  seed{seed} FINAL: SCut F1={f1:.4f} | AUC={auc:.4f}")

    # 多种子平均
    ens_val = np.mean(seed_val_probs, axis=0)
    ens_f1 = macro_f1(ens_val, y_va, search_scut(ens_val, y_va))
    ens_auc = roc_auc_score(y_va, ens_val, average="macro")
    print("\n" + "=" * 70)
    print(f"  Single-seed F1: {[f'{f:.4f}' for f in seed_f1s]}")
    print(f"  Multi-seed ensemble F1 : {ens_f1:.4f} | AUC: {ens_auc:.4f}")
    print(f"  (I060's single PLM-MLP was F1 0.2590 / AUC 0.7839)")
    print("=" * 70)

    np.savez_compressed(VAL_OUT, val=ens_val)

    print("\n--- Predicting test set with all seeds ---")
    test_probs = [predict_test(s, X_te_all, in_dim) for s in SEEDS]
    ens_test = np.mean(test_probs, axis=0)
    np.savez_compressed(TEST_OUT, test=ens_test)
    print(f"  Saved {VAL_OUT} and {TEST_OUT} (shape {ens_test.shape})")
    print(f"\nCompleted in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
