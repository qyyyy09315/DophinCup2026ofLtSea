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
MODEL_OUT = os.path.join(PROJECT_ROOT, "models", "fast_plm_mlp.pt")
VAL_PROBS_OUT = os.path.join(REL_DIR, "fast_plm_mlp_val_probs.npz")
TEST_PROBS_OUT = os.path.join(REL_DIR, "fast_plm_mlp_test_probs.npz")

print("1. Loading 1664-D multi-PLM concatenated embeddings (ESM2 + ProtTrans)...")
t0 = time.time()
df_esm_tr = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
df_ptb_tr = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
df_esm_te = pd.read_parquet(os.path.join(REL_DIR, "test_esm2_embeddings.parquet"))
df_ptb_te = pd.read_parquet(os.path.join(REL_DIR, "test_embeddings.parquet"))

esm_cols = [c for c in df_esm_tr.columns if c.startswith("emb_")]
ptb_cols = [c for c in df_ptb_tr.columns if c.startswith("emb_")]
label_cols = [c for c in df_esm_tr.columns if c.startswith("label_")]
n_labels = len(label_cols)

X_tr_all = np.concatenate([df_esm_tr[esm_cols].values, df_ptb_tr[ptb_cols].values], axis=1).astype(np.float32)
X_te_all = np.concatenate([df_esm_te[esm_cols].values, df_ptb_te[ptb_cols].values], axis=1).astype(np.float32)
Y_tr_all = df_esm_tr[label_cols].values.astype(np.float32)

print(f"  Combined Train feature shape: {X_tr_all.shape} (1664 dims)")
print(f"  Combined Test feature shape : {X_te_all.shape}")

idx_train, idx_val = train_test_split(
    np.arange(len(X_tr_all)), test_size=0.2, random_state=42
)
X_tr, X_va = X_tr_all[idx_train], X_tr_all[idx_val]
y_tr, y_va = Y_tr_all[idx_train], Y_tr_all[idx_val]

# 构建 PyTorch 数据集 (大 batch 在 GPU 上极快)
batch_size = 512
train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
test_ds = TensorDataset(torch.from_numpy(X_te_all))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

# 2. 定义深层残差多层感知机分类器 (带 LayerNorm、GELU、Dropout 与残差跳跃)
class PlmResidualMLP(nn.Module):
    def __init__(self, in_dim=1664, hidden_dim=1024, out_dim=500, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # Block 1
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.act1 = nn.GELU()
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.act2 = nn.GELU()
        
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h0 = self.in_proj(x)
        h1 = self.act1(h0 + self.block1(h0)) # residual 1
        h2 = self.act2(h1 + self.block2(h1)) # residual 2
        return self.head(h2)

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-6):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits.float()).clamp(self.eps, 1.0 - self.eps)
        pos_loss = -targets * ((1.0 - p) ** self.gamma_pos) * torch.log(p)
        p_m = torch.clamp(p - self.clip, min=0.0)
        neg_loss = -(1.0 - targets) * (p_m ** self.gamma_neg) * torch.log(1.0 - p_m)
        return (pos_loss + neg_loss).mean()

def search_scut(ens, y):
    cands = np.linspace(0.01, 0.99, 100)
    best = np.full(n_labels, 0.5, dtype=np.float64)
    for i in range(n_labels):
        yt = y[:, i]
        if yt.sum() == 0: continue
        pr = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0)
        fp = (pr & ~ytm).sum(0)
        fn = (~pr & ytm).sum(0)
        f1t = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
        best[i] = cands[f1t.argmax()]
    return best

def macro_f1(probs, targets, ths):
    preds = (probs >= ths).astype(bool)
    targs = targets.astype(bool)
    tp = (preds & targs).sum(0)
    fp = (preds & ~targs).sum(0)
    fn = (~preds & targs).sum(0)
    p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return float(np.mean(f))

print("\n2. Training Deep Residual PLM MLP on GPU...")
model = PlmResidualMLP(in_dim=1664, hidden_dim=1024, out_dim=500, dropout=0.2).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-5)
criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
scaler = torch.amp.GradScaler("cuda")

epochs = 35
best_f1 = -1.0
best_val_probs = None

for ep in range(epochs):
    model.train()
    total_loss = 0.0
    for bx, by in train_loader:
        bx, by = bx.cuda(non_blocking=True), by.cuda(non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(bx)
            loss = criterion(logits, by)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(bx)
    scheduler.step()
    
    # 评估
    if (ep + 1) % 5 == 0 or ep == epochs - 1:
        model.eval()
        preds_list = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.cuda(non_blocking=True)
                with torch.amp.autocast("cuda"):
                    logits = model(bx)
                preds_list.append(torch.sigmoid(logits.float()).cpu().numpy())
        va_probs = np.concatenate(preds_list, axis=0)
        ths = search_scut(va_probs, y_va)
        f1_curr = macro_f1(va_probs, y_va, ths)
        auc_curr = roc_auc_score(y_va, va_probs, average="macro")
        print(f"  Epoch {ep+1:02d}/{epochs} | Loss: {total_loss/len(X_tr):.4f} | Val SCut F1: {f1_curr:.4f} | AUC: {auc_curr:.4f}")
        if f1_curr > best_f1:
            best_f1 = f1_curr
            best_val_probs = va_probs
            torch.save(model.state_dict(), MODEL_OUT)

print(f"\n3. Best Standalone PLM-MLP Macro F1: {best_f1:.4f}")
np.savez_compressed(VAL_PROBS_OUT, val=best_val_probs)
print(f"Saved validation probabilities to {VAL_PROBS_OUT}")

# 4. 推理测试集
print("\n4. Predicting Test set...")
model.load_state_dict(torch.load(MODEL_OUT, weights_only=True))
model.eval()
test_preds_list = []
with torch.no_grad():
    for (bx,) in test_loader:
        bx = bx.cuda(non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(bx)
        test_preds_list.append(torch.sigmoid(logits.float()).cpu().numpy())
te_probs = np.concatenate(test_preds_list, axis=0)
np.savez_compressed(TEST_PROBS_OUT, test=te_probs)
print(f"Saved test probabilities to {TEST_PROBS_OUT} (shape: {te_probs.shape})")

print(f"\nAll completed in {time.time()-t0:.1f}s")