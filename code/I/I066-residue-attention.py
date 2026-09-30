"""
I066 — 残基级注意力池化支路 (冻结 ESM2 隐层 256x640)
=====================================================
文献依据:
  - Locality-aware pooling (Hoang & Singh, Bioinformatics 2025, doi:10.1093/bioinformatics/btaf178):
    注意力池化在 17 项实验中胜出 15 项, 优于 CLS/均值池化。
  - Light Attention (Stärk et al., Bioinformatics Advances 2021, doi:10.1093/bioadv/vbab035):
    冻结 PLM 残基嵌入 + 浅层注意力头即可匹配基于 MSA 的方法。
  - CSRA (Zhu & Wu, ICCV 2021, doi:10.1109/iccv48922.2021.00025): 类别特异性残差注意力。
设计:
  输入 = E004a 冻结 ESM2 残基隐层 (L=256, d=640), mmap 流式读取避免 37GB 驻留内存。
  头 = 投影 -> 8 头注意力池化 + 残基 max 池化 -> 拼接 -> 分类头。
  损失 = ASL (同 C002b/I060)。
  产出 = 与现有 1664D 均值池化支路"异质"的新分支（不同读出机制 -> 高多样性）。
用法: python code/I/I066-residue-attention.py
输出: rel/resattn_val_probs.npz / rel/resattn_test_probs.npz / models/resattn*.pt
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
TRAIN_H = os.path.join(REL_DIR, "E004a_esm_hidden_train.npy")
TRAIN_M = os.path.join(REL_DIR, "E004a_esm_hidden_train.mask.npy")
TEST_H = os.path.join(REL_DIR, "E004a_esm_hidden_test.npy")
TEST_M = os.path.join(REL_DIR, "E004a_esm_hidden_test.mask.npy")
VAL_OUT = os.path.join(REL_DIR, "resattn_val_probs.npz")
TEST_OUT = os.path.join(REL_DIR, "resattn_test_probs.npz")
N_LABELS = 500
EPOCHS = 30
BATCH = 192
LR = 8e-4
SEEDS = [7]


class MemmapDS(Dataset):
    """mmap 流式读取残基隐层，避免 37GB 驻留内存。"""

    def __init__(self, hid_path, mask_path, indices, y=None):
        self.hid = np.load(hid_path, mmap_mode="r")
        self.mask = np.load(mask_path, mmap_mode="r")
        self.idx = np.asarray(indices)
        self.y = y

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        h = np.asarray(self.hid[j], dtype=np.float32)   # (256, 640)
        m = np.asarray(self.mask[j], dtype=np.float32)  # (256,)
        if self.y is None:
            return torch.from_numpy(h), torch.from_numpy(m)
        return torch.from_numpy(h), torch.from_numpy(m), torch.from_numpy(self.y[j])


class ResidueAttentionHead(nn.Module):
    """Light Attention + 多头注意力池化 + 残基 max 池化 (互补)。"""

    def __init__(self, d_in=640, d_model=512, n_heads=8, n_out=N_LABELS, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.drop = nn.Dropout(dropout)
        # 注意力池化 (d_model) + max 池化 (d_model) -> 2*d_model
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, n_out))

    def forward(self, h, m):
        x = self.proj(h)                                  # (B, L, d)
        B, L, _ = x.shape
        xh = x.view(B, L, self.n_heads, self.head_dim)
        scores = torch.einsum("blhd,hd->bhl", xh, self.query)
        scores = scores.masked_fill(m[:, None, :] == 0, -1e4)
        w = torch.softmax(scores, dim=-1)
        attn = torch.einsum("bhl,blhd->bhd", w, xh).reshape(B, -1)     # 注意力池化
        xm = x.masked_fill(m[:, :, None] == 0, -1e4)
        mx = xm.max(dim=1).values                                       # max 池化
        return self.head(self.drop(torch.cat([attn, mx], dim=-1)))


class ASL(nn.Module):
    def forward(self, lg, tg):
        p = torch.sigmoid(lg.float()).clamp(1e-6, 1 - 1e-6)
        pos = -tg * torch.log(p)
        pm = torch.clamp(p - 0.05, min=0.0)
        neg = -(1 - tg) * (pm ** 4.0) * torch.log(1 - pm)
        return (pos + neg).mean()


def search_scut(ens, y, n=60):
    cands = np.linspace(0.01, 0.99, n)
    best = np.full(N_LABELS, 0.5)
    for i in range(N_LABELS):
        yt = y[:, i]
        if yt.sum() == 0:
            continue
        pr = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0); fp = (pr & ~ytm).sum(0); fn = (~pr & ytm).sum(0)
        best[i] = cands[(2*tp/np.maximum(2*tp+fp+fn, 1e-9)).argmax()]
    return best


def macro_f1(probs, y, ths):
    preds = (probs >= ths).astype(bool); yb = y.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
    p = np.where(tp+fp>0, tp/(tp+fp), 0.0); r = np.where(tp+fn>0, tp/(tp+fn), 0.0)
    return float(np.mean(np.where(p+r>0, 2*p*r/(p+r), 0.0)))


@torch.no_grad()
def predict(model, loader):
    model.eval()
    out = []
    for batch in loader:
        h, m = batch[0], batch[1]
        h, m = h.cuda(non_blocking=True), m.cuda(non_blocking=True)
        with torch.amp.autocast("cuda"):
            lg = model(h, m)
        out.append(torch.sigmoid(lg.float()).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    t0 = time.time()
    print("=" * 72)
    print("  I066: Residue-Level Attention Pooling on Frozen ESM2 Hidden (256x640)")
    print("=" * 72)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.float32)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)

    tr_ds = MemmapDS(TRAIN_H, TRAIN_M, idx_tr, y_full)
    va_ds = MemmapDS(TRAIN_H, TRAIN_M, idx_va, y_full)
    te_ds = MemmapDS(TEST_H, TEST_M, np.arange(28450), None)
    print(f"  Train {len(tr_ds)} | Val {len(va_ds)} | Test {len(te_ds)} (mmap, no RAM blow-up)")

    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    y_va = y_full[idx_va]
    seed_probs = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
        model = ResidueAttentionHead().cuda()
        n_par = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\n--- seed{seed} | params {n_par:.2f}M ---")
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
        crit = ASL(); sc = torch.amp.GradScaler("cuda")
        best_f1, best_p = -1, None
        for ep in range(EPOCHS):
            model.train()
            tot = 0.0; nb = 0
            for h, m, y in tr_loader:
                h, m, y = h.cuda(non_blocking=True), m.cuda(non_blocking=True), y.cuda(non_blocking=True)
                opt.zero_grad()
                with torch.amp.autocast("cuda"):
                    loss = crit(model(h, m), y)
                sc.scale(loss).backward(); sc.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                sc.step(opt); sc.update()
                tot += loss.item(); nb += 1
            sched.step()
            if (ep+1) % 5 == 0 or ep == EPOCHS - 1:
                p = predict(model, va_loader)
                f1 = macro_f1(p, y_va, search_scut(p, y_va))
                auc = roc_auc_score(y_va, p, average="macro")
                if f1 > best_f1:
                    best_f1, best_p = f1, p
                    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"resattn_s{seed}.pt"))
                el = (time.time()-t0)/60
                print(f"  seed{seed} ep{ep+1:02d} loss={tot/max(nb,1):.4f} | F1={f1:.4f} AUC={auc:.4f} "
                      f"| best {best_f1:.4f} | {el:.1f}min", flush=True)
        seed_probs.append(best_p)
        print(f"  seed{seed} FINAL F1={best_f1:.4f}")
        # 保存该种子测试概率
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"resattn_s{seed}.pt"), weights_only=True))
        te_p = predict(model, te_loader)
        np.savez_compressed(os.path.join(REL_DIR, f"resattn_test_s{seed}.npz"), test=te_p)
        print(f"  seed{seed} test probs saved")

    ens_v = np.mean(seed_probs, axis=0)
    ens_f1 = macro_f1(ens_v, y_va, search_scut(ens_v, y_va))
    ens_auc = roc_auc_score(y_va, ens_v, average="macro")
    np.savez_compressed(VAL_OUT, val=ens_v)
    print("\n" + "=" * 72)
    print(f"  Standalone: seeds {[f'{macro_f1(p, y_va, search_scut(p, y_va)):.4f}' for p in seed_probs]}")
    print(f"  Residue-attention branch: F1={ens_f1:.4f} AUC={ens_auc:.4f}")
    print(f"  (1664D mean-pooled multi-seed was F1 0.2729 / AUC 0.7947)")
    print("=" * 72)
    print(f"\nCompleted in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
