import os, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "E"))
e006c = __import__("E006c-esm2-650-mah-asl")

print("1. Loading validation tokens and labels...")
tr_tok = np.load(os.path.join(PROJECT_ROOT, "rel", "esm2_tokens_train.npz"))
ids_tr = tr_tok["input_ids"][:, :e006c.MAX_LEN]
mask_tr = tr_tok["attention_mask"][:, :e006c.MAX_LEN]

train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train.csv"))
label_cols = [c for c in train_df.columns if c.startswith("label_")]
y_full = train_df[label_cols].values.astype(np.int8)

idx_train, idx_val = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
y_val = y_full[idx_val]

va_ds = e006c.TokenDS(torch.from_numpy(ids_tr[idx_val]).long(),
                      torch.from_numpy(mask_tr[idx_val]).long(),
                      torch.from_numpy(y_val.astype(np.float32)))
va_loader = torch.utils.data.DataLoader(va_ds, batch_size=16, shuffle=False, num_workers=0,
                                        collate_fn=e006c.collate_dyn, pin_memory=True)

print("2. Loading E006c checkpoint...")
base = e006c.load_esm()
model, _, _ = e006c.attach_lora(base, e006c.FREEZE_FIRST)
ckpt = torch.load(e006c.BEST_PATH, map_location="cuda", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model = model.cuda().eval()

print("3. Predicting validation probabilities...")
probs_list = []
with torch.no_grad():
    for ids, mask, _ in va_loader:
        ids, mask = ids.cuda(), mask.cuda()
        with torch.amp.autocast("cuda"):
            logits = model(input_ids=ids, attention_mask=mask)
        probs_list.append(torch.sigmoid(logits.float()).cpu().numpy())
probs = np.concatenate(probs_list, axis=0)

print(f"Probabilities shape: {probs.shape}, range: [{probs.min():.4f}, {probs.max():.4f}], mean: {probs.mean():.4f}")

# 保存 val probs 以备融合
np.savez_compressed(os.path.join(PROJECT_ROOT, "rel", "E006c_val_probs.npz"), val=probs)
print("Saved rel/E006c_val_probs.npz")

# 评估1: 固定阈值 0.5
f1_05 = e006c.macro_f1(probs, y_val, t=0.5)

# 评估2: 寻找最优单阈值 (Fmax)
cands = np.linspace(0.01, 0.50, 50)
best_single_f1 = -1.0
best_single_th = 0.5
for th in cands:
    f1 = e006c.macro_f1(probs, y_val, t=th)
    if f1 > best_single_f1:
        best_single_f1 = f1
        best_single_th = th

# 评估3: 逐标签自适应最优阈值 (SCut)
best_scut_th = np.zeros(500)
for j in range(500):
    yt = y_val[:, j].astype(bool)
    if yt.sum() == 0:
        best_scut_th[j] = 0.2
        continue
    p_j = probs[:, j]
    f1s = []
    for th in np.linspace(0.02, 0.60, 30):
        pred = p_j >= th
        tp = (pred & yt).sum()
        fp = (pred & ~yt).sum()
        fn = (~pred & yt).sum()
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    best_scut_th[j] = np.linspace(0.02, 0.60, 30)[np.argmax(f1s)]

scut_f1 = e006c.macro_f1(probs, y_val, t=best_scut_th)

# 评估4: 宏平均 ROC-AUC
auc = e006c.macro_auc(probs, y_val)

print("=" * 60)
print(f"E006c Checkpoint Evaluation Results:")
print(f"  Fixed Threshold F1(0.50): {f1_05:.4f}")
print(f"  Optimal Global Threshold: F1({best_single_th:.2f}) = {best_single_f1:.4f}")
print(f"  Per-Label Adaptive SCut : Macro F1 = {scut_f1:.4f}")
print(f"  Continuous Macro ROC-AUC: {auc:.4f}")
print("=" * 60)
PY