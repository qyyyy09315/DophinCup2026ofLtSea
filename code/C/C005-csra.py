"""
C005-csra — C002b 骨干 + 1D CSRA 类别特异残差注意力头
================================================================
依据:
  - CSRA: Zhu & Wu, ICCV 2021, doi:10.1109/iccv48922.2021.00025
    多标签不用共享 GAP：对每个类的空间打分做 softmax 加权池化，
    再与 GAP 残差相加。原文明 1×1 卷积出类热图，T 温度，λ 残差。
  - C002b 四路膨胀 CNN 在通道维拼接后对 500 标签共用 max+mean 池化。
    功能 motif 是局部的，共享 GAP 会抹掉类特异位置。
  - 不是 CK：只改 CNN 结构/头，不改 k-mer/比对融合。Mixup/CutMix/
    InterLabelLoss 已关；CSRA 是池化头，正交。
方案:
  骨干与 C002b 相同（4 分支 128 通道，k=3/5/7/9，d=1/2/3/4，SE+残差）。
  去掉共享 max/mean + Linear；拼接空间特征 (B,512,L) 后 1×1 Conv
  得到 (B,500,L)：logits = GAP + λ · Σ softmax(s/T) ⊙ s。
  ASL / 80ep / cosine / seed=42 不变。默认 T=1.0，λ=0.1（原文）。
  判断：独立 F1(0.5) 或 F1(100) 超 C002b 0.2714 才考虑换四路 CNN。
用法:
  python code/C/C005-csra.py
  python code/C/C005-csra.py --lam=0.3 --T=0.5
  python code/C/C005-csra.py --eval-only
输出:
  models/cnn-c005.pt
  rel/C005_report.json
  rel/C005_val_probs.npz
  submit/submission_C005.csv
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec_path = os.path.join(PROJECT_ROOT, "code", "C", "C002b-cnn.py")
import importlib.util
_spec = importlib.util.spec_from_file_location("c002b", _spec_path)
c002b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c002b)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUB_PATH = os.path.join(SUB_DIR, "submission_C005.csv")
REPORT_PATH = os.path.join(REL_DIR, "C005_report.json")
PROBS_PATH = os.path.join(REL_DIR, "C005_val_probs.npz")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn-c005.pt")

MAX_LEN = 1500
BATCH_SIZE = 256
N_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10
VAL_SIZE = 0.2
RANDOM_STATE = 42
CSRA_T = 1.0
CSRA_LAM = 0.1
C002B_F1_05 = 0.2714
DEVICE = c002b.DEVICE


class MultiScaleCNNCSRA(c002b.MultiScaleCNNv3):
    """C002b 骨干，头换成 1D CSRA（Zhu & Wu ICCV 2021）。"""

    def __init__(self, n_labels=500, T=CSRA_T, lam=CSRA_LAM, **kwargs):
        super().__init__(n_labels=n_labels, **kwargs)
        n_ch = c002b.FILTERS * len(c002b.KERNELS)
        self.head = nn.Conv1d(n_ch, n_labels, kernel_size=1, bias=True)
        self.T = float(T)
        self.lam = float(lam)
        del self.fc

    def forward(self, x):
        feats = []
        for branch, shortcut, se in zip(self.branches, self.shortcuts, self.ses):
            h = branch(x) + shortcut(x)
            h = se(h)
            feats.append(h)
        h = torch.cat(feats, dim=1)
        score = self.head(h)
        gap = score.mean(dim=2)
        att = torch.softmax(score / max(self.T, 1e-6), dim=2)
        csra = (score * att).sum(dim=2)
        return gap + self.lam * csra


def parse_args():
    T, lam = CSRA_T, CSRA_LAM
    eval_only = "--eval-only" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--T="):
            T = float(a.split("=", 1)[1])
        elif a.startswith("--lam="):
            lam = float(a.split("=", 1)[1])
    return T, lam, eval_only


def main():
    T, lam, eval_only = parse_args()
    t_start = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REL_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)
    print("=" * 60)
    print(f"C005 CSRA: T={T} lam={lam} epochs={N_EPOCHS} device={DEVICE} eval_only={eval_only}")

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
                              collate_fn=c002b.collate_onehot, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=c002b.collate_onehot, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=c002b.collate_onehot, pin_memory=True, num_workers=0)
    print(f"  训练 {len(train_ds)} 验证 {len(val_ds)}")

    model = MultiScaleCNNCSRA(n_labels=len(label_cols), T=T, lam=lam).to(DEVICE)
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
                torch.save({"state_dict": model.state_dict(), "T": T, "lam": lam,
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
    y_pred, macro_f1, macro_prec, macro_rec = c002b.evaluate_at_thresholds(
        val_prob, val_true, opt_t)
    f1_05 = float(np.mean([
        f1_score(val_true[:, i], (val_prob[:, i] >= 0.5).astype(int), zero_division=0)
        for i in range(len(label_cols)) if val_true[:, i].sum() > 0]))
    macro_auc = float(np.mean([
        roc_auc_score(val_true[:, i], val_prob[:, i])
        for i in range(len(label_cols)) if val_true[:, i].sum() > 0]))
    beat = f1_05 > C002B_F1_05 or macro_f1 > C002B_F1_05
    print(f"  F1(0.5)={f1_05:.4f} F1(100)={macro_f1:.4f} R={macro_rec:.4f} "
          f"P={macro_prec:.4f} AUC={macro_auc:.4f}")
    print(f"  vs C002b F1(0.5)={C002B_F1_05}: {'提升' if beat else '未超'}")

    y_test = c002b.predict_with_thresholds(model, test_loader, opt_t)
    sub = pd.concat([test_df[["protein_id"]],
                     pd.DataFrame(y_test, columns=label_cols)], axis=1)
    sub.to_csv(SUB_PATH, index=False)
    np.savez_compressed(PROBS_PATH, val=val_prob, y_true=val_true, thresholds=opt_t)
    torch.save({"state_dict": model.state_dict(), "thresholds": opt_t,
                "T": T, "lam": lam, "label_cols": label_cols}, MODEL_PATH)
    report = {
        "T": T,
        "lam": lam,
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
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  提交: {SUB_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
