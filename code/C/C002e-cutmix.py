"""
C002e-cutmix — C002b 结构 + 1D CutMix 重训
================================================================
依据:
  - CutMix (Yun et al., ICCV 2019, doi:10.1109/iccv.2019.00612)
    图像剪一块粘到另一张，标签按面积比混合。1D 版沿序列长度切一段。
  - Mixup (Zhang et al., ICLR 2018) 在 C002d 已负：全局凸组合把正例打稀。
    CutMix 保留局部 motif，更贴近蛋白结构域。
  - C002b：膨胀多尺度 CNN + ASL + 80ep + cosine。
方案:
  结构/切分/ASL/超参与 C002b 相同（seed=42）。训练时对 one-hot (B,C,L)
  在长度维切 [cut, cut+w)，换成另一条的对应段；标签按 w/L 混合。
  lam ~ Beta(α,α)，切段长度 = round((1-lam)*L)，默认 α=1.0（CutMix 原文）。
  验证/测试不做 CutMix。
  判断：独立 F1(0.5) 或 F1(100) 超 C002b 0.2714 → 可替换四路 CNN。
用法:
  python code/C/C002e-cutmix.py
  python code/C/C002e-cutmix.py --alpha=0.2
  python code/C/C002e-cutmix.py --eval-only
输出:
  models/cnn-c002e.pt
  rel/C002e_report.json
  rel/C002e_val_probs.npz
  submit/submission_C002e.csv
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "C"))
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
SUB_PATH = os.path.join(SUB_DIR, "submission_C002e.csv")
REPORT_PATH = os.path.join(REL_DIR, "C002e_report.json")
PROBS_PATH = os.path.join(REL_DIR, "C002e_val_probs.npz")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn-c002e.pt")

MAX_LEN = 1500
BATCH_SIZE = 256
N_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10
VAL_SIZE = 0.2
RANDOM_STATE = 42
CUTMIX_ALPHA = 1.0
C002B_F1_05 = 0.2714
DEVICE = c002b.DEVICE


def parse_args():
    alpha = CUTMIX_ALPHA
    eval_only = "--eval-only" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--alpha="):
            alpha = float(a.split("=", 1)[1])
    return alpha, eval_only


def cutmix_1d(x, y, alpha):
    """Yun et al. ICCV 2019 的 1D 版：x (B,C,L)，沿 L 切一段替换。"""
    if alpha <= 0:
        return x, y
    lam = float(np.random.beta(alpha, alpha))
    b, _, length = x.shape
    idx = torch.randperm(b, device=x.device)
    cut_len = int(round((1.0 - lam) * length))
    cut_len = max(1, min(cut_len, length))
    start = int(np.random.randint(0, length - cut_len + 1))
    x = x.clone()
    x[:, :, start:start + cut_len] = x[idx, :, start:start + cut_len]
    lam_area = 1.0 - cut_len / float(length)
    y = lam_area * y + (1.0 - lam_area) * y[idx]
    return x, y


def train_epoch(model, loader, optimizer, criterion, scaler, alpha):
    model.train()
    total_loss = 0.0
    n_samples = 0
    pbar = tqdm(loader, desc="Epoch", unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x, y = cutmix_1d(x, y, alpha)
        optimizer.zero_grad(set_to_none=True)
        with c002b.autocast_ctx():
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    pbar.close()
    return total_loss / max(n_samples, 1)


def main():
    alpha, eval_only = parse_args()
    t_start = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REL_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)
    print("=" * 60)
    print(f"C002e CutMix: alpha={alpha} epochs={N_EPOCHS} device={DEVICE} eval_only={eval_only}")

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

    model = c002b.MultiScaleCNNv3(n_labels=len(label_cols)).to(DEVICE)
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
            train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, alpha)
            val_prob, val_true = c002b.get_probs(model, val_loader)
            _, val_f1, _, _ = c002b.evaluate_at_thresholds(
                val_prob, val_true, np.full(len(label_cols), 0.5))
            scheduler.step()
            print(f"  Epoch {epoch:2d}/{N_EPOCHS} | loss={train_loss:.4f} | "
                  f"val_f1(0.5)={val_f1:.4f} | {time.time() - t_ep:.0f}s")
            if val_f1 > best_f1:
                best_f1, best_epoch, patience_counter = val_f1, epoch, 0
                torch.save({"state_dict": model.state_dict(), "alpha": alpha,
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
                "alpha": alpha, "label_cols": label_cols}, MODEL_PATH)
    report = {
        "alpha": alpha,
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
