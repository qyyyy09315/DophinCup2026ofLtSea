"""
C006-aa-embed — C002b 骨干 + 理化初始化的可学习氨基酸嵌入
================================================================
依据:
  - Yoon Kim, EMNLP 2014, doi:10.3115/v1/d14-1181
    文本 CNN 用可学习词嵌入，one-hot 是特例。本赛 C002b 一直 one-hot。
  - Atchley et al., PNAS 2005, doi:10.1073/pnas.0408677102
    20 氨基酸 → 5 维理化因子（极性/二级结构/体积/密码子多样性/电荷）。
  - Zacharaki, PeerJ CS 2017, doi:10.7717/peerj-cs.124
    BLOSUM 通道是固定编码；这里固定编码只作初始化，训练中可学。
自己的改法:
  不是纯随机嵌入，也不是冻死 BLOSUM。Embedding(21,16) 前 5 维用 Atchley
  标准化后填入，其余 Xavier；padding 行全 0。骨干/ASL/80ep/cosine 同 C002b。
  判据：独立 F1(0.5) 超 C002b 0.2714。
用法:
  python code/C/C006-aa-embed.py
  python code/C/C006-aa-embed.py --emb-dim=20
  python code/C/C006-aa-embed.py --eval-only
输出:
  models/cnn-c006.pt
  rel/C006_report.json
  rel/C006_val_probs.npz
  submit/submission_C006.csv
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
import importlib.util
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
SUB_PATH = os.path.join(SUB_DIR, "submission_C006.csv")
REPORT_PATH = os.path.join(REL_DIR, "C006_report.json")
PROBS_PATH = os.path.join(REL_DIR, "C006_val_probs.npz")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn-c006.pt")

MAX_LEN = 1500
BATCH_SIZE = 256
N_EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10
VAL_SIZE = 0.2
RANDOM_STATE = 42
EMB_DIM = 16
C002B_F1_05 = 0.2714
DEVICE = c002b.DEVICE

# Atchley et al. PNAS 2005 five factors, 20 standard amino acids
ATCHLEY = {
    "A": (-0.591, -1.302, -0.733, 1.570, -0.146),
    "C": (-1.343, 0.465, -0.862, -1.020, -0.255),
    "D": (1.050, 0.302, -3.656, -0.259, -3.242),
    "E": (1.357, -1.453, 1.477, 0.113, -0.837),
    "F": (-1.006, -0.590, 1.891, -0.397, 0.412),
    "G": (-0.384, 1.652, 1.330, 1.045, 2.064),
    "H": (0.336, -0.417, -1.673, -1.474, -0.078),
    "I": (-1.239, -0.547, 2.131, 0.393, 0.816),
    "K": (1.831, -0.561, 0.533, -0.277, 1.648),
    "L": (-1.019, -0.987, -1.505, 1.266, -0.912),
    "M": (-0.663, -1.524, 2.219, -1.005, 1.212),
    "N": (0.945, 0.828, 1.299, -0.169, 0.933),
    "P": (0.189, 2.081, -1.628, 0.421, -1.392),
    "Q": (0.931, -0.179, -3.005, -0.503, -1.853),
    "R": (1.538, -0.055, 1.502, 0.440, 2.897),
    "S": (-0.228, 1.399, -4.760, 0.670, -2.647),
    "T": (-0.032, 0.326, 2.213, 0.908, 1.313),
    "V": (-1.337, -0.279, -0.544, -0.212, -0.121),
    "W": (-0.595, 0.009, 0.672, -2.128, -0.184),
    "Y": (0.260, 0.830, 3.097, -0.838, 1.512),
}


def collate_idx(batch):
    seqs = torch.stack([item[0] for item in batch]).long()
    labels = torch.stack([item[1] for item in batch])
    return seqs, labels


def atchley_init(emb: nn.Embedding, dim: int):
    w = torch.empty(c002b.N_CHANNELS, dim)
    nn.init.xavier_uniform_(w)
    mat = np.array([ATCHLEY[aa] for aa in c002b.AMINO_ACIDS], dtype=np.float32)
    mat = (mat - mat.mean(0)) / (mat.std(0) + 1e-8)
    n = min(5, dim)
    for aa, idx in c002b.AA_TO_IDX.items():
        w[idx, :n] = torch.from_numpy(mat[idx, :n])
    w[c002b.PAD_IDX].zero_()
    with torch.no_grad():
        emb.weight.copy_(w)
        emb.weight[c002b.PAD_IDX].zero_()


class MultiScaleCNNEmbed(nn.Module):
    """C002b 四路膨胀 CNN，输入从 one-hot 换成可学习嵌入。"""

    def __init__(self, n_labels=500, emb_dim=EMB_DIM, filters=c002b.FILTERS,
                 kernels=c002b.KERNELS, dilations=c002b.DILATIONS,
                 dropout=c002b.DROPOUT):
        super().__init__()
        self.embed = nn.Embedding(c002b.N_CHANNELS, emb_dim,
                                  padding_idx=c002b.PAD_IDX)
        atchley_init(self.embed, emb_dim)
        self.branches = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        self.ses = nn.ModuleList()
        for k, d in zip(kernels, dilations):
            pad = d * (k - 1) // 2
            branch = nn.Sequential(
                nn.Conv1d(emb_dim, filters, k, padding=pad, dilation=d),
                nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
                nn.Conv1d(filters, filters, k, padding=pad, dilation=d),
                nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
            )
            self.branches.append(branch)
            self.shortcuts.append(nn.Conv1d(emb_dim, filters, 1))
            self.ses.append(c002b.SEBlock(filters))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(kernels) * 2, n_labels)

    def forward(self, idx):
        x = self.embed(idx).permute(0, 2, 1)
        feats = []
        for branch, shortcut, se in zip(self.branches, self.shortcuts, self.ses):
            h = branch(x) + shortcut(x)
            h = se(h)
            feats.append(h.max(dim=2)[0])
            feats.append(h.mean(dim=2))
        return self.fc(self.dropout(torch.cat(feats, dim=1)))


def parse_args():
    dim, eval_only = EMB_DIM, "--eval-only" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--emb-dim="):
            dim = int(a.split("=", 1)[1])
    return dim, eval_only


def main():
    emb_dim, eval_only = parse_args()
    t_start = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REL_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)
    print("=" * 60)
    print(f"C006 AA-embed: dim={emb_dim} epochs={N_EPOCHS} "
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
                              collate_fn=collate_idx, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_idx, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_idx, pin_memory=True, num_workers=0)
    print(f"  训练 {len(train_ds)} 验证 {len(val_ds)}")

    model = MultiScaleCNNEmbed(n_labels=len(label_cols), emb_dim=emb_dim).to(DEVICE)
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
                torch.save({"state_dict": model.state_dict(), "emb_dim": emb_dim,
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
                "emb_dim": emb_dim, "label_cols": label_cols}, MODEL_PATH)
    report = {
        "emb_dim": emb_dim,
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
