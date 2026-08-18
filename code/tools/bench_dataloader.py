"""临时基准：DataLoader 配置对比"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
N_CHANNELS = 21
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = 20
MAX_LEN = 1500


class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, max_len):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        indices = [AA_TO_IDX.get(aa, PAD_IDX) for aa in seq[:self.max_len]]
        n = len(indices)
        if n < self.max_len:
            indices += [PAD_IDX] * (self.max_len - n)
        return (torch.tensor(indices, dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float32))


def collate_onehot(batch):
    seqs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    one_hot = F.one_hot(seqs, num_classes=N_CHANNELS).permute(0, 2, 1).float()
    return one_hot, labels


class MultiScaleCNN(nn.Module):
    def __init__(self, n_labels=500, filters=128):
        super().__init__()
        self.branch_a = nn.Sequential(
            nn.Conv1d(N_CHANNELS, filters, 3, padding=1), nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, 3, padding=1), nn.BatchNorm1d(filters), nn.ReLU(inplace=True))
        self.branch_b = nn.Sequential(
            nn.Conv1d(N_CHANNELS, filters, 5, padding=2), nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, 5, padding=2), nn.BatchNorm1d(filters), nn.ReLU(inplace=True))
        self.branch_c = nn.Sequential(
            nn.Conv1d(N_CHANNELS, filters, 7, padding=3), nn.BatchNorm1d(filters), nn.ReLU(inplace=True),
            nn.Conv1d(filters, filters, 7, padding=3), nn.BatchNorm1d(filters), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(filters * 3, n_labels)

    def forward(self, x):
        f1 = self.branch_a(x).max(dim=2)[0]
        f2 = self.branch_b(x).max(dim=2)[0]
        f3 = self.branch_c(x).max(dim=2)[0]
        return self.fc(self.dropout(torch.cat([f1, f2, f3], dim=1)))


def main():
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    train = pd.read_csv("data/train.csv", nrows=8192)
    y = train[[c for c in train.columns if c.startswith("label_")]].values.astype(np.int8)
    seqs = train["sequence"].values
    ds = ProteinDataset(seqs, y, MAX_LEN)

    def bench(bs, nw):
        loader = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_onehot,
                            pin_memory=True, num_workers=nw)
        model = MultiScaleCNN(n_labels=500, filters=128).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.BCEWithLogitsLoss()
        model.train()
        for x, yb in loader:
            x, yb = x.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), yb); loss.backward(); opt.step()
            break
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(1):
            for x, yb in loader:
                x, yb = x.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); loss = crit(model(x), yb); loss.backward(); opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1024**3
        print(f"batch={bs:3d} workers={nw} | {dt:.2f}s/epoch ({len(ds)/dt:.0f}/s) | VRAM {vram:.2f}GB")
        del model, opt, loader
        torch.cuda.empty_cache()

    bench(64, 0)
    bench(128, 0)
    bench(256, 0)
    bench(256, 2)
    bench(128, 2)


if __name__ == "__main__":
    main()
