"""
I004-calib-head — CK005a 分数残差校准头
================================================================
依据:
  - I001/I003：旧四路 3000 维→PCA 200 再逐标签重训，全部低于线性加权。
    未在 CK005a 上、未做过不降维的 500→500 头。
  - CK005h：0.52 上界来自 hit 逐标签过预测，不是整数 k（CK005j 已关）。
方案:
  复用 CK005a 四路加权分数 ens。半折训练，另一折出 OOF 分数再 SCut。
  残差: score' = clip(score + α · head(x), 0, 1)
  x ∈ {ens 500 维, ens+四路 2000 维}
  head ∈ {Linear 500→500, MLP  d→256→500}
  α ∈ {0.1, 0.3, 1.0}；损失 BCE（logits 上的残差用线性输出）
  主指标全局 100 档 Macro F1，门槛 +0.001。
用法:
  python code/I/I004-calib-head.py
  python code/I/I004-calib-head.py --cpu
输出:
  rel/I004_report.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
REPORT_PATH = os.path.join(REL_DIR, "I004_report.json")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
ALIGN_W = 0.4
SIM_W = 0.75
TOP_K = 3
MAX_ACCEPT = 5
RANDOM_STATE = 42
EPOCHS = 12
BATCH = 512
LR = 1e-3


def parse_aln(tsv, query_ids, id_map, evalue=0.001):
    idx = np.full((len(query_ids), MAX_ACCEPT), -1, np.int64)
    bits = np.zeros((len(query_ids), MAX_ACCEPT), np.float32)
    qpos = {q: i for i, q in enumerate(query_ids)}
    with open(tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            q, t = p[0], p[1]
            if q not in qpos or t not in id_map:
                continue
            i = qpos[q]
            try:
                bs, ev = float(p[2]), float(p[3])
            except ValueError:
                continue
            if ev > evalue:
                continue
            j = 0
            while j < MAX_ACCEPT and bits[i, j] > bs:
                j += 1
            if j < MAX_ACCEPT:
                idx[i, j:] = np.roll(idx[i, j:], 1)
                bits[i, j:] = np.roll(bits[i, j:], 1)
                idx[i, j] = id_map[t]
                bits[i, j] = bs
    return idx, bits


def agg_labels(idx, w, y_space, top_k=TOP_K):
    n = idx.shape[0]
    valid = idx[:, :top_k] >= 0
    lab = np.zeros((n, top_k, y_space.shape[1]), np.float32)
    for j in range(top_k):
        lab[:, j] = y_space[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :top_k], 1e-7)
    ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)


def build_kmer_val(y_train):
    sim = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        prefix = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{prefix}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_train[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        s = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        sim += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return sim


def search_t(ens, y):
    cands = np.linspace(0.01, 0.99, 100)
    best = np.full(N_LABELS, 0.5)
    for i in range(N_LABELS):
        yt = y[:, i]
        if yt.sum() == 0:
            continue
        pr = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0)
        fp = (pr & ~ytm).sum(0)
        fn = (~pr & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best[i] = cands[f1t.argmax()]
    return best


def macro_f1(ens, y, t):
    pred = ens >= t
    f1s = []
    for i in range(y.shape[1]):
        yt = y[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum())
        fp = int((p & ~yt).sum())
        fn = int((~p & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def macro_rp(ens, y, t):
    pred = ens >= t
    rs, ps = [], []
    for i in range(y.shape[1]):
        yt = y[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum())
        fp = int((p & ~yt).sum())
        fn = int((~p & yt).sum())
        rs.append(tp / (tp + fn) if tp + fn else 0.0)
        ps.append(tp / (tp + fp) if tp + fp else 0.0)
    return float(np.mean(rs)), float(np.mean(ps))


def macro_auc(y, p):
    aucs = []
    for i in range(y.shape[1]):
        yt = y[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, p[:, i]))
    return float(np.mean(aucs))


def eval_mask(ens, y, t, mask):
    return macro_f1(ens[mask], y[mask], t)


class CalibHead(nn.Module):
    def __init__(self, d_in, hidden=0):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, N_LABELS),
            )
        else:
            self.net = nn.Linear(d_in, N_LABELS)

    def forward(self, x):
        return self.net(x)


def train_head(x_tr, y_tr, x_va, device, hidden, epochs=EPOCHS):
    model = CalibHead(x_tr.shape[1], hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    pos = y_tr.mean(0)
    pw = np.clip((1 - pos) / np.maximum(pos, 1e-4), 1.0, 20.0)
    pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr.astype(np.float32))),
        batch_size=BATCH, shuffle=True)
    x_va_t = torch.from_numpy(x_va).to(device)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        delta = model(x_va_t).cpu().numpy()
    return delta


def oof_delta(x, y, hidden, device):
    n = len(y)
    rng = np.random.RandomState(RANDOM_STATE)
    perm = rng.permutation(n)
    mid = n // 2
    folds = [perm[:mid], perm[mid:]]
    out = np.zeros((n, N_LABELS), np.float32)
    for i, hold in enumerate(folds):
        fit = folds[1 - i]
        out[hold] = train_head(x[fit], y[fit], x[hold], device, hidden)
    return out


def pack(name, ens, y, t, hit, miss, extra=None):
    rec = {
        "name": name,
        "f1_100": round(macro_f1(ens, y, t), 6),
        "recall": round(macro_rp(ens, y, t)[0], 6),
        "precision": round(macro_rp(ens, y, t)[1], 6),
        "auc": round(macro_auc(y, ens), 6),
        "f1_miss": round(eval_mask(ens, y, t, miss), 6) if miss.any() else 0.0,
        "f1_hit": round(eval_mask(ens, y, t, hit), 6) if hit.any() else 0.0,
    }
    if extra:
        rec.update(extra)
    return rec


def main():
    use_cpu = "--cpu" in sys.argv
    t0 = time.time()
    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
    print("=" * 60)
    print(f"I004: CK005a 残差校准头  device={device}")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}

    tsv = os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    if not os.path.exists(tsv):
        raise FileNotFoundError("需要 CK005a very-sensitive val tsv")
    idx_v, bits_v = parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    print(f"  hit {int(hit.sum())}/{len(hit)} miss {int(miss.sum())}")

    sim_bit = agg_labels(idx_v, bits_v, y_train)
    sim_kmer = build_kmer_val(y_train)
    sim = ALIGN_W * sim_bit + (1 - ALIGN_W) * sim_kmer
    vp = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    cnn, t001a, t002 = vp["cnn"], t1["t001a_val"], t2["t002"]
    ens = w["t001a"] * t001a + w["t002"] * t002 + w["cnn"] * cnn + SIM_W * sim
    ens = ens.astype(np.float32)

    t_base = search_t(ens, y_val)
    rows = [pack("baseline", ens, y_val, t_base, hit, miss)]
    print(f"  基线 F1={rows[0]['f1_100']:.4f}")

    feats = {
        "ens": ens,
        "stack": np.concatenate([ens, cnn, t001a, t002], axis=1).astype(np.float32),
    }
    configs = [
        ("ens_lin", "ens", 0),
        ("ens_mlp", "ens", 256),
        ("stack_lin", "stack", 0),
        ("stack_mlp", "stack", 256),
    ]
    alphas = [0.1, 0.3, 1.0]

    for cname, fkey, hidden in configs:
        print(f"[train] {cname} d={feats[fkey].shape[1]} hidden={hidden or 'lin'}...")
        delta = oof_delta(feats[fkey], y_val, hidden, device)
        for a in alphas:
            cal = np.clip(ens + a * delta, 0.0, 1.0)
            t = search_t(cal, y_val)
            rec = pack(
                f"{cname}_a{a:g}", cal, y_val, t, hit, miss,
                {"alpha": a, "feat": fkey, "hidden": hidden,
                 "delta": round(macro_f1(cal, y_val, t) - rows[0]["f1_100"], 6)})
            rows.append(rec)
            print(f"  {rec['name']:18s} F1={rec['f1_100']:.4f} d={rec['delta']:+.4f} "
                  f"miss={rec['f1_miss']:.4f}")

    base = rows[0]
    best = max(rows[1:], key=lambda r: r["f1_100"])
    gain = best["f1_100"] - base["f1_100"]
    useful = bool(gain >= 0.001)
    print(f"  最优 {best['name']} F1={best['f1_100']:.4f} Δ={gain:+.4f} "
          f"→ {'有效' if useful else '无增益'}")

    report = {
        "baseline": base,
        "best": best,
        "gain_all_f1": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "seconds": round(time.time() - t0, 1),
        "device": str(device),
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
