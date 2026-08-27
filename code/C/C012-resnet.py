"""
C012-resnet — 深层残差膨胀 CNN（换一元 C002b）
================================================================
依据:
  - ProteInfer: Sanderson, Bileschi, Belanger & Colwell, eLife 2023,
    doi:10.7554/eLife.80942。残差膨胀 CNN 与 BLAST 学到不同信息。
    原版 5 层 × 1100 滤，8GB 放不下。
  - ResNet: He et al., CVPR 2016, doi:10.1109/cvpr.2016.90。
    恒等残差让深度卷积可训。
  - C002b 是 4 分支浅膨胀（每路 2 层）+ 全局池，不是深层残差栈。
    C002a 长卷积+分桶已负。不重开 Mixup/CSRA/坐标/spanmask。
自己的改法:
  stem 21→96，5 个恒等残差块，kernel=9，dilation=(1,2,4,8,1)，
  每块 SE，max/mean 池化，ASL + 80 ep cosine，max_len=1500。
  独立对照 C002b F1(0.5)=0.2714；接到 I019，α∈{0.50,1.00} 换 CNN 路。
门槛:
  独立 F1(0.5)>0.2714 且 I019 半折 OOF ≥ +0.001。
用法:
  python code/C/C012-resnet.py
  python code/C/C012-resnet.py --eval-only
  python code/C/C012-resnet.py --gen-sub
输出:
  rel/C012_report.json
  rel/C012_probs.npz
  models/c012-resnet.pt
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

C002B_MOD = importlib.util.spec_from_file_location(
    "c002b", os.path.join(SRC, "C002b-cnn.py"))
c002b = importlib.util.module_from_spec(C002B_MOD)
C002B_MOD.loader.exec_module(c002b)

CHANNELS = 96
N_BLOCKS = 5
KERNEL = 9
DILATIONS = (1, 2, 4, 8, 1)
BATCH_SIZE = 128
N_EPOCHS = c002b.N_EPOCHS
LR = c002b.LR
WEIGHT_DECAY = c002b.WEIGHT_DECAY
PATIENCE = c002b.PATIENCE
DROPOUT = c002b.DROPOUT
SPLIT_SEED = 42
A_GRID = (0.50, 1.00)
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
CNN_W = 0.2
C002B_F05 = 0.2714

MODEL_PATH = os.path.join(MODEL_DIR, "c012-resnet.pt")
REPORT_PATH = os.path.join(REL_DIR, "C012_report.json")
PROBS_PATH = os.path.join(REL_DIR, "C012_probs.npz")
TOK_CACHE = os.path.join(REL_DIR, "c012_tokens.npz")

EVAL_ONLY = "--eval-only" in sys.argv
GEN_SUB = "--gen-sub" in sys.argv
DEVICE = c002b.DEVICE


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class DilatedResBlock(nn.Module):
    def __init__(self, channels, kernel, dilation):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel,
                               padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel,
                               padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = c002b.SEBlock(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return self.act(x + h)


class ResidualDilatedCNN(nn.Module):
    def __init__(self, n_labels=500, channels=CHANNELS, n_blocks=N_BLOCKS,
                 kernel=KERNEL, dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(c002b.N_CHANNELS, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[DilatedResBlock(channels, kernel, d) for d in dilations[:n_blocks]])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * 2, n_labels)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        feats = torch.cat([h.max(dim=2)[0], h.mean(dim=2)], dim=1)
        return self.fc(self.dropout(feats))


def f1_05(p, y):
    pred = (p >= 0.5)
    f1s = []
    for i in range(y.shape[1]):
        yt = y[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        pr = pred[:, i]
        tp = int((pr & yt).sum())
        fp = int((pr & ~yt).sum())
        fn = int((~pr & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


@torch.no_grad()
def evaluate_probs(model, loader):
    model.eval()
    probs = []
    for x, _ in loader:
        x = x.to(DEVICE)
        with c002b.autocast_ctx():
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


def i019_from_cnn(cnn, deep_rest, sim, y_train, y_val):
    deep = deep_rest + CNN_W * cnn
    s011 = deep + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
        i011.rrf_pair(deep, sim, K_RRF))
    n = s011.shape[0]
    pi = y_train.mean(0).astype(np.float64)
    t = i011.search_t(s011, y_val)
    row_m = np.maximum((s011 >= t).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(pi * n * RHO, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / TAU_SK)
    p = i019.sinkhorn(ker, row_m, col_m)
    return i019.mix(s011, p, ALPHA_SK)


def load_tokens(seqs, test_seqs, idx_train, idx_val):
    if os.path.exists(TOK_CACHE):
        d = np.load(TOK_CACHE)
        return d["X_tr"], d["X_va"], d["X_te"]
    clean = lambda s: s.replace("U", "X").replace("Z", "X").replace("O", "X")
    X_tr = c002b.encode_sequences([clean(s) for s in seqs[idx_train]], c002b.MAX_LEN)
    X_va = c002b.encode_sequences([clean(s) for s in seqs[idx_val]], c002b.MAX_LEN)
    X_te = c002b.encode_sequences([clean(s) for s in test_seqs], c002b.MAX_LEN)
    os.makedirs(REL_DIR, exist_ok=True)
    np.savez_compressed(TOK_CACHE, X_tr=X_tr, X_va=X_va, X_te=X_te)
    return X_tr, X_va, X_te


def train_model(model, tr_loader, va_loader, y_val):
    criterion = c002b.AsymmetricLoss().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS, eta_min=LR / 100)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    best_f1, best_epoch, patience_counter = -1.0, 0, 0
    os.makedirs(MODEL_DIR, exist_ok=True)
    for ep in range(1, N_EPOCHS + 1):
        t_ep = time.time()
        loss = c002b.train_epoch(model, tr_loader, opt, criterion, scaler)
        ev = evaluate_probs(model, va_loader)
        f1 = f1_05(ev, y_val)
        sched.step()
        print(f"  Epoch {ep:2d}/{N_EPOCHS} | loss={loss:.4f} | "
              f"F1(0.5)={f1:.4f} | {time.time()-t_ep:.0f}s", flush=True)
        if f1 > best_f1:
            best_f1, best_epoch, patience_counter = f1, ep, 0
            torch.save({
                "state_dict": model.state_dict(),
                "best_f1": float(best_f1),
                "best_epoch": int(best_epoch),
            }, MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {ep}", flush=True)
                break
    print(f"  最佳 epoch: {best_epoch}, F1(0.5)={best_f1:.4f}", flush=True)
    return best_epoch, best_f1


def main():
    global i011, i019
    t0 = time.time()
    print("=" * 60)
    print(f"C012: 残差膨胀 CNN | ch={CHANNELS} blocks={N_BLOCKS} k={KERNEL}")
    print(f"  Device: {DEVICE}")

    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train.csv"))
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    n_labels = len(label_cols)
    y_full = train_df[label_cols].values.astype(np.int8)
    seqs = train_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=SPLIT_SEED)
    y_train, y_val = y_full[idx_train], y_full[idx_val]

    X_tr, X_va, X_te = load_tokens(seqs, test_df["sequence"].values, idx_train, idx_val)
    tr_loader = DataLoader(
        list(zip(torch.from_numpy(X_tr).long(), torch.from_numpy(y_train.astype(np.float32)))),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=c002b.collate_onehot)
    va_loader = DataLoader(
        list(zip(torch.from_numpy(X_va).long(), torch.from_numpy(y_val.astype(np.float32)))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=c002b.collate_onehot)
    te_loader = DataLoader(
        list(zip(torch.from_numpy(X_te).long(), torch.zeros(len(X_te), n_labels))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=c002b.collate_onehot)

    model = ResidualDilatedCNN(n_labels=n_labels).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}  (C002b ~1.00M)")

    if EVAL_ONLY:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(MODEL_PATH)
        c002b.load_checkpoint(model, MODEL_PATH)
        best_epoch, best_f1 = None, None
    else:
        print("[train] ...")
        best_epoch, best_f1 = train_model(model, tr_loader, va_loader, y_val)
        c002b.load_checkpoint(model, MODEL_PATH)

    print("[infer] ...")
    c012_val = evaluate_probs(model, va_loader)
    c012_test = evaluate_probs(model, te_loader)
    cnn42 = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    f1_new = f1_05(c012_val, y_val)
    f1_old = f1_05(cnn42, y_val)
    corr = float(np.corrcoef(cnn42.ravel(), c012_val.ravel())[0, 1])
    print(f"  C002b F1(0.5)={f1_old:.4f}  C012 F1(0.5)={f1_new:.4f}  corr={corr:.3f}")

    i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
    i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim = i011.ALIGN_W * i011.agg_labels(idx_v, bits_v, y_train) + (
        1.0 - i011.ALIGN_W) * i011.build_kmer_val(y_train)
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep_rest = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"]

    s019 = i019_from_cnn(cnn42, deep_rest, sim, y_train, y_val)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    print(f"  I019 F1={rec0['f1_100']:.4f} miss={rec0['f1_miss']:.4f}")

    rows = [rec0]
    configs = {"i019": s019}
    mixed_cnn = {"i019": cnn42}
    for a in A_GRID:
        mix_cnn = (1.0 - a) * cnn42 + a * c012_val
        s = i019_from_cnn(mix_cnn, deep_rest, sim, y_train, y_val)
        name = f"cnn_a{a:.2f}"
        rec = i011.eval_ens(s, y_val, hit, miss, name, {
            "alpha": a, "f1_05": round(f1_05(mix_cnn, y_val), 6)})
        rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        configs[name] = s
        mixed_cnn[name] = mix_cnn
        print(f"  {name:12s} F1(0.5)={rec['f1_05']:.4f} F1={rec['f1_100']:.4f} "
              f"d={rec['delta_i019']:+.4f} miss={rec['f1_miss']:.4f}")

    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(y_val))
    mid = len(y_val) // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_pred(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            th = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= th).astype(np.int8)
        return pred

    oof_rows = []
    oof_i019 = None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "i019":
            oof_i019 = rec["f1_100"]
        print(f"  oof_{name:12s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_i019"] = round(rec["f1_100"] - oof_i019, 6)

    cand = [r for r in oof_rows if r["name"] != "oof_i019"]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof_i019
    beat_ind = bool(f1_new > C002B_F05)
    useful = bool(beat_ind and gain >= 0.001)
    print(f"  独立 {'超' if beat_ind else '未超'} C002b 0.2714")
    print(f"  OOF 最优 {best_oof['name']} vs I019 {gain:+.4f} → "
          f"{'有效' if useful else '无增益'}")

    os.makedirs(REL_DIR, exist_ok=True)
    np.savez_compressed(PROBS_PATH, val=c012_val, test=c012_test, y_true=y_val)

    report = {
        "n_params": n_params,
        "channels": CHANNELS,
        "n_blocks": N_BLOCKS,
        "kernel": KERNEL,
        "dilations": list(DILATIONS),
        "best_epoch": best_epoch,
        "f1_05_c002b": round(f1_old, 6),
        "f1_05_c012": round(f1_new, 6),
        "corr_c002b_c012": round(corr, 6),
        "beat_c002b": beat_ind,
        "baseline_i019": rec0,
        "oof_i019": oof_i019,
        "best_oof": best_oof,
        "gain_oof_vs_i019": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    if useful:
        report["conclusion"] = "独立超 C002b 且 I019 OOF≥+0.001，可换 CNN 路"
    elif not beat_ind:
        report["conclusion"] = "独立未超 C002b 0.2714，深层残差关闭"
    else:
        report["conclusion"] = "独立涨、融合未过 +0.001"

    if GEN_SUB and useful:
        cnn_te = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
        t002_te = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
        best_name = best_oof["name"].replace("oof_", "")
        alpha = 1.0 if best_name == "i019" else float(best_name.split("a")[1])
        mix_te = (1.0 - alpha) * cnn_te + alpha * c012_test
        deep_te = w["t001a"] * np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"] \
            + w["t002"] * t002_te + w["cnn"] * mix_te
        tsv_te = os.path.join(i011.CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
        full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
        idx_t, bits_t = i011.parse_aln(tsv_te, test_df["protein_id"].values, full_map)
        sim_bit_te = i011.agg_labels(idx_t, bits_t, y_full)
        sim_kmer_te = 0.0
        for kk, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
            if wt <= 0:
                continue
            d = np.load(os.path.join(
                REL_DIR, "TCK002a_neigh_full.npz" if kk == 6
                else f"CK004_neigh_k{kk}_full.npz"))
            n2 = d["idx_test"].shape[0]
            lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
            s = d["sim_test"][:, :5].astype(np.float32)
            ww = np.maximum(s, 1e-7) ** 2
            sim_kmer_te = sim_kmer_te + wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
        sim_te = i011.ALIGN_W * sim_bit_te + (1.0 - i011.ALIGN_W) * sim_kmer_te
        s011_te = deep_te + i011.SIM_W * sim_te + BETA_RRF * i011.col_norm(
            i011.rrf_pair(deep_te, sim_te, K_RRF))
        mix_v = mixed_cnn[best_name]
        deep_v = deep_rest + CNN_W * mix_v
        s011_v = deep_v + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
            i011.rrf_pair(deep_v, sim, K_RRF))
        t_sub = i011.search_t(configs[best_name], y_val)
        t_i011 = i011.search_t(s011_v, y_val)
        n_te = s011_te.shape[0]
        pi = y_train.mean(0).astype(np.float64)
        row_m = np.maximum((s011_te >= t_i011).sum(1).astype(np.float64), 1.0)
        col_m = np.maximum(pi * n_te * RHO, 1e-6)
        col_m *= row_m.sum() / col_m.sum()
        ker = np.exp((s011_te - s011_te.max(axis=1, keepdims=True)) / TAU_SK)
        p_te = i019.sinkhorn(ker, row_m, col_m)
        s_te = i019.mix(s011_te, p_te, ALPHA_SK)
        y_pred = (s_te >= t_sub).astype(np.int8)
        os.makedirs(SUB_DIR, exist_ok=True)
        sub = pd.concat([test_df[["protein_id"]],
                         pd.DataFrame(y_pred, columns=label_cols)], axis=1)
        sub_path = os.path.join(SUB_DIR, "submission_C012_resnet.csv")
        sub.to_csv(sub_path, index=False)
        report["submission"] = sub_path
        report["submit_alpha"] = alpha
        print(f"  提交: {sub_path}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  概率: {PROBS_PATH}")
    print(f"  用时 {report['seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
