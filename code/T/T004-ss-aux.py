"""
T004-ss-aux — T001a + 冻结 S4Pred 二级结构辅任务
================================================================
依据:
  - S4Pred: Moffat & Jones, Bioinformatics 2021,
    doi:10.1093/bioinformatics/btab491。单序列 SS，不要 MSA。
  - C018：同一套 SS 接到 C002b 池化前，独立 F1(0.5)=0.2217，
    OOF 较 C011a −0.0005，关闭。CNN 全局池冲掉残基信号。
自己的改法:
  骨干/训练配方同 T001a（RoPE 256/6，lr=3e-4，50ep，BCE+pos_weight）。
  复用 C018 的 S4Pred dump（截到 512）。final_norm 后加 Linear(256,3)
  残基头，L = BCE + λ·CE，λ=0.20，只在非 PAD。推理只用功能头。
  接到 C011a，换 T001a 路，α∈{0.50,1.00}。不重开 T003。
门槛:
  相对 C011a 半折 OOF ≥ +0.001。
用法:
  python code/T/T004-ss-aux.py
  python code/T/T004-ss-aux.py --fresh
  python code/T/T004-ss-aux.py --eval-only
输出:
  rel/T004_report.json
  rel/T004_probs.npz
  models/t004-ssaux.pt
  models/t004-ssaux.last.pt
"""

import importlib.util
import json
import math
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
from pathlib import Path

SRC = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

T001A_MOD = importlib.util.spec_from_file_location(
    "t001a", os.path.join(SRC, "T001a-rope.py"))
t001a = importlib.util.module_from_spec(T001A_MOD)
T001A_MOD.loader.exec_module(t001a)

BATCH_SIZE = t001a.BATCH_SIZE
VAL_BATCH = t001a.VAL_BATCH_SIZE
N_EPOCHS = t001a.N_EPOCHS
LR = t001a.LR
WEIGHT_DECAY = t001a.WEIGHT_DECAY
PATIENCE = t001a.PATIENCE
VALIDATE_EVERY = t001a.VALIDATE_EVERY
MAX_LEN = t001a.MAX_LEN
PAD_IDX = t001a.PAD_IDX
SPLIT_SEED = 42
A_GRID = (0.50, 1.00)
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
CNN_W = 0.2
C011A_ALPHA = 0.50
LAMBDA_SS = 0.20
SS_PAD = -100
T001A_F05 = 0.2426
SS_CACHE = os.path.join(REL_DIR, "c018_tmp", "ss_labels.npz")

MODEL_PATH = os.path.join(MODEL_DIR, "t004-ssaux.pt")
LAST_PATH = os.path.join(MODEL_DIR, "t004-ssaux.last.pt")
REPORT_PATH = os.path.join(REL_DIR, "T004_report.json")
PROBS_PATH = os.path.join(REL_DIR, "T004_probs.npz")

EVAL_ONLY = "--eval-only" in sys.argv
GEN_SUB = "--gen-sub" in sys.argv
FRESH = "--fresh" in sys.argv
DEVICE = t001a.DEVICE


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


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


def load_ss_labels(n_seqs):
    if not os.path.exists(SS_CACHE):
        raise FileNotFoundError(
            f"需要 C018 dump {SS_CACHE}。先跑完 C018 或 --dump-only")
    ss = np.load(SS_CACHE)["ss"]
    if ss.shape[0] != n_seqs:
        raise ValueError(f"SS dump n={ss.shape[0]} != train n={n_seqs}")
    print(f"  复用 {SS_CACHE}  截到 {MAX_LEN}  非PAD="
          f"{(ss[:, :MAX_LEN] != SS_PAD).mean():.3f}", flush=True)
    return ss[:, :MAX_LEN]


class SSAuxT001a(t001a.TransformerClassifier):
    def __init__(self, n_labels=500):
        super().__init__(n_labels=n_labels)
        self.ss_head = nn.Linear(t001a.D_MODEL, 3)

    def encode(self, x, mask):
        h = self.embed(x) * math.sqrt(self.d_model)
        key_padding_mask = ~mask.bool()
        for block in self.blocks:
            h = block(h, key_padding_mask)
        return self.final_norm(h)

    def forward(self, x, mask, return_ss=False):
        h = self.encode(x, mask)
        valid = mask.unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        logits = self.head(pooled)
        if return_ss:
            return logits, self.ss_head(h)
        return logits


def ss_ce(ss_logit, ss_y, mask):
    b, l, c = ss_logit.shape
    target = ss_y.clone()
    target = target.masked_fill(~mask.bool(), SS_PAD)
    return nn.functional.cross_entropy(
        ss_logit.reshape(-1, c),
        target.reshape(-1),
        ignore_index=SS_PAD)


def collate_ss(batch):
    x = torch.stack([item[0] for item in batch]).long()
    y = torch.stack([item[1] for item in batch])
    ss = torch.stack([item[2] for item in batch]).long()
    return x, y, ss


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="Epoch", unit="batch", leave=False)
    for x, y, ss in pbar:
        x, y, ss = x.to(DEVICE), y.to(DEVICE), ss.to(DEVICE)
        mask = (x != PAD_IDX).long()
        optimizer.zero_grad(set_to_none=True)
        with t001a.autocast_ctx():
            logits, ss_logit = model(x, mask, return_ss=True)
            loss = criterion(logits, y) + LAMBDA_SS * ss_ce(
                ss_logit.float(), ss, mask)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(n, 1)


@torch.no_grad()
def evaluate_probs(model, loader):
    model.eval()
    probs = []
    for batch in loader:
        x = batch[0].to(DEVICE)
        mask = (x != PAD_IDX).long()
        with t001a.autocast_ctx():
            logits = model(x, mask)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


def i019_from_t001a(t001a_p, cnn_lock, t002_p, w, sim, y_train, y_val):
    deep = (w["t001a"] * t001a_p + w["t002"] * t002_p + CNN_W * cnn_lock)
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


def save_last(model, opt, sched, scaler, epoch, best_f1, best_epoch, patience_counter):
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "epoch": int(epoch),
        "best_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "patience_counter": int(patience_counter),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, LAST_PATH)


def train_model(model, tr_loader, va_loader, y_val, y_train_labels, n_labels):
    pos = y_train_labels.sum(axis=0).astype(np.float32)
    neg = len(y_train_labels) - pos
    pos_weight = np.ones(n_labels, dtype=np.float32)
    mask = pos > 0
    pos_weight[mask] = np.clip(neg[mask] / pos[mask], 1.0, t001a.POS_WEIGHT_CLIP)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=t001a.LR_FACTOR, patience=t001a.LR_PATIENCE)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    best_f1, best_epoch, patience_counter = -1.0, 0, 0
    start_epoch = 1
    os.makedirs(MODEL_DIR, exist_ok=True)

    auto_resume = (not FRESH) and os.path.exists(LAST_PATH)
    if auto_resume:
        ckpt = torch.load(LAST_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        best_f1 = float(ckpt.get("best_f1", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        patience_counter = int(ckpt.get("patience_counter", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"  [resume] last={LAST_PATH} 下一轮={start_epoch} "
              f"best_ep={best_epoch} F1(0.5)={best_f1:.4f} "
              f"patience={patience_counter}", flush=True)
        if start_epoch > N_EPOCHS:
            print("  last 已训满，跳过训练", flush=True)
            return best_epoch, best_f1

    for ep in range(start_epoch, N_EPOCHS + 1):
        t_ep = time.time()
        loss = train_epoch(model, tr_loader, opt, criterion, scaler)
        if ep % VALIDATE_EVERY == 0 or ep == N_EPOCHS:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            ev = evaluate_probs(model, va_loader)
            f1 = f1_05(ev, y_val)
            sched.step(f1)
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
            save_last(model, opt, sched, scaler, ep, best_f1, best_epoch,
                      patience_counter)
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {ep}", flush=True)
                break
        else:
            print(f"  Epoch {ep:2d}/{N_EPOCHS} | loss={loss:.4f} | "
                  f"(跳过验证) | {time.time()-t_ep:.0f}s", flush=True)
            save_last(model, opt, sched, scaler, ep, best_f1, best_epoch,
                      patience_counter)
    print(f"  最佳 epoch: {best_epoch}, F1(0.5)={best_f1:.4f}", flush=True)
    return best_epoch, best_f1


def main():
    global i011, i019
    t0 = time.time()
    print("=" * 60)
    print(f"T004: T001a + S4Pred SS aux λ={LAMBDA_SS:g}")
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
    train_ids = train_df["protein_id"].values[idx_train]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_ids)}

    print("[ss] 复用 C018 S4Pred dump ...")
    ss_full = load_ss_labels(len(seqs))
    ss_tr = ss_full[idx_train]
    ss_va = ss_full[idx_val]

    X_tr = t001a.encode_sequences(seqs[idx_train], MAX_LEN)
    X_va = t001a.encode_sequences(seqs[idx_val], MAX_LEN)
    X_te = t001a.encode_sequences(test_df["sequence"].values, MAX_LEN)

    tr_loader = DataLoader(
        list(zip(torch.from_numpy(X_tr).long(),
                 torch.from_numpy(y_train.astype(np.float32)),
                 torch.from_numpy(ss_tr.astype(np.int64)))),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=collate_ss, pin_memory=True)
    va_loader = DataLoader(
        list(zip(torch.from_numpy(X_va).long(),
                 torch.from_numpy(y_val.astype(np.float32)),
                 torch.from_numpy(ss_va.astype(np.int64)))),
        batch_size=VAL_BATCH, shuffle=False, num_workers=0,
        collate_fn=collate_ss, pin_memory=True)
    te_loader = DataLoader(
        list(zip(torch.from_numpy(X_te).long(),
                 torch.zeros(len(X_te), n_labels),
                 torch.full((len(X_te), MAX_LEN), SS_PAD, dtype=torch.long))),
        batch_size=VAL_BATCH, shuffle=False, num_workers=0,
        collate_fn=collate_ss, pin_memory=True)

    model = SSAuxT001a(n_labels=n_labels).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    if EVAL_ONLY:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(MODEL_PATH)
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        best_epoch, best_f1 = ckpt.get("best_epoch"), ckpt.get("best_f1")
    else:
        print("[train] ...")
        best_epoch, best_f1 = train_model(
            model, tr_loader, va_loader, y_val, y_train, n_labels)
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)

    print("[infer] ...")
    t004_val = evaluate_probs(model, va_loader)
    t004_test = evaluate_probs(model, te_loader)
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t001a_val = t1["t001a_val"]
    cnn42 = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    cnn_lock = (1.0 - C011A_ALPHA) * cnn42 + C011A_ALPHA * c005_val
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    f1_new = f1_05(t004_val, y_val)
    f1_old = f1_05(t001a_val, y_val)
    corr = float(np.corrcoef(t001a_val.ravel(), t004_val.ravel())[0, 1])
    print(f"  T001a F1(0.5)={f1_old:.4f}  T004={f1_new:.4f}  corr={corr:.3f}")

    i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
    i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss_va = ~hit
    sim = i011.ALIGN_W * i011.agg_labels(idx_v, bits_v, y_train) + (
        1.0 - i011.ALIGN_W) * i011.build_kmer_val(y_train)
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]

    s_lock = i019_from_t001a(t001a_val, cnn_lock, t2, w, sim, y_train, y_val)
    rec0 = i011.eval_ens(s_lock, y_val, hit, miss_va, "c011a")
    print(f"  C011a F1={rec0['f1_100']:.4f} miss={rec0['f1_miss']:.4f}")

    rows = [rec0]
    configs = {"c011a": s_lock}
    mixed_t = {"c011a": t001a_val}
    for a in A_GRID:
        mix_t = (1.0 - a) * t001a_val + a * t004_val
        s = i019_from_t001a(mix_t, cnn_lock, t2, w, sim, y_train, y_val)
        name = f"t001a_a{a:.2f}"
        rec = i011.eval_ens(s, y_val, hit, miss_va, name, {
            "alpha": a, "f1_05": round(f1_05(mix_t, y_val), 6)})
        rec["delta_c011a"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        configs[name] = s
        mixed_t[name] = mix_t
        print(f"  {name:14s} F1(0.5)={rec['f1_05']:.4f} F1={rec['f1_100']:.4f} "
              f"d={rec['delta_c011a']:+.4f} miss={rec['f1_miss']:.4f}")

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
    oof_lock = None
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        pred = oof_pred(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": round(i011.macro_f1(pred, y_val), 6),
            "f1_miss": round(i011.macro_f1(pred[miss_va], y_val[miss_va]), 6),
        }
        oof_rows.append(rec)
        if name == "c011a":
            oof_lock = rec["f1_100"]
        print(f"  oof_{name:14s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_c011a"] = round(rec["f1_100"] - oof_lock, 6)

    cand = [r for r in oof_rows if r["name"] != "oof_c011a"]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof_lock
    useful = bool(gain >= 0.001)
    print(f"  独立 {'超' if f1_new > T001A_F05 else '未超'} T001a {T001A_F05}")
    print(f"  OOF 最优 {best_oof['name']} vs C011a {gain:+.4f} → "
          f"{'有效' if useful else '无增益'}")

    os.makedirs(REL_DIR, exist_ok=True)
    np.savez_compressed(PROBS_PATH, val=t004_val, test=t004_test, y_true=y_val)
    report = {
        "n_params": n_params,
        "lambda_ss": LAMBDA_SS,
        "best_epoch": best_epoch,
        "f1_05_t001a": round(f1_old, 6),
        "f1_05_t004": round(f1_new, 6),
        "corr_t001a_t004": round(corr, 6),
        "beat_t001a": bool(f1_new > T001A_F05),
        "baseline_c011a": rec0,
        "oof_c011a": oof_lock,
        "best_oof": best_oof,
        "gain_oof_vs_c011a": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    if useful:
        report["conclusion"] = "C011a OOF≥+0.001，可换 T001a 路"
    elif f1_new <= f1_old:
        report["conclusion"] = "独立未超 T001a，Transformer SS 辅任务关闭"
    else:
        report["conclusion"] = "独立涨、融合未过 +0.001"

    if GEN_SUB and useful:
        cnn_te = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
        c005_te = np.load(os.path.join(REL_DIR, "C005_test_probs.npz"))["test"]
        cnn_lock_te = (1.0 - C011A_ALPHA) * cnn_te + C011A_ALPHA * c005_te
        t002_te = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
        t001a_te = t1["t001a_test"]
        best_name = best_oof["name"].replace("oof_", "")
        alpha = 0.0 if best_name == "c011a" else float(best_name.split("a")[1])
        mix_te = (1.0 - alpha) * t001a_te + alpha * t004_test
        deep_te = w["t001a"] * mix_te + w["t002"] * t002_te + w["cnn"] * cnn_lock_te
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
        mix_v = mixed_t[best_name]
        deep_v = w["t001a"] * mix_v + w["t002"] * t2 + CNN_W * cnn_lock
        s011_v = deep_v + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
            i011.rrf_pair(deep_v, sim, K_RRF))
        t_sub = i011.search_t(configs[best_name], y_val)
        t_i011 = i011.search_t(s011_v, y_val)
        n_te = deep_te.shape[0]
        s011_te = deep_te + i011.SIM_W * sim_te + BETA_RRF * i011.col_norm(
            i011.rrf_pair(deep_te, sim_te, K_RRF))
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
        sub_path = os.path.join(SUB_DIR, "submission_T004_ssaux.csv")
        sub.to_csv(sub_path, index=False)
        report["submission"] = sub_path
        report["submit_alpha"] = alpha
        print(f"  提交: {sub_path}")

    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print(f"  概率: {PROBS_PATH}")
    print(f"  用时 {report['seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
