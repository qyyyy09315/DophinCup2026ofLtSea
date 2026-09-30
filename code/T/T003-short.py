"""
T003-short — 短序列专用 RoPE Transformer（换一元 T001a）
================================================================
依据:
  - PeptideBERT: Guntuboina et al., J. Phys. Chem. Lett. 2023,
    doi:10.1021/acs.jpclett.3c02398。短肽单独训 Transformer，
    不和全长蛋白抢梯度。
  - CK005a miss 分层：≤128aa miss 率 23.9%；>1500aa 仅 2.2%。
    缺口是短远程同源，不是 1500 截断。
  - I005 短序列压 sim 全面变差：短样本更依赖同源，不能改融合权。
    本实验不改 sim / RRF / Sinkhorn，只换 deep 里的 T001a。
  - T001a 全长 max_len=512、loss 按蛋白计，短序列被长蛋白梯度淹没。
自己的改法:
  切分仍 seed=42。只在 idx_train 且 L≤256 上训，结构同 T001a
  （d=256 / 6 层 / RoPE / BCE+pos_weight），max_len=256。
  推理：短样本用 T003 替换 T001a，长样本保持 T001a，重建 I019。
  另扫短行混合 α∈{0.50,1.00}。
门槛:
  短 val 独立 F1(0.5) 对照同子集 T001a；融合相对 I019 半折 ≥ +0.001。
用法:
  python code/T/T003-short.py
  python code/T/T003-short.py --resume
  python code/T/T003-short.py --eval-only
  python code/T/T003-short.py --gen-sub
输出:
  rel/T003_report.json
  rel/T003_short_probs.npz
  models/t003-short.pt
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
from pathlib import Path

SRC = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

LEN_CUT = 256
MAX_LEN = 256
BATCH_SIZE = 256
VAL_BATCH_SIZE = 512

T001A_MOD = importlib.util.spec_from_file_location(
    "t001a", os.path.join(SRC, "T001a-rope.py"))
t001a = importlib.util.module_from_spec(T001A_MOD)
T001A_MOD.loader.exec_module(t001a)
t001a.MAX_LEN = MAX_LEN  # TAttention.RoPE 读模块级 MAX_LEN

N_EPOCHS = t001a.N_EPOCHS
LR = t001a.LR
WEIGHT_DECAY = t001a.WEIGHT_DECAY
PATIENCE = t001a.PATIENCE
LR_PATIENCE = t001a.LR_PATIENCE
LR_FACTOR = t001a.LR_FACTOR
VALIDATE_EVERY = t001a.VALIDATE_EVERY
POS_WEIGHT_CLIP = t001a.POS_WEIGHT_CLIP
SPLIT_SEED = 42
A_GRID = (0.50, 1.00)
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60

MODEL_PATH = os.path.join(MODEL_DIR, "t003-short.pt")
REPORT_PATH = os.path.join(REL_DIR, "T003_report.json")
PROBS_PATH = os.path.join(REL_DIR, "T003_short_probs.npz")

EVAL_ONLY = "--eval-only" in sys.argv
RESUME = "--resume" in sys.argv
GEN_SUB = "--gen-sub" in sys.argv
DEVICE = t001a.DEVICE


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(PROJECT_ROOT, "code", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def seq_lens(arr):
    return np.array([len(s) for s in arr], dtype=np.int32)


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


def f1_100(p, y):
    t = i011.search_t(p, y)
    return i011.macro_f1((p >= t).astype(np.int8), y)


def f1_on_mask(score, y_all, mask, y_mask):
    t = i011.search_t(score, y_all)
    return round(i011.macro_f1((score[mask] >= t).astype(np.int8), y_mask), 6)


@torch.no_grad()
def infer_rows(model, sequences, row_idx):
    if len(row_idx) == 0:
        return np.zeros((0, 500), dtype=np.float32)
    ds = t001a.ProteinDataset(
        sequences[row_idx],
        np.zeros((len(row_idx), 500), dtype=np.int8),
        MAX_LEN)
    loader = DataLoader(ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
                        pin_memory=DEVICE.type == "cuda", num_workers=0)
    model.eval()
    chunks = []
    for x, _ in loader:
        x = x.to(DEVICE)
        mask = (x != t001a.PAD_IDX).long()
        with t001a.autocast_ctx():
            logits = model(x, mask)
        chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.vstack(chunks).astype(np.float32)


def i019_from_t001a(t001a_val, cnn, t002, sim, y_train, y_val, w):
    deep = w["t001a"] * t001a_val + w["t002"] * t002 + w["cnn"] * cnn
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


def blend_short(base, specialist, short_mask, alpha):
    out = base.copy()
    out[short_mask] = (1.0 - alpha) * base[short_mask] + alpha * specialist[short_mask]
    return out


def train_short(model, train_loader, val_loader, y_val_short, pos_weight):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    if RESUME:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(MODEL_PATH)
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        best_f1 = float(ckpt.get("best_f1", 0.0)) if isinstance(ckpt, dict) else 0.0
        best_epoch = int(ckpt.get("best_epoch", 0)) if isinstance(ckpt, dict) else 0
        start_epoch = best_epoch + 1
        print(f"  [resume] best_epoch={best_epoch} F1(0.5)={best_f1:.4f}")
    else:
        best_f1, best_epoch, start_epoch = -1.0, 0, 1

    patience_counter = 0
    os.makedirs(MODEL_DIR, exist_ok=True)
    for epoch in range(start_epoch, N_EPOCHS + 1):
        t_ep = time.time()
        train_loss = t001a.train_epoch(model, train_loader, optimizer, criterion, scaler)
        if epoch % VALIDATE_EVERY == 0:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            val_prob, _ = t001a.get_probs(model, val_loader)
            val_f1 = f1_05(val_prob, y_val_short)
            scheduler.step(val_f1)
            print(f"  Epoch {epoch:2d}/{N_EPOCHS} | loss={train_loss:.4f} | "
                  f"short F1(0.5)={val_f1:.4f} | {time.time()-t_ep:.0f}s", flush=True)
            if val_f1 > best_f1:
                best_f1, best_epoch, patience_counter = val_f1, epoch, 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "best_f1": float(best_f1),
                    "best_epoch": int(best_epoch),
                    "epoch": int(epoch),
                    "len_cut": LEN_CUT,
                    "max_len": MAX_LEN,
                }, MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}", flush=True)
                    break
        else:
            print(f"  Epoch {epoch:2d}/{N_EPOCHS} | loss={train_loss:.4f} | "
                  f"(跳过验证) | {time.time()-t_ep:.0f}s", flush=True)
    print(f"  最佳 epoch: {best_epoch}, short F1(0.5)={best_f1:.4f}", flush=True)
    return best_epoch, best_f1


def main():
    global i011, i019
    t0 = time.time()
    print("=" * 60)
    print(f"T003: 短序列 RoPE Transformer | cut={LEN_CUT} max_len={MAX_LEN}")
    print(f"  Device: {DEVICE}")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    n_labels = len(label_cols)
    y_full = train_df[label_cols].values.astype(np.int8)
    seqs = train_df["sequence"].values
    test_seqs = test_df["sequence"].values
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=SPLIT_SEED)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    lens_tr = seq_lens(seqs[idx_train])
    lens_va = seq_lens(seqs[idx_val])
    lens_te = seq_lens(test_seqs)
    short_tr = lens_tr <= LEN_CUT
    short_va = lens_va <= LEN_CUT
    short_te = lens_te <= LEN_CUT
    print(f"  train short {int(short_tr.sum())}/{len(idx_train)} "
          f"val short {int(short_va.sum())}/{len(idx_val)} "
          f"test short {int(short_te.sum())}/{len(test_seqs)}")

    i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
    i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))

    train_short_idx = idx_train[short_tr]
    val_short_idx = idx_val[short_va]
    y_tr_short = y_full[train_short_idx]
    y_va_short = y_full[val_short_idx]
    train_ds = t001a.ProteinDataset(seqs[train_short_idx], y_tr_short, MAX_LEN)
    val_ds = t001a.ProteinDataset(seqs[val_short_idx], y_va_short, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=DEVICE.type == "cuda", num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
                            pin_memory=DEVICE.type == "cuda", num_workers=0)

    t001a.MAX_LEN = MAX_LEN
    model = t001a.TransformerClassifier(n_labels=n_labels, max_len=MAX_LEN).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    pos = y_tr_short.sum(axis=0).astype(np.float32)
    neg = len(y_tr_short) - pos
    pos_weight = np.ones(n_labels, dtype=np.float32)
    mask = pos > 0
    pos_weight[mask] = np.clip(neg[mask] / pos[mask], 1.0, POS_WEIGHT_CLIP)
    pw_tensor = torch.tensor(pos_weight, device=DEVICE)
    n_empty = int((pos == 0).sum())
    print(f"  短训练集标签全零: {n_empty}/500")

    if EVAL_ONLY:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(MODEL_PATH)
        t001a.load_checkpoint(model, MODEL_PATH)
        best_epoch, best_f1 = None, None
    else:
        print("[train] 短序列子集...")
        best_epoch, best_f1 = train_short(
            model, train_loader, val_loader, y_va_short, pw_tensor)
        t001a.load_checkpoint(model, MODEL_PATH)

    print("[infer] 只对短样本前向...")
    t003_va_short = infer_rows(model, seqs, val_short_idx)
    t003_te_short = infer_rows(model, test_seqs, np.where(short_te)[0])

    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t001a_val = t1["t001a_val"]
    t001a_test = t1["t001a_test"]
    specialist_val = t001a_val.copy()
    specialist_val[short_va] = t003_va_short
    specialist_test = t001a_test.copy()
    specialist_test[short_te] = t003_te_short

    t001a_short_f05 = f1_05(t001a_val[short_va], y_va_short)
    t003_short_f05 = f1_05(t003_va_short, y_va_short)
    t001a_short_f100 = f1_100(t001a_val[short_va], y_va_short)
    t003_short_f100 = f1_100(t003_va_short, y_va_short)
    corr = float(np.corrcoef(t001a_val[short_va].ravel(), t003_va_short.ravel())[0, 1])
    le128 = lens_va <= 128
    print(f"  短 val T001a F1(0.5)={t001a_short_f05:.4f} F1(100)={t001a_short_f100:.4f}")
    print(f"  短 val T003  F1(0.5)={t003_short_f05:.4f} F1(100)={t003_short_f100:.4f}")
    print(f"  corr(T001a,T003|short)={corr:.4f}")
    if le128.any():
        print(f"  ≤128 T001a F1(0.5)={f1_05(t001a_val[le128], y_val[le128]):.4f} "
              f"T003={f1_05(specialist_val[le128], y_val[le128]):.4f} "
              f"n={int(le128.sum())}")

    print("[I019] 短行替换 T001a...")
    cnn = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim = i011.ALIGN_W * i011.agg_labels(idx_v, bits_v, y_train) + (
        1.0 - i011.ALIGN_W) * i011.build_kmer_val(y_train)
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]

    s019 = i019_from_t001a(t001a_val, cnn, t2, sim, y_train, y_val, w)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    rec0["f1_short"] = f1_on_mask(s019, y_val, short_va, y_va_short)
    print(f"  I019 F1={rec0['f1_100']:.4f} short={rec0['f1_short']:.4f} "
          f"miss={rec0['f1_miss']:.4f}")

    rows = [rec0]
    configs = {"i019": s019}
    mixed_t1 = {"i019": t001a_val}
    for a in A_GRID:
        mixed = blend_short(t001a_val, specialist_val, short_va, a)
        s = i019_from_t001a(mixed, cnn, t2, sim, y_train, y_val, w)
        name = f"short_a{a:.2f}"
        rec = i011.eval_ens(s, y_val, hit, miss, name, {"alpha": a})
        rec["delta_i019"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rec["f1_short"] = f1_on_mask(s, y_val, short_va, y_va_short)
        rows.append(rec)
        configs[name] = s
        mixed_t1[name] = mixed
        print(f"  {name:12s} F1={rec['f1_100']:.4f} d={rec['delta_i019']:+.4f} "
              f"short={rec['f1_short']:.4f} miss={rec['f1_miss']:.4f}")

    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(y_val))
    mid = len(y_val) // 2
    folds = [perm[:mid], perm[mid:]]

    def oof_f1(score):
        pred = np.zeros_like(score, dtype=np.int8)
        for i, hold in enumerate(folds):
            fit = folds[1 - i]
            th = i011.search_t(score[fit], y_val[fit])
            pred[hold] = (score[hold] >= th).astype(np.int8)
        return round(i011.macro_f1(pred, y_val), 6), pred

    oof_rows = []
    oof0, _ = oof_f1(s019)
    for name, score in tqdm(configs.items(), desc="    OOF", unit="cfg"):
        f1, pred = oof_f1(score)
        rec = {
            "name": f"oof_{name}",
            "f1_100": f1,
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
            "f1_short": round(i011.macro_f1(pred[short_va], y_va_short), 6),
        }
        rec["delta_i019"] = round(f1 - oof0, 6)
        oof_rows.append(rec)
        print(f"  oof_{name:12s} F1={f1:.4f} d={rec['delta_i019']:+.4f} "
              f"short={rec['f1_short']:.4f} miss={rec['f1_miss']:.4f}")

    cand = [r for r in oof_rows if r["name"] != "oof_i019"]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof0
    useful = bool(gain >= 0.001)
    print(f"  OOF 最优 {best_oof['name']} vs I019 {gain:+.4f} → "
          f"{'有效' if useful else '无增益'}")

    os.makedirs(REL_DIR, exist_ok=True)
    np.savez_compressed(
        PROBS_PATH,
        t003_val=specialist_val,
        t003_test=specialist_test,
        t003_val_short=t003_va_short,
        short_val=short_va,
        short_test=short_te,
        y_val_short=y_va_short,
    )

    report = {
        "len_cut": LEN_CUT,
        "max_len": MAX_LEN,
        "n_params": n_params,
        "n_train_short": int(short_tr.sum()),
        "n_val_short": int(short_va.sum()),
        "n_test_short": int(short_te.sum()),
        "n_label_empty_short_train": n_empty,
        "best_epoch": best_epoch,
        "best_short_f1_05": None if best_f1 is None else round(float(best_f1), 6),
        "t001a_short_f1_05": round(t001a_short_f05, 6),
        "t003_short_f1_05": round(t003_short_f05, 6),
        "t001a_short_f1_100": round(t001a_short_f100, 6),
        "t003_short_f1_100": round(t003_short_f100, 6),
        "corr_t001a_t003_short": round(corr, 6),
        "baseline_i019": rec0,
        "oof_i019": oof0,
        "best_oof": best_oof,
        "gain_oof_vs_i019": round(gain, 6),
        "useful": useful,
        "grid": rows,
        "oof_grid": oof_rows,
        "seconds": round(time.time() - t0, 1),
    }
    if useful:
        report["conclusion"] = "I019 OOF≥+0.001，短序列 T001a 可换"
    else:
        report["conclusion"] = "短序列专家未过融合门槛"

    if GEN_SUB and useful:
        cnn_te = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
        t002_te = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
        best_name = best_oof["name"].replace("oof_", "")
        alpha = 1.0 if best_name == "i019" else float(best_name.split("a")[1])
        mixed_te = blend_short(t001a_test, specialist_test, short_te, alpha)
        deep_te = w["t001a"] * mixed_te + w["t002"] * t002_te + w["cnn"] * cnn_te
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
        mixed_v = configs[best_name]
        t_sub = i011.search_t(mixed_v, y_val)
        t1_v = mixed_t1[best_name]
        deep_v = w["t001a"] * t1_v + w["t002"] * t2 + w["cnn"] * cnn
        s011_v = deep_v + i011.SIM_W * sim + BETA_RRF * i011.col_norm(
            i011.rrf_pair(deep_v, sim, K_RRF))
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
        sub_path = os.path.join(SUB_DIR, "submission_T003_short.csv")
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
