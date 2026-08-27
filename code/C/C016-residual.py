"""
C016-residual — C002b 拟合 y−sim 残差
================================================================
依据:
  - Friedman, Annals of Statistics 2001, doi:10.1214/aos/1013203451。
    后一阶段拟合前一阶段残差，不拟合原目标。
  - ProteInfer: Sanderson et al., eLife 2023, doi:10.7554/eLife.80942。
    CNN 与 BLAST 学到不同信息。DeepGOPlus 在推理相加；C002b 对 y
    拟合，梯度被 hit 同源主导。
  - C015 FiLM 把同源灌进 CNN，corr(sim)=0.73，miss 掉，已关。
自己的改法:
  骨干/ASL/80ep cosine 同 C002b。输入只有序列。
  训练目标 y_res = clip(y − sim_hat, 0, 1)。
  sim_hat = 0.4·DIAMOND + 0.6·k-mer 多k（train 内自身命中去掉）。
  推理 p = clip(σ(logit) + sim, 0, 1)，接到 C011a。
门槛:
  相对 C011a 半折 OOF ≥ +0.001，且 missF1 不掉。
用法:
  python code/C/C016-residual.py
  python code/C/C016-residual.py --fresh
  python code/C/C016-residual.py --eval-only
输出:
  rel/C016_report.json
  rel/C016_probs.npz
  models/c016-residual.pt
  models/c016-residual.last.pt
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
C014_TMP = os.path.join(REL_DIR, "c014_tmp")
C016_TMP = os.path.join(REL_DIR, "c016_tmp")

C002B_MOD = importlib.util.spec_from_file_location(
    "c002b", os.path.join(SRC, "C002b-cnn.py"))
c002b = importlib.util.module_from_spec(C002B_MOD)
C002B_MOD.loader.exec_module(c002b)

C014_MOD = importlib.util.spec_from_file_location(
    "c014", os.path.join(SRC, "C014-homology-weight.py"))
c014 = importlib.util.module_from_spec(C014_MOD)
C014_MOD.loader.exec_module(c014)

C015_MOD = importlib.util.spec_from_file_location(
    "c015", os.path.join(SRC, "C015-film.py"))
c015m = importlib.util.module_from_spec(C015_MOD)
C015_MOD.loader.exec_module(c015m)

BATCH_SIZE = c002b.BATCH_SIZE
N_EPOCHS = c002b.N_EPOCHS
LR = c002b.LR
WEIGHT_DECAY = c002b.WEIGHT_DECAY
PATIENCE = c002b.PATIENCE
SPLIT_SEED = 42
A_GRID = (0.50, 1.00)
K_RRF, BETA_RRF = 30, 0.50
TAU_SK, RHO, ALPHA_SK = 0.20, 1.0, 0.60
CNN_W = 0.2
C002B_F05 = 0.2714
C011A_ALPHA = 0.50
ALIGN_W = 0.4
TOP_K = 3

MODEL_PATH = os.path.join(MODEL_DIR, "c016-residual.pt")
LAST_PATH = os.path.join(MODEL_DIR, "c016-residual.last.pt")
REPORT_PATH = os.path.join(REL_DIR, "C016_report.json")
PROBS_PATH = os.path.join(REL_DIR, "C016_probs.npz")
TOK_CACHE = os.path.join(REL_DIR, "c004_tokens.npz")
SIM_CACHE = os.path.join(C016_TMP, "sim_hat.npz")
TSV_TRAIN = os.path.join(C014_TMP, "align_train_very-sensitive_e0.001.tsv")

EVAL_ONLY = "--eval-only" in sys.argv
GEN_SUB = "--gen-sub" in sys.argv
FRESH = "--fresh" in sys.argv
DEVICE = c002b.DEVICE


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


@torch.no_grad()
def evaluate_logits(model, loader):
    model.eval()
    outs = []
    for x, _ in loader:
        x = x.to(DEVICE)
        with c002b.autocast_ctx():
            logits = model(x)
        outs.append(logits.float().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def train_epoch(model, loader, optimizer, asl, scaler):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="Epoch", unit="batch", leave=False)
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with c002b.autocast_ctx():
            logits = model(x)
            loss = asl(logits, y)
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

    def clean(s):
        return s.replace("U", "X").replace("Z", "X").replace("O", "X")

    X_tr = c002b.encode_sequences([clean(s) for s in seqs[idx_train]], c002b.MAX_LEN)
    X_va = c002b.encode_sequences([clean(s) for s in seqs[idx_val]], c002b.MAX_LEN)
    X_te = c002b.encode_sequences([clean(s) for s in test_seqs], c002b.MAX_LEN)
    os.makedirs(REL_DIR, exist_ok=True)
    np.savez_compressed(TOK_CACHE, X_tr=X_tr, X_va=X_va, X_te=X_te)
    return X_tr, X_va, X_te


def kmer_from_neigh(path, y_space, split):
    d = np.load(path)
    key_i = "idx" if split == "train" else f"idx_{split}"
    key_s = "sim" if split == "train" else f"sim_{split}"
    idx, s = d[key_i], d[key_s]
    n = idx.shape[0]
    lab = y_space[idx[:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
    if split == "train":
        self = idx[:, 0] == np.arange(n)
        s = s.copy()
        s[self, 0] = 0.0
        lab[self, 0] = 0.0
    ww = np.maximum(s[:, :5], 1e-7) ** 2
    return (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]


def build_kmer(y_train, y_full, split):
    sim = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        if split == "train":
            if k != 6:
                continue
            path = os.path.join(REL_DIR, "ECK003_neigh_train_k6.npz")
            sim = sim + kmer_from_neigh(path, y_train, "train")
            continue
        if split == "val":
            prefix = "CK003" if k >= 7 else "CK002"
            path = os.path.join(REL_DIR, f"{prefix}_neigh_k{k}.npz")
            sim = sim + wt * kmer_from_neigh(path, y_train, "val")
        else:
            path = os.path.join(
                REL_DIR, "TCK002a_neigh_full.npz" if k == 6
                else f"CK004_neigh_k{k}_full.npz")
            sim = sim + wt * kmer_from_neigh(path, y_full, "test")
    if split == "train":
        return sim
    return sim


def load_sim(train_ids, val_ids, test_ids, y_train, y_full, train_map, full_map):
    os.makedirs(C016_TMP, exist_ok=True)
    if os.path.exists(SIM_CACHE):
        d = np.load(SIM_CACHE)
        if d["tr"].shape[0] == len(train_ids) and d["va"].shape[0] == len(val_ids):
            return d["tr"], d["va"], d["te"]
    if not os.path.exists(TSV_TRAIN):
        c014.run_diamond_train(train_ids)
    idx_tr, bits_tr = c015m.parse_aln_noself(TSV_TRAIN, train_ids, train_map)
    bit_tr = i011.agg_labels(idx_tr, bits_tr, y_train, top_k=TOP_K)
    kmer_tr = build_kmer(y_train, y_full, "train")
    sim_tr = ALIGN_W * bit_tr + (1.0 - ALIGN_W) * kmer_tr
    tsv_va = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_va, bits_va = i011.parse_aln(tsv_va, val_ids, train_map)
    sim_va = ALIGN_W * i011.agg_labels(idx_va, bits_va, y_train) + (
        1.0 - ALIGN_W) * i011.build_kmer_val(y_train)
    tsv_te = os.path.join(i011.CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
    idx_te, bits_te = i011.parse_aln(tsv_te, test_ids, full_map)
    sim_bit_te = i011.agg_labels(idx_te, bits_te, y_full)
    sim_kmer_te = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        path = os.path.join(
            REL_DIR, "TCK002a_neigh_full.npz" if k == 6
            else f"CK004_neigh_k{k}_full.npz")
        sim_kmer_te = sim_kmer_te + wt * kmer_from_neigh(path, y_full, "test")
    sim_te = ALIGN_W * sim_bit_te + (1.0 - ALIGN_W) * sim_kmer_te
    np.savez_compressed(SIM_CACHE, tr=sim_tr, va=sim_va, te=sim_te)
    return sim_tr, sim_va, sim_te


def combine(logits, sim):
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    return np.clip(p + sim, 0.0, 1.0).astype(np.float32)


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


def train_model(model, tr_loader, va_loader, y_val, sim_va):
    asl = c002b.AsymmetricLoss().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=N_EPOCHS, eta_min=LR / 100)
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    best_f1, best_epoch, patience_counter = -1.0, 0, 0
    start_epoch = 1
    os.makedirs(MODEL_DIR, exist_ok=True)

    auto_resume = (not FRESH) and os.path.exists(LAST_PATH)
    if auto_resume:
        ckpt = torch.load(LAST_PATH, map_location=DEVICE, weights_only=False)
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
        loss = train_epoch(model, tr_loader, opt, asl, scaler)
        logits = evaluate_logits(model, va_loader)
        pred = combine(logits, sim_va)
        f1 = f1_05(pred, y_val)
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
        save_last(model, opt, sched, scaler, ep, best_f1, best_epoch, patience_counter)
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {ep}", flush=True)
            break
    print(f"  最佳 epoch: {best_epoch}, F1(0.5)={best_f1:.4f}", flush=True)
    return best_epoch, best_f1


def main():
    global i011, i019
    t0 = time.time()
    print("=" * 60)
    print("C016: C002b residual on y-sim")
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
    test_ids = test_df["protein_id"].values
    train_map = {pid: i for i, pid in enumerate(train_ids)}
    full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}

    i011 = load_mod("i011", os.path.join("I", "I011-rrf.py"))
    i019 = load_mod("i019", os.path.join("I", "I019-sinkhorn.py"))

    print("[sim] train/val/test homology scores ...")
    sim_tr, sim_va, sim_te = load_sim(
        train_ids, val_ids, test_ids, y_train, y_full, train_map, full_map)
    y_res = np.clip(y_train.astype(np.float32) - sim_tr, 0.0, 1.0)
    print(f"  sim_tr mean={float(sim_tr.mean()):.4f} "
          f"y_res mean={float(y_res.mean()):.4f} "
          f"y mean={float(y_train.mean()):.4f}", flush=True)

    X_tr, X_va, X_te = load_tokens(seqs, test_df["sequence"].values, idx_train, idx_val)
    tr_loader = DataLoader(
        list(zip(torch.from_numpy(X_tr).long(),
                 torch.from_numpy(y_res))),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=c002b.collate_onehot)
    va_loader = DataLoader(
        list(zip(torch.from_numpy(X_va).long(),
                 torch.from_numpy(y_val.astype(np.float32)))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=c002b.collate_onehot)
    te_loader = DataLoader(
        list(zip(torch.from_numpy(X_te).long(),
                 torch.zeros(len(X_te), n_labels))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=c002b.collate_onehot)

    model = c002b.MultiScaleCNNv3(n_labels=n_labels).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    if EVAL_ONLY:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(MODEL_PATH)
        c002b.load_checkpoint(model, MODEL_PATH)
        best_epoch, best_f1 = None, None
    else:
        print("[train] ...")
        best_epoch, best_f1 = train_model(model, tr_loader, va_loader, y_val, sim_va)
        c002b.load_checkpoint(model, MODEL_PATH)

    print("[infer] ...")
    logits_va = evaluate_logits(model, va_loader)
    logits_te = evaluate_logits(model, te_loader)
    c016_val = combine(logits_va, sim_va)
    c016_test = combine(logits_te, sim_te)
    cnn_raw = 1.0 / (1.0 + np.exp(-np.clip(logits_va, -20, 20)))
    cnn42 = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    cnn_lock = (1.0 - C011A_ALPHA) * cnn42 + C011A_ALPHA * c005_val
    f1_new = f1_05(c016_val, y_val)
    f1_raw = f1_05(cnn_raw, y_val)
    f1_old = f1_05(cnn42, y_val)
    f1_lock = f1_05(cnn_lock, y_val)
    corr_sim = float(np.corrcoef(sim_va.ravel(), cnn_raw.ravel())[0, 1])
    print(f"  C002b F1(0.5)={f1_old:.4f}  C011a-CNN={f1_lock:.4f}  "
          f"C016-raw={f1_raw:.4f}  C016+sim={f1_new:.4f}  corr(raw,sim)={corr_sim:.3f}")

    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim = sim_va
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    with open(i011.TCK002_CONFIG, encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    deep_rest = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"]

    s_lock = i019_from_cnn(cnn_lock, deep_rest, sim, y_train, y_val)
    rec0 = i011.eval_ens(s_lock, y_val, hit, miss, "c011a")
    print(f"  C011a F1={rec0['f1_100']:.4f} miss={rec0['f1_miss']:.4f}")

    rows = [rec0]
    configs = {"c011a": s_lock}
    mixed_cnn = {"c011a": cnn_lock}
    for a in A_GRID:
        mix_cnn = (1.0 - a) * cnn_lock + a * c016_val
        s = i019_from_cnn(mix_cnn, deep_rest, sim, y_train, y_val)
        name = f"cnn_a{a:.2f}"
        rec = i011.eval_ens(s, y_val, hit, miss, name, {
            "alpha": a, "f1_05": round(f1_05(mix_cnn, y_val), 6)})
        rec["delta_c011a"] = round(rec["f1_100"] - rec0["f1_100"], 6)
        rows.append(rec)
        configs[name] = s
        mixed_cnn[name] = mix_cnn
        print(f"  {name:12s} F1(0.5)={rec['f1_05']:.4f} F1={rec['f1_100']:.4f} "
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
            "f1_miss": round(i011.macro_f1(pred[miss], y_val[miss]), 6),
        }
        oof_rows.append(rec)
        if name == "c011a":
            oof_lock = rec["f1_100"]
        print(f"  oof_{name:12s} F1={rec['f1_100']:.4f}")
    for rec in oof_rows:
        rec["delta_c011a"] = round(rec["f1_100"] - oof_lock, 6)

    cand = [r for r in oof_rows if r["name"] != "oof_c011a"]
    best_oof = max(cand, key=lambda r: r["f1_100"]) if cand else oof_rows[0]
    gain = best_oof["f1_100"] - oof_lock
    miss0 = rec0["f1_miss"]
    miss_ok = best_oof["f1_miss"] + 1e-9 >= miss0
    useful = bool(gain >= 0.001 and miss_ok)
    print(f"  OOF 最优 {best_oof['name']} vs C011a {gain:+.4f} "
          f"miss {best_oof['f1_miss']:.4f} vs {miss0:.4f} → "
          f"{'有效' if useful else '无增益'}")

    os.makedirs(REL_DIR, exist_ok=True)
    np.savez_compressed(PROBS_PATH, val=c016_val, test=c016_test,
                        raw=cnn_raw, y_true=y_val)
    report = {
        "n_params": n_params,
        "sim_tr_mean": round(float(sim_tr.mean()), 6),
        "y_res_mean": round(float(y_res.mean()), 6),
        "best_epoch": best_epoch,
        "f1_05_c002b": round(f1_old, 6),
        "f1_05_c011a_cnn": round(f1_lock, 6),
        "f1_05_c016_raw": round(f1_raw, 6),
        "f1_05_c016": round(f1_new, 6),
        "corr_raw_sim": round(corr_sim, 6),
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
        report["conclusion"] = "C011a OOF≥+0.001 且 miss 不掉，可换 CNN 路"
    else:
        report["conclusion"] = "残差 CNN 未过门槛，关闭"

    if GEN_SUB and useful:
        cnn_te = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
        c005_te = np.load(os.path.join(REL_DIR, "C005_test_probs.npz"))["test"]
        cnn_lock_te = (1.0 - C011A_ALPHA) * cnn_te + C011A_ALPHA * c005_te
        t002_te = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
        best_name = best_oof["name"].replace("oof_", "")
        alpha = 0.0 if best_name == "c011a" else float(best_name.split("a")[1])
        mix_te = (1.0 - alpha) * cnn_lock_te + alpha * c016_test
        deep_te = (w["t001a"] * np.load(
            os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"]
                   + w["t002"] * t002_te + w["cnn"] * mix_te)
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
        sub_path = os.path.join(SUB_DIR, "submission_C016_residual.csv")
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
