"""
C014-homology-weight — C002b + 同源 bitscore 反比样本权
================================================================
依据:
  - DeepGOPlus: Kulmanov & Hoehndorf, Bioinformatics 2021,
    doi:10.1093/bioinformatics/btaa763。BLAST 命中走同源，无命中走 CNN。
    I043 把分流放在推理上，OOF +0.0001，已关。本实验把分流放到训练权。
  - Curriculum learning: Bengio et al., ICML 2009,
    doi:10.1145/1553374.1553380。由易到难排样本。这里反过来：
    同源强的样本 sim 已经会做，训练权压低，让 CNN 多看 miss。
  - Logit adjustment: Menon et al., ICLR 2021, arXiv:2007.07314。
    长尾校正的样本侧版本：用 bitscore 当「已被模板覆盖」的置信度。
自己的改法:
  骨干/ASL/80ep cosine 同 C002b。每条训练样本
    w = 1 / (1 + bitscore_max / τ)，τ=50。
  bitscore 来自 train→train DIAMOND（very-sensitive, e=0.001），
  去掉自身命中。无命中 w=1。接到 I019，α∈{0.50,1.00}。
门槛:
  独立 F1(0.5)>0.2714，或 missF1 明显涨且 I019 半折 OOF ≥ +0.001。
用法:
  python code/C/C014-homology-weight.py            # 有 last 则自动续训
  python code/C/C014-homology-weight.py --fresh    # 无视 last，从头训
  python code/C/C014-homology-weight.py --eval-only
输出:
  rel/C014_report.json
  rel/C014_probs.npz
  models/c014-homology.pt
  models/c014-homology.last.pt
"""

import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

SRC = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CK005_TMP = os.path.join(REL_DIR, "ck005_tmp")
C014_TMP = os.path.join(REL_DIR, "c014_tmp")
DIAMOND = os.path.join(PROJECT_ROOT, "tools", "diamond", "diamond.exe")

C002B_MOD = importlib.util.spec_from_file_location(
    "c002b", os.path.join(SRC, "C002b-cnn.py"))
c002b = importlib.util.module_from_spec(C002B_MOD)
C002B_MOD.loader.exec_module(c002b)

TAU_BIT = 50.0
EVALUE = 0.001
MAX_ACCEPT = 5
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

MODEL_PATH = os.path.join(MODEL_DIR, "c014-homology.pt")
LAST_PATH = os.path.join(MODEL_DIR, "c014-homology.last.pt")
REPORT_PATH = os.path.join(REL_DIR, "C014_report.json")
PROBS_PATH = os.path.join(REL_DIR, "C014_probs.npz")
TOK_CACHE = os.path.join(REL_DIR, "c004_tokens.npz")
WEIGHT_CACHE = os.path.join(C014_TMP, "train_max_bitscore.npz")
TSV_PATH = os.path.join(C014_TMP, "align_train_very-sensitive_e0.001.tsv")

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
def evaluate_probs(model, loader):
    model.eval()
    probs = []
    for x, _ in loader:
        x = x.to(DEVICE)
        with c002b.autocast_ctx():
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


class WeightedASL(c002b.AsymmetricLoss):
    def forward(self, logits, targets, sample_w=None):
        if sample_w is None:
            return super().forward(logits, targets)
        p = torch.sigmoid(logits.float()).clamp(self.eps, 1.0 - self.eps)
        pos_loss = -targets * ((1.0 - p) ** self.gamma_pos) * torch.log(p)
        p_m = torch.clamp(p - self.clip, min=0.0)
        neg_loss = -(1.0 - targets) * (p_m ** self.gamma_neg) * torch.log(1.0 - p_m)
        per = (pos_loss + neg_loss).mean(dim=1)
        w = sample_w.float()
        return (per * w).sum() / w.sum().clamp(min=1e-6)


def train_epoch(model, loader, optimizer, asl, scaler):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="Epoch", unit="batch", leave=False)
    for batch in pbar:
        if len(batch) == 3:
            x, y, w = batch
        else:
            x, y = batch
            w = None
        x, y = x.to(DEVICE), y.to(DEVICE)
        if w is not None:
            w = w.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with c002b.autocast_ctx():
            logits = model(x)
            loss = asl(logits, y, w)
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


def collate_weighted(batch):
    seqs = torch.stack([item[0] for item in batch]).long()
    labels = torch.stack([item[1] for item in batch])
    weights = torch.stack([item[2] for item in batch])
    one_hot = torch.nn.functional.one_hot(
        seqs, num_classes=c002b.N_CHANNELS).permute(0, 2, 1).float()
    return one_hot, labels, weights


def run_diamond_train(train_ids):
    os.makedirs(C014_TMP, exist_ok=True)
    if os.path.exists(TSV_PATH) and os.path.getsize(TSV_PATH) > 0:
        print(f"  复用 {TSV_PATH}", flush=True)
        return TSV_PATH
    fa_train = os.path.join(CK005_TMP, "train.fasta")
    db_train = os.path.join(CK005_TMP, "train.dmnd")
    if not os.path.exists(fa_train) or not os.path.exists(db_train):
        raise FileNotFoundError("缺少 ck005_tmp/train.fasta 或 train.dmnd")
    args = [
        DIAMOND, "blastp", "-d", db_train, "-q", fa_train, "-o", TSV_PATH,
        "--max-target-seqs", str(MAX_ACCEPT + 1), "-e", str(EVALUE),
        "--very-sensitive", "--outfmt", "6",
        "qseqid", "sseqid", "bitscore", "evalue", "pident",
        "--threads", "8",
    ]
    print("  DIAMOND train→train very-sensitive e=0.001 ...", flush=True)
    r = subprocess.run(args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print((r.stdout or "")[-2000:])
        print((r.stderr or "")[-2000:])
        raise RuntimeError("diamond train→train 失败")
    return TSV_PATH


def max_bitscore_per_query(tsv, query_ids):
    mx = np.zeros(len(query_ids), dtype=np.float32)
    qpos = {q: i for i, q in enumerate(query_ids)}
    with open(tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            q, t = p[0], p[1]
            if q not in qpos:
                continue
            if q == t:
                continue
            try:
                bs, ev = float(p[2]), float(p[3])
            except ValueError:
                continue
            if ev > EVALUE:
                continue
            i = qpos[q]
            if bs > mx[i]:
                mx[i] = bs
    return mx


def sample_weights(train_ids):
    os.makedirs(C014_TMP, exist_ok=True)
    if os.path.exists(WEIGHT_CACHE):
        d = np.load(WEIGHT_CACHE)
        if len(d["w"]) == len(train_ids):
            return d["w"], d["bits"]
    tsv = run_diamond_train(train_ids)
    bits = max_bitscore_per_query(tsv, train_ids)
    w = 1.0 / (1.0 + bits / TAU_BIT)
    np.savez_compressed(WEIGHT_CACHE, w=w.astype(np.float32), bits=bits)
    return w.astype(np.float32), bits


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


def train_model(model, tr_loader, va_loader, y_val):
    asl = WeightedASL().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=N_EPOCHS, eta_min=LR / 100)
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
        loss = train_epoch(model, tr_loader, opt, asl, scaler)
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
    print(f"C014: C002b + homology weight τ={TAU_BIT}")
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

    print("[weights] train→train DIAMOND ...")
    w_tr, bits = sample_weights(train_ids)
    n_hit = int((bits > 0).sum())
    print(f"  hit={n_hit}/{len(bits)} ({n_hit/len(bits):.3f}) "
          f"bits median={float(np.median(bits[bits>0])) if n_hit else 0:.1f} "
          f"w mean={float(w_tr.mean()):.3f} min={float(w_tr.min()):.3f}",
          flush=True)

    X_tr, X_va, X_te = load_tokens(seqs, test_df["sequence"].values, idx_train, idx_val)
    tr_data = list(zip(
        torch.from_numpy(X_tr).long(),
        torch.from_numpy(y_train.astype(np.float32)),
        torch.from_numpy(w_tr)))
    tr_loader = DataLoader(
        tr_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=collate_weighted)
    va_loader = DataLoader(
        list(zip(torch.from_numpy(X_va).long(),
                 torch.from_numpy(y_val.astype(np.float32)))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=c002b.collate_onehot)
    te_loader = DataLoader(
        list(zip(torch.from_numpy(X_te).long(), torch.zeros(len(X_te), n_labels))),
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
        best_epoch, best_f1 = train_model(model, tr_loader, va_loader, y_val)
        c002b.load_checkpoint(model, MODEL_PATH)

    print("[infer] ...")
    c014_val = evaluate_probs(model, va_loader)
    c014_test = evaluate_probs(model, te_loader)
    cnn42 = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    f1_new = f1_05(c014_val, y_val)
    f1_old = f1_05(cnn42, y_val)
    corr = float(np.corrcoef(cnn42.ravel(), c014_val.ravel())[0, 1])
    print(f"  C002b F1(0.5)={f1_old:.4f}  C014 F1(0.5)={f1_new:.4f}  corr={corr:.3f}")

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
        mix_cnn = (1.0 - a) * cnn42 + a * c014_val
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
    useful = bool(gain >= 0.001 and (beat_ind or best_oof["f1_miss"] > rec0["f1_miss"] + 0.01))
    print(f"  独立 {'超' if beat_ind else '未超'} C002b 0.2714")
    print(f"  OOF 最优 {best_oof['name']} vs I019 {gain:+.4f} → "
          f"{'有效' if useful else '无增益'}")

    os.makedirs(REL_DIR, exist_ok=True)
    np.savez_compressed(PROBS_PATH, val=c014_val, test=c014_test, y_true=y_val)
    report = {
        "n_params": n_params,
        "tau_bit": TAU_BIT,
        "n_train_hit": n_hit,
        "w_mean": round(float(w_tr.mean()), 6),
        "w_min": round(float(w_tr.min()), 6),
        "best_epoch": best_epoch,
        "f1_05_c002b": round(f1_old, 6),
        "f1_05_c014": round(f1_new, 6),
        "corr_c002b_c014": round(corr, 6),
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
        report["conclusion"] = "I019 OOF≥+0.001，可换 CNN 路"
    elif not beat_ind:
        report["conclusion"] = "独立未超 C002b 0.2714，同源样本权关闭"
    else:
        report["conclusion"] = "独立涨、融合未过 +0.001"

    if GEN_SUB and useful:
        cnn_te = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
        t002_te = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
        best_name = best_oof["name"].replace("oof_", "")
        alpha = 1.0 if best_name == "i019" else float(best_name.split("a")[1])
        mix_te = (1.0 - alpha) * cnn_te + alpha * c014_test
        deep_te = (w["t001a"] * np.load(
            os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"]
                   + w["t002"] * t002_te + w["cnn"] * mix_te)
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
        sub_path = os.path.join(SUB_DIR, "submission_C014_homology.csv")
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
