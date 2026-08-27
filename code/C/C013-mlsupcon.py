"""
C013-mlsupcon — C002b + Jaccard 加权多标签对比损失
================================================================
依据:
  - Khosla et al., NeurIPS 2020, arXiv:2004.11362。Supervised Contrastive
    用同类当正对；单标签公式。
  - Zhang & Wu, AAAI 2024, doi:10.1609/aaai.v38i15.29619。多标签对比
    用标签重叠加权，不能「共享一个标签就当正对」。
  - ProtSent: Ofer et al., 2026, arXiv:2605.06830。对比拉开 mean-pool。
    第 96 轮 E005 把这套用在 ESM2 上，要上云。本实验先在本机 CNN 上验证损失。
  - 本赛每蛋白约 25.4 个标签，batch 内「至少共享 1 个标签」几乎全是正对，
    对比信号被冲掉。正对权重 = Jaccard(y_i, y_j)。
自己的改法:
  骨干同 C002b（4 分支膨胀+SE+ASL）。池化特征加 128 维投影头。
  L = ASL + λ·Jaccard-SupCon，λ=0.10，τ=0.20，80 ep cosine。
  独立对照 C002b F1(0.5)=0.2714；接到 I019，α∈{0.50,1.00}。
门槛:
  独立 F1(0.5)>0.2714 且 I019 半折 OOF ≥ +0.001。
用法:
  python code/C/C013-mlsupcon.py            # 有 last 则自动续训
  python code/C/C013-mlsupcon.py --fresh    # 无视 last，从头训
  python code/C/C013-mlsupcon.py --eval-only
输出:
  rel/C013_report.json
  rel/C013_probs.npz
  models/c013-mlsupcon.pt       # 最佳
  models/c013-mlsupcon.last.pt  # 每 epoch 完整状态，中断后续训
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
import torch.nn.functional as F
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

PROJ_DIM = 128
LAMBDA_CON = 0.10
TAU = 0.20
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

MODEL_PATH = os.path.join(MODEL_DIR, "c013-mlsupcon.pt")
LAST_PATH = os.path.join(MODEL_DIR, "c013-mlsupcon.last.pt")
REPORT_PATH = os.path.join(REL_DIR, "C013_report.json")
PROBS_PATH = os.path.join(REL_DIR, "C013_probs.npz")
TOK_CACHE = os.path.join(REL_DIR, "c004_tokens.npz")

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


class CNNWithProj(c002b.MultiScaleCNNv3):
    def __init__(self, n_labels=500, proj_dim=PROJ_DIM, **kwargs):
        super().__init__(n_labels=n_labels, **kwargs)
        feat_dim = c002b.FILTERS * len(c002b.KERNELS) * 2
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def encode(self, x):
        feats = []
        for branch, shortcut, se in zip(self.branches, self.shortcuts, self.ses):
            h = branch(x) + shortcut(x)
            h = se(h)
            feats.append(h.max(dim=2)[0])
            feats.append(h.mean(dim=2))
        return self.dropout(torch.cat(feats, dim=1))

    def forward(self, x):
        h = self.encode(x)
        return self.fc(h), self.proj(h)


def jaccard_supcon(z, y, tau=TAU):
    # 必须在 autocast 外：AMP 会把 fp32 GEMM 再打成 fp16，-1e9 溢出
    with torch.amp.autocast("cuda", enabled=False):
        z = F.normalize(z.float(), dim=1)
        sim = z @ z.T / float(tau)
        yf = y.float()
        inter = yf @ yf.T
        card = yf.sum(1)
        union = card[:, None] + card[None, :] - inter
        jac = inter / union.clamp(min=1.0)
        b = z.size(0)
        mask = ~torch.eye(b, dtype=torch.bool, device=z.device)
        logits = sim.masked_fill(~mask, -1e4)
        log_prob = sim - torch.logsumexp(logits, dim=1, keepdim=True)
        w = jac * mask.float()
        w_sum = w.sum(1)
        has_pos = w_sum > 1e-6
        if not has_pos.any():
            return z.new_zeros(())
        loss = -(w * log_prob).sum(1) / w_sum.clamp(min=1e-6)
        return loss[has_pos].mean()


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
            logits, _ = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs).astype(np.float32)


def train_epoch(model, loader, optimizer, asl, scaler):
    model.train()
    total, n, n_con = 0.0, 0, 0.0
    pbar = tqdm(loader, desc="Epoch", unit="batch", leave=False)
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with c002b.autocast_ctx():
            logits, z = model(x)
            l_asl = asl(logits, y)
        l_con = jaccard_supcon(z, y)
        loss = l_asl.float() + LAMBDA_CON * l_con
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
        n_con += float(l_con.detach()) * bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(n, 1), n_con / max(n, 1)


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
        loss, lcon = train_epoch(model, tr_loader, opt, asl, scaler)
        ev = evaluate_probs(model, va_loader)
        f1 = f1_05(ev, y_val)
        sched.step()
        print(f"  Epoch {ep:2d}/{N_EPOCHS} | loss={loss:.4f} con={lcon:.4f} | "
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
    print(f"C013: C002b + Jaccard-SupCon λ={LAMBDA_CON} τ={TAU}")
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
        list(zip(torch.from_numpy(X_tr).long(),
                 torch.from_numpy(y_train.astype(np.float32)))),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=c002b.collate_onehot)
    va_loader = DataLoader(
        list(zip(torch.from_numpy(X_va).long(),
                 torch.from_numpy(y_val.astype(np.float32)))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=c002b.collate_onehot)
    te_loader = DataLoader(
        list(zip(torch.from_numpy(X_te).long(), torch.zeros(len(X_te), n_labels))),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=c002b.collate_onehot)

    model = CNNWithProj(n_labels=n_labels).to(DEVICE)
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
    c013_val = evaluate_probs(model, va_loader)
    c013_test = evaluate_probs(model, te_loader)
    cnn42 = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    f1_new = f1_05(c013_val, y_val)
    f1_old = f1_05(cnn42, y_val)
    corr = float(np.corrcoef(cnn42.ravel(), c013_val.ravel())[0, 1])
    print(f"  C002b F1(0.5)={f1_old:.4f}  C013 F1(0.5)={f1_new:.4f}  corr={corr:.3f}")

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
        mix_cnn = (1.0 - a) * cnn42 + a * c013_val
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
    np.savez_compressed(PROBS_PATH, val=c013_val, test=c013_test, y_true=y_val)
    report = {
        "n_params": n_params,
        "lambda_con": LAMBDA_CON,
        "tau": TAU,
        "proj_dim": PROJ_DIM,
        "best_epoch": best_epoch,
        "f1_05_c002b": round(f1_old, 6),
        "f1_05_c013": round(f1_new, 6),
        "corr_c002b_c013": round(corr, 6),
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
        report["conclusion"] = "独立未超 C002b 0.2714，Jaccard 对比关闭"
    else:
        report["conclusion"] = "独立涨、融合未过 +0.001"

    if GEN_SUB and useful:
        cnn_te = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
        t002_te = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
        best_name = best_oof["name"].replace("oof_", "")
        alpha = 1.0 if best_name == "i019" else float(best_name.split("a")[1])
        mix_te = (1.0 - alpha) * cnn_te + alpha * c013_test
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
        sub_path = os.path.join(SUB_DIR, "submission_C013_mlsupcon.csv")
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
