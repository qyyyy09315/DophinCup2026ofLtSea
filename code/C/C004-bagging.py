"""
C004-bagging — C002b 多种子 Deep Ensemble（换一元 CNN 分数）
================================================================
依据:
  - Lakshminarayanan, Pritzel & Blundell, NeurIPS 2017,
    arXiv:1612.01474。随机初始化的独立网络平均，校准与
    精度都优于单模型；多样性来自不同盆地，不是 Dropout。
  - Huang et al., ICLR 2017, arXiv:1704.00109。Snapshot
    Ensemble 一次训练取多快照，省算力但多样性弱于独立种子。
  - Wang, Jia & Breckon, ICIP 2019, doi:10.1109/icip.2019.8803793
    多标签 CNN 集成是强基线。
  - C005 换头进四路 +0.0007：CNN 只占 deep 的 0.2，再被 sim 0.5
    稀释。本实验必须接到 I019 上评，不能只看 TCK002。
方案:
  切分仍 random_state=42。复用 seed=42 的 TC003 CNN 概率。
  默认只加 seed=1 做探针；确认 val 相关不太高、I019 OOF≥+0.001
  再加 2,3,5。各 80 epoch，ASL+cosine，与 C002b 同超参。
用法:
  python code/C/C004-bagging.py                  # 探针 seed=1
  python code/C/C004-bagging.py --seeds=1,2,3,5  # 满配
  python code/C/C004-bagging.py --epochs=80 --gen-sub
输出:
  rel/C004_bag_report.json
  rel/C004_bag.npz
  models/c004-seed{k}.pt
"""

import importlib.util
import json
import os
import random
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

SRC = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SRC))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
SPLIT_SEED = 42  # 切分必须与全项目一致；bagging seed 只控训练随机性

# ─── 加载 C002b ───
C002B_MOD = importlib.util.spec_from_file_location("c002b", os.path.join(SRC, "C002b-cnn.py"))
c002b = importlib.util.module_from_spec(C002B_MOD)
C002B_MOD.loader.exec_module(c002b)

BATCH = c002b.BATCH_SIZE
N_EPOCHS = c002b.N_EPOCHS
LR = c002b.LR  # 1e-3


# ─── 对新种子做完整训练 ───
def train_seed(seed, n_epochs, lazy_tokens=False):
    """训练一个 seed 的 C002b + val/test 概率。返回 {'val': np(22760,500), 'test': np(28450,500)}"""
    print(f"\n── seed={seed} ──", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_df = pd.read_csv(f"{PROJECT_ROOT}/data/train.csv")
    test_df = pd.read_csv(f"{PROJECT_ROOT}/data/test.csv")
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    n_labels = len(label_cols)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=SPLIT_SEED)
    y_train, y_val = train_df[label_cols].values.astype(np.int8)[idx_train], \
                     train_df[label_cols].values.astype(np.int8)[idx_val]
    seqs = train_df["sequence"].values

    tok_cache = f"{REL_DIR}/c004_tokens.npz"
    if os.path.exists(tok_cache):
        d = np.load(tok_cache)
        X_tr, X_va, X_te = d["X_tr"], d["X_va"], d["X_te"]
    else:
        X_tr = c002b.encode_sequences([s.replace("U","X").replace("Z","X").replace("O","X")
                                        for s in seqs[idx_train]], c002b.MAX_LEN)
        X_va = c002b.encode_sequences([s.replace("U","X").replace("Z","X").replace("O","X")
                                        for s in seqs[idx_val]], c002b.MAX_LEN)
        X_te = c002b.encode_sequences([s.replace("U","X").replace("Z","X").replace("O","X")
                                        for s in test_df["sequence"].values], c002b.MAX_LEN)
        np.savez_compressed(tok_cache, X_tr=X_tr, X_va=X_va, X_te=X_te)

    tr_loader = DataLoader(
        list(zip(torch.from_numpy(X_tr).long(), torch.from_numpy(y_train.astype(np.float32)))),
        batch_size=BATCH, shuffle=True, num_workers=0, collate_fn=c002b.collate_onehot)
    va_loader = DataLoader(
        list(zip(torch.from_numpy(X_va).long(), torch.from_numpy(y_val.astype(np.float32)))),
        batch_size=BATCH, shuffle=False, num_workers=0, collate_fn=c002b.collate_onehot)
    te_loader = DataLoader(
        list(zip(torch.from_numpy(X_te).long(), torch.zeros(len(X_te), n_labels))),
        batch_size=BATCH, shuffle=False, num_workers=0, collate_fn=c002b.collate_onehot)

    model = c002b.MultiScaleCNNv3(n_labels=n_labels).cuda()
    ckpt = f"{MODEL_DIR}/c004-seed{seed}.pt"
    if os.path.exists(ckpt):
        print(f"  已有 checkpoint，跳过训练: {ckpt}", flush=True)
        model.load_state_dict(torch.load(ckpt, map_location="cuda"))
        return {"val": evaluate_probs(model, va_loader), "test": evaluate_probs(model, te_loader)}
    criterion = c002b.AsymmetricLoss().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_f1 = -1
    for ep in range(n_epochs):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(tr_loader, desc=f"seed{seed} ep{ep+1}", leave=False):
            x, y = x.cuda(), y.cuda()
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * x.size(0)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == n_epochs - 1:
            ev = evaluate_probs(model, va_loader)
            f1 = np.mean([f1_score(y_val[:, i].astype(bool), ev[:, i] >= 0.5, zero_division=0)
                          for i in range(n_labels) if y_val[:, i].sum() > 0])
            print(f"  ep{ep+1}: loss={total_loss/len(tr_loader.dataset):.4f} F1(0.5)={f1:.4f}", flush=True)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), f"{MODEL_DIR}/c004-seed{seed}.pt")

    ckpt = f"{MODEL_DIR}/c004-seed{seed}.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cuda"))
    probs_val = evaluate_probs(model, va_loader)
    probs_test = evaluate_probs(model, te_loader)
    return {"val": probs_val, "test": probs_test}


@torch.no_grad()
def evaluate_probs(model, loader):
    model.eval()
    probs = []
    for x, _ in loader:
        x = x.cuda()
        with torch.amp.autocast("cuda"):
            logits = model(x)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs)


def macro_f1_fixed(ens, y_true, t=0.5):
    pred = (ens >= t)
    f1s = []
    for i in range(500):
        yt = y_true[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def macro_rp(ens, y_true, t):
    pred = (ens >= t)
    rs, ps = [], []
    for i in range(500):
        yt = y_true[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
        rs.append(tp / (tp + fn) if tp + fn else 0.0)
        ps.append(tp / (tp + fp) if tp + fp else 0.0)
    return float(np.mean(rs)), float(np.mean(ps))


def macro_auc(ens, y_true):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for i in range(500):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, ens[:, i]))
    return float(np.mean(aucs))


def _build_sim_k6():
    """TCK002 同款 k=6 top-3 加权 sim（val 91,036 / test 全量）。"""
    train_df = pd.read_csv(f"{PROJECT_ROOT}/data/train.csv")
    y_full = train_df[[c for c in train_df.columns if c.startswith("label_")]].values.astype(np.int8)
    idx_tr, idx_val = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=SPLIT_SEED)
    y_tr, y_val = y_full[idx_tr], y_full[idx_val]
    d = np.load(f"{REL_DIR}/CK002_neigh_k6.npz")
    n = d["idx_val"].shape[0]
    lab = y_tr[d["idx_val"][:, :3].ravel()].reshape(n, 3, -1).astype(np.float32)
    s = d["sim_val"][:, :3].astype(np.float32)
    ww = np.maximum(s, 1e-7)
    sim_val = (lab * ww[:, :, None]).sum(axis=1) / ww.sum(axis=1)[:, None]
    dte = np.load(f"{REL_DIR}/TCK002a_neigh_full.npz")
    n2 = dte["idx_test"].shape[0]
    labt = y_full[dte["idx_test"][:, :3].ravel()].reshape(n2, 3, -1).astype(np.float32)
    st = dte["sim_test"][:, :3].astype(np.float32)
    wwt = np.maximum(st, 1e-7)
    sim_test = (labt * wwt[:, :, None]).sum(axis=1) / wwt.sum(axis=1)[:, None]
    return sim_val, sim_test, y_val


def eval_ensemble(cnn_probs_val, cnn_probs_test, suffix, sim_val, sim_test, y_val):
    """代入四路集成 → val F1 + test probs（含 sim 路，权重同 TCK002）。"""
    t1 = np.load(f"{REL_DIR}/TCK001_t001a_probs.npz")
    t001a_val, t001a_test = t1["t001a_val"], t1["t001a_test"]
    t2 = np.load(f"{REL_DIR}/T002_val_probs.npz")
    t002_val = t2["t002"]
    t002_test = np.load(f"{REL_DIR}/T002_test_probs.npz")["t002"]
    w = json.load(open(f"{PROJECT_ROOT}/models/tck002-ensemble.json"))["best_w"]
    ens_val = (w["t001a"] * t001a_val + w["t002"] * t002_val
               + w["cnn"] * cnn_probs_val + w["sim"] * sim_val)
    ens_test = (w["t001a"] * t001a_test + w["t002"] * t002_test
                + w["cnn"] * cnn_probs_test + w["sim"] * sim_test)
    cands = np.linspace(0.01, 0.99, 100)
    best_t = np.full(500, 0.5)
    for i in tqdm(range(500), desc=f"    {suffix} 阈值", unit="label", leave=False):
        yt = y_val[:, i]
        if yt.sum() == 0:
            continue
        p = ens_val[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (p & ytm).sum(0); fp = (p & ~ytm).sum(0); fn = (~p & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_t[i] = cands[f1t.argmax()]
    # F1
    f1 = np.mean([f1_score(y_val[:, i].astype(bool), ens_val[:, i] >= best_t[i], zero_division=0)
                  for i in range(500) if y_val[:, i].sum() > 0])
    # R/P/AUC
    r, p = macro_rp(ens_val, y_val, best_t)
    auc = macro_auc(ens_val, y_val)
    print(f"  {suffix} 四路集成 100档: F1={f1:.4f} R={r:.4f} P={p:.4f} AUC={auc:.4f}")
    return {"f1_100": round(f1, 4), "recall": round(r, 4), "precision": round(p, 4), "auc": round(auc, 4),
            "ens_val": ens_val, "ens_test": ens_test, "best_t": best_t, "y_val": y_val}


def main():
    n_epochs = N_EPOCHS
    seeds_new = [1]  # 探针；满配再 --seeds=1,2,3,5
    for a in sys.argv:
        if a.startswith("--seeds="):
            seeds_new = [int(s) for s in a.split("=")[1].split(",")]
        if a.startswith("--epochs="):
            n_epochs = int(a.split("=")[1])
    lazy_t = "--lazy" in sys.argv
    gen_sub = "--gen-sub" in sys.argv

    t0 = time.time()
    print("=" * 60)
    print(f"C004-bagging: 多种子 CNN bagging | 种子={seeds_new} 各{n_epochs}ep")

    # ─── 1. 读现有 seed=42 probs ───
    print("[1/4] 读取 seed=42 概率 (TC003)...")
    s42_val = np.load(f"{REL_DIR}/TC003_val_probs.npz")["cnn"]   # (22760,500)
    s42_test = np.load(f"{REL_DIR}/TC003_test_probs.npz")["cnn"] # (28450,500)

    # ─── 2. 对新种子训练 ───
    print(f"[2/4] 训练 seeds={seeds_new}...")
    probs = {"val": [s42_val], "test": [s42_test]}
    for s in seeds_new:
        p = train_seed(s, n_epochs, lazy_t)
        probs["val"].append(p["val"])
        probs["test"].append(p["test"])

    # ─── 3. 平均 ───
    print("[3/4] 平均各 seed 概率...")
    n_seeds = len(probs["val"])
    cnn_bag_val = np.mean(probs["val"], axis=0)
    cnn_bag_test = np.mean(probs["test"], axis=0)
    np.savez_compressed(f"{REL_DIR}/C004_bag.npz",
                         val=cnn_bag_val, test=cnn_bag_test,
                         val_seeds=np.stack(probs["val"], axis=-1),
                         test_seeds=np.stack(probs["test"], axis=-1))

    # ─── 4. 接到 I019 评估（主）+ 旧四路（辅）───
    print("[4/4] I019 评估...")
    i011_spec = importlib.util.spec_from_file_location(
        "i011", os.path.join(PROJECT_ROOT, "code", "I", "I011-rrf.py"))
    i011 = importlib.util.module_from_spec(i011_spec)
    i011_spec.loader.exec_module(i011)
    i019_spec = importlib.util.spec_from_file_location(
        "i019", os.path.join(PROJECT_ROOT, "code", "I", "I019-sinkhorn.py"))
    i019 = importlib.util.module_from_spec(i019_spec)
    i019_spec.loader.exec_module(i019)

    train_df = pd.read_csv(os.path.join(i011.DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=SPLIT_SEED)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    val_ids = train_df["protein_id"].values[idx_val]
    train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    tsv = os.path.join(i011.CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
    idx_v, bits_v = i011.parse_aln(tsv, val_ids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim_bit = i011.agg_labels(idx_v, bits_v, y_train)
    sim_kmer = i011.build_kmer_val(y_train)
    sim = i011.ALIGN_W * sim_bit + (1.0 - i011.ALIGN_W) * sim_kmer
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))
    w = json.load(open(os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")))["best_w"]

    def i019_from_cnn(cnn):
        deep = w["t001a"] * t1["t001a_val"] + w["t002"] * t2["t002"] + w["cnn"] * cnn
        s011 = deep + i011.SIM_W * sim + 0.50 * i011.col_norm(
            i011.rrf_pair(deep, sim, 30))
        n = s011.shape[0]
        pi = y_train.mean(0).astype(np.float64)
        t_i011 = i011.search_t(s011, y_val)
        row_m = np.maximum((s011 >= t_i011).sum(1).astype(np.float64), 1.0)
        col_m = np.maximum(pi * n * 1.0, 1e-6)
        col_m *= row_m.sum() / col_m.sum()
        ker = np.exp((s011 - s011.max(axis=1, keepdims=True)) / 0.20)
        p = i019.sinkhorn(ker, row_m, col_m)
        return i019.mix(s011, p, 0.60)

    s019 = i019_from_cnn(s42_val)
    sbag = i019_from_cnn(cnn_bag_val)
    rec0 = i011.eval_ens(s019, y_val, hit, miss, "i019")
    recb = i011.eval_ens(sbag, y_val, hit, miss, f"bag_n{n_seeds}")
    recb["delta_i019"] = round(recb["f1_100"] - rec0["f1_100"], 6)
    print(f"  I019     F1={rec0['f1_100']:.4f} miss={rec0['f1_miss']:.4f}")
    print(f"  bag n={n_seeds} F1={recb['f1_100']:.4f} d={recb['delta_i019']:+.4f} "
          f"miss={recb['f1_miss']:.4f}")

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
        return round(i011.macro_f1(pred, y_val), 6)

    oof0 = oof_f1(s019)
    oofb = oof_f1(sbag)
    gain_oof = round(oofb - oof0, 6)
    print(f"  OOF I019={oof0:.4f} bag={oofb:.4f} d={gain_oof:+.4f}")

    corr = float(np.corrcoef(s42_val.ravel(), cnn_bag_val.ravel())[0, 1])
    print(f"  corr(seed42, bag)={corr:.4f}")

    sim_val, sim_test, y_val_k6 = _build_sim_k6()
    res42 = eval_ensemble(s42_val, s42_test, "seed=42", sim_val, sim_test, y_val_k6)
    res_bag = eval_ensemble(cnn_bag_val, cnn_bag_test, f"n={n_seeds} bag",
                            sim_val, sim_test, y_val_k6)

    useful = bool(gain_oof >= 0.001)
    report = {
        "n_seeds": n_seeds,
        "seeds_new": seeds_new,
        "epochs": n_epochs,
        "corr_seed42_bag": round(corr, 6),
        "i019": rec0,
        "bag_i019": recb,
        "oof_i019": oof0,
        "oof_bag": oofb,
        "gain_oof_vs_i019": gain_oof,
        "useful": useful,
        "tck002_seed42": {k: res42[k] for k in ("f1_100", "recall", "precision", "auc")},
        "tck002_bag": {k: res_bag[k] for k in ("f1_100", "recall", "precision", "auc")},
    }
    if useful:
        report["conclusion"] = "I019 OOF≥+0.001，可加种子"
        print("结论: 有效，可加种子")
    else:
        report["conclusion"] = "I019 OOF 未过 +0.001"
        print("结论: 未过门槛")
    if gen_sub and useful:
        test_df = pd.read_csv(os.path.join(i011.DATA_DIR, "test.csv"))
        y_pred = (res_bag["ens_test"] >= res_bag["best_t"][None, :]).astype(np.int8)
        os.makedirs(SUB_DIR, exist_ok=True)
        sub = test_df[["protein_id"]].copy()
        sub = pd.concat([sub, pd.DataFrame(y_pred, columns=label_cols, index=sub.index)], axis=1)
        sub_path = os.path.join(SUB_DIR, "submission_C004_bag.csv")
        sub.to_csv(sub_path, index=False)
        report["submission"] = sub_path
        print(f"  提交: {sub_path}")

    with open(f"{REL_DIR}/C004_bag_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REL_DIR}/C004_bag_report.json")
    print(f"  种子数: {n_seeds} (42 + {seeds_new})")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()