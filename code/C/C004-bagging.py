"""
C004-bagging — 基模型 Bagging（多种子集成，提升泛化）
================================================================
动机:
  当前所有基模型只用一个 seed（42）训练。CNN 训练有随机性
  （初始化/Dropout/shuffle），不同 seed 的误差部分不相关 →
  平均后方差降低。Bagging 是经典集成手段，本项目未试过。
方案:
  1. 复用 seed=42 的 C002b 概率（rel/TC003_{val,test}_probs['cnn']）
  2. 新训 seeds=1,2,3,5（各 80epoch），保存模型+概率
     切分固定 random_state=42（与全项目一致）；seed 只控初始化/shuffle/dropout
  3. 平均所有 seed 的概率 → cnn_bag_val/test
  4. 代入四路集成（t001a+t002+cnn_bag+sim），与 TCK002 同权重
  5. 有效 → 生成提交；无效 → 记录负结果
用法:
  python code/C/C004-bagging.py [--seeds=1,2,3,5] [--epochs=80] [--gen-sub]
输出:
  rel/C004_bag_report.json
  submit/submission_C004_bag.csv（若有效）
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
    seeds_new = [1, 2, 3, 5]
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

    # ─── 4. 集成评估 ───
    print("[4/4] 集成评估...")
    sim_val, sim_test, y_val = _build_sim_k6()
    res42 = eval_ensemble(s42_val, s42_test, "seed=42", sim_val, sim_test, y_val)
    res_bag = eval_ensemble(cnn_bag_val, cnn_bag_test, f"n={n_seeds} bag", sim_val, sim_test, y_val)
    gain = res_bag["f1_100"] - res42["f1_100"]
    print(f"  增益: {gain:+.4f} ({res42['f1_100']:.4f} → {res_bag['f1_100']:.4f})")

    def _metrics(res):
        return {k: res[k] for k in ("f1_100", "recall", "precision", "auc")}

    report = {"n_seeds": n_seeds, "seeds_new": seeds_new, "epochs": n_epochs,
              "single_seed_42": _metrics(res42), "bagged": _metrics(res_bag),
              "gain": round(float(gain), 6)}
    if gain <= 0:
        report["conclusion"] = "无增益, bagging 无效"
        print("结论: bagging 无增益, 不生成提交")
    else:
        report["conclusion"] = "有增益"
        print("结论: bagging 有增益")
        if gen_sub:
            test_df = pd.read_csv(f"{PROJECT_ROOT}/data/test.csv")
            label_cols = [c for c in pd.read_csv(f"{PROJECT_ROOT}/data/train.csv", nrows=0).columns
                          if c.startswith("label_")]
            y_pred = (res_bag["ens_test"] >= res_bag["best_t"][None, :]).astype(np.int8)
            os.makedirs(SUB_DIR, exist_ok=True)
            sub = test_df[["protein_id"]].copy()
            sub = pd.concat([sub, pd.DataFrame(y_pred, columns=label_cols, index=sub.index)], axis=1)
            sub_path = f"{SUB_DIR}/submission_C004_bag.csv"
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