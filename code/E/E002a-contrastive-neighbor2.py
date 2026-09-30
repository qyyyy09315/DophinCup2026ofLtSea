"""
E002a-contrastive-neighbor2 — 对比学习嵌入邻居 v2 (FuncSeek 全文细节对齐)
=================================================================
E002 失败复盘 (FuncSeek 全文深读, bioRxiv 2026.08.07):
  - E002 用 batch 内随机负样本 + 无 LayerNorm + 无增强 + 20 epoch
  - FuncSeek/CLEAN 关键组件: SupCon (温度 0.1), LayerNorm 头,
    jitter 增强 (N(0,0.01)), Adam lr 5e-4 cosine, 100-500 epoch,
    困难负样本挖掘 (每类 8 正 + 30 近质心负)
  - 单 ESM2 对比训练 P@1000: 51.6% → 87.1% (+35.5, FuncSeek Table 4)
    "functional signal already latent in ESM2, requires learned projection"
本版补齐: LayerNorm + 温度 0.1 SupCon + jitter + 100 epoch + lr 5e-4 cosine
  + batch 512 (大 batch 自带更多困难负样本; 多标签无天然类,
  正对 = 共享标签 Jaccard ≥ τ, 负样本 = batch 内其余全部)
文献:
  - FuncSeek: Cloete & Patterton, bioRxiv 2026, doi:10.64898/2026.08.07.738656
  - SupCon: Khosla et al., NeurIPS 2020, arXiv:2004.11362
  - CLEAN: Yu et al., 2023 (SupConH 来源)
  - ESM2: Lin et al., Science 2023, doi:10.1126/science.ade2574
输出:
  rel/E002a_report.json
  rel/E002a_emb_proj.npz
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
REPORT_PATH = f"{REL_DIR}/E002a_report.json"
SUB_PATH = f"{SUB_DIR}/submission_E002a_ensemble.csv"
PROJ_PATH = f"{REL_DIR}/E002a_emb_proj.npz"

ESM_TRAIN = f"{REL_DIR}/train_esm2_embeddings.parquet"
ESM_TEST = f"{REL_DIR}/test_esm2_embeddings.parquet"
NEIGH_K6_PATH = f"{REL_DIR}/CK002_neigh_k6.npz"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
T001A_PROBS_PATH = f"{REL_DIR}/TCK001_t001a_probs.npz"
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"
TCK002_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

# FuncSeek 对齐超参
PROJ_DIMS = [640, 512, 256]
TEMPERATURE = 0.1
JACCARD_TAU = 0.3
BATCH_SIZE = 1872      # 48 锚点 × (1+8+30), FuncSeek 式批次组成
EPOCHS = 60
LR = 5e-4
JITTER_STD = 0.01
SEED = 42
# 困难负样本挖掘 (FuncSeek/CLEAN SupConH 核心)
N_POS = 8                # 每锚点正样本数
N_HARD_NEG = 30          # 每锚点困难负样本数
HARD_TOP_K = 50          # 投影空间近邻候选池大小
REFRESH_EVERY = 10       # 近邻池刷新间隔 (epoch)
N_ANCHOR_PER_BATCH = 48  # 每 batch 锚点数 → batch = 48×(8+30) = 1824

TOP_K = 3
VAL_SIZE = 0.2
RANDOM_STATE = 42


class ProjectionHead(nn.Module):
    """3 层 FFN, 每激活前 LayerNorm (FuncSeek 对齐), 输出 L2 归一化。"""

    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            if i > 0:
                layers.append(nn.LayerNorm(dims[i]))
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.nn.functional.normalize(self.net(x), dim=1)


def supcon_loss(z, y_batch, tau, jac_tau):
    """SupCon (Khosla 2020 公式 2) 多标签版: 正对 = Jaccard≥τ 且共享≥1 标签。
    温度 0.1, 对角掩码, loss 只对含正对的锚点平均。"""
    n = z.shape[0]
    sim = (z @ z.T) / tau
    sim.fill_diagonal_(-1e9)
    inter = y_batch @ y_batch.T
    sums = y_batch.sum(1, keepdim=True)
    jac = inter / (sums + sums.T - inter).clamp(min=1e-9)
    pos = (jac >= jac_tau) & (inter > 0)
    n_pos = pos.sum(1)
    valid = n_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z.device), 0.0
    logits = torch.exp(sim)                       # 对角已 -1e9 → exp≈0
    denom = logits.sum(1, keepdim=True) + 1e-9
    log_prob = (logits / denom).clamp(min=1e-9).log()   # (n, n)
    loss = -(log_prob * pos.float()).sum(1) / n_pos.clamp(min=1)
    loss = loss[valid].mean()
    return loss, float(pos.sum() / n)


def macro_auc(y_true, y_prob):
    aucs = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, y_prob[:, i]))
    return float(np.mean(aucs))


def macro_f1_fixed(ens, y_true, t=0.5):
    pred = (ens >= t)
    f1s = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def knn_aggregate(query_emb, db_emb, db_y, top_k):
    q = torch.from_numpy(query_emb.astype(np.float32))
    db = torch.from_numpy(db_emb.astype(np.float32))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q, db = q.to(dev), db.to(dev)
    n = q.shape[0]
    idx = np.zeros((n, 5), dtype=np.int64)
    sims = np.zeros((n, 5), dtype=np.float32)
    block = 1000 if dev == "cuda" else 200
    for s in tqdm(range(0, n, block), desc="最近邻检索"):
        e = min(s + block, n)
        sim = q[s:e] @ db.T
        top = torch.topk(sim, 5, dim=1)
        idx[s:e] = top.indices.cpu().numpy()
        sims[s:e] = top.values.cpu().numpy()
    labels = db_y[idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    w = np.maximum(sims[:, :top_k], 1e-7)
    return (labels * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]


def load_emb(path, prot_ids):
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    emb = df[emb_cols].values.astype(np.float32)
    ids = df["protein_id"].values
    if not np.array_equal(ids, prot_ids):
        pos = pd.Series(ids).reset_index().set_index(0).loc[prot_ids, "index"].values
        emb = emb[pos]
    return emb


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"设备: {dev} | 模式: {'含提交' if gen_sub else '评估'}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("[1/7] 读取数据与嵌入...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    emb_all = load_emb(ESM_TRAIN, train_df["protein_id"].values)
    emb_test = load_emb(ESM_TEST, test_df["protein_id"].values)
    emb_train, emb_val = emb_all[idx_train], emb_all[idx_val]
    print(f"  嵌入: train {emb_train.shape} / val {emb_val.shape} / test {emb_test.shape}")

    print(f"[2/7] 训练投影头 (SupCon τ=0.1 + 困难负样本 {N_POS}+{N_HARD_NEG}/锚点, "
          f"LayerNorm, jitter, {EPOCHS} epoch, lr {LR} cosine)...")
    model = ProjectionHead(PROJ_DIMS).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    Xt = torch.from_numpy(emb_train.astype(np.float32)).to(dev)
    Yt = torch.from_numpy(y_train.astype(np.float32)).to(dev)
    ytr_np = y_train.astype(np.float32)
    n_tr = len(idx_train)

    # 标签 Jaccard 矩阵 (n×n 太大, 分块预计算正对索引)
    print("  预计算标签正对索引 (Jaccard≥τ)...")
    jac_cache = None
    inter_cache = None

    def build_pos_neg_pool(anchor_ids):
        """对锚点集合构建 (正对索引, 困难负样本索引)。
        困难负样本 = 投影空间 top-HARD_TOP_K 近邻中 Jaccard<τ 的蛋白。"""
        with torch.no_grad():
            za = model(Xt[torch.from_numpy(anchor_ids.astype(np.int64)).to(dev)])
            zall = model(Xt)
        sim_all = za @ zall.T                                    # (n_a, n_tr) GPU
        top = torch.topk(sim_all, HARD_TOP_K + 1, dim=1)
        nn_idx = top.indices.cpu().numpy()[:, 1:]                # 排除自身
        pos_idx, neg_idx = [], []
        for r, a in enumerate(anchor_ids):
            nb = nn_idx[r]
            ya, yb = ytr_np[a][None, :], ytr_np[nb]
            inter = (ya * yb).sum(1)
            union = ya.sum() + yb.sum(1) - inter
            jac = inter / np.maximum(union, 1e-9)
            p = nb[jac >= JACCARD_TAU][:N_POS]
            h = nb[(jac < JACCARD_TAU) & (inter == 0)][:N_HARD_NEG]
            # 若近邻池正对不足, 补随机同标签样本; 困难负不足则补随机不同标签
            if len(p) < N_POS:
                extra = np.where(jac >= JACCARD_TAU)[0]
                if len(extra) == 0:
                    extra = np.where((ytr_np @ ytr_np[a] > 0))[0]
                if len(extra) > 0:
                    rnd = np.random.choice(extra, min(N_POS - len(p), len(extra)),
                                           replace=False)
                    p = np.concatenate([p, rnd])
            if len(p) < N_POS:  # 最终兜底: 随机样本 (保证 batch 结构一致)
                p = np.concatenate([p, np.random.choice(n_tr, N_POS - len(p),
                                                        replace=False)])
            if len(h) < N_HARD_NEG:
                rnd = np.random.choice(n_tr, N_HARD_NEG - len(h), replace=False)
                h = np.concatenate([h, rnd])
            pos_idx.append(p[:N_POS]); neg_idx.append(h[:N_HARD_NEG])
        return np.stack(pos_idx), np.stack(neg_idx)

    for ep in range(EPOCHS):
        if ep % REFRESH_EVERY == 0:
            anchors = np.random.choice(n_tr, min(2000, n_tr), replace=False)
            print(f"  刷新近邻池 (epoch {ep}, {len(anchors)} 锚点)...")
            pos_pool, neg_pool = build_pos_neg_pool(anchors)
            pool_map = {a: i for i, a in enumerate(anchors)}
        losses, pos_rates = [], []
        n_batch = len(anchors) // N_ANCHOR_PER_BATCH
        perm = np.random.permutation(n_batch)
        for bi in tqdm(range(n_batch), desc=f"epoch {ep+1}/{EPOCHS}", unit="batch",
                       leave=False):
            a_ids = anchors[perm[bi] * N_ANCHOR_PER_BATCH:(perm[bi] + 1) * N_ANCHOR_PER_BATCH]
            b_ids = []
            for a in a_ids:
                pi = pool_map[a]
                b_ids += [a] + list(pos_pool[pi]) + list(neg_pool[pi])
            b_ids = np.array(b_ids[:BATCH_SIZE], dtype=np.int64)
            xb = Xt[torch.from_numpy(b_ids)]
            if JITTER_STD > 0:
                xb = xb + torch.randn_like(xb) * JITTER_STD
            z = model(xb)
            loss, pr = supcon_loss(z, Yt[torch.from_numpy(b_ids)], TEMPERATURE, JACCARD_TAU)
            if loss.item() == 0.0:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item()); pos_rates.append(pr)
        sched.step()
        if losses and ((ep + 1) % 5 == 0 or ep == 0):
            print(f"  epoch {ep+1}: loss={np.mean(losses):.4f} 正对率={np.mean(pos_rates):.2f}")
    torch.save(model.state_dict(), f"{REL_DIR}/E002a_proj_head.pt")

    print("[3/7] 投影 + val 检索 (91,036 空间)...")
    model.eval()
    with torch.no_grad():
        proj_train = model(Xt).cpu().numpy().astype(np.float32)
        proj_val = model(torch.from_numpy(emb_val.astype(np.float32)).to(dev)).cpu().numpy()
    sim_emb_val = knn_aggregate(proj_val, proj_train, y_train, TOP_K)
    f1_emb = macro_f1_fixed(sim_emb_val, y_val)
    auc_emb = macro_auc(y_val, sim_emb_val)
    print(f"  sim_emb 独立: F1(0.5)={f1_emb:.4f} AUC={auc_emb:.4f}")

    print("[4/7] 与 kmer_sim (k6) 对比与融合...")
    d6 = np.load(NEIGH_K6_PATH)
    n = len(idx_val)
    labels = y_train[d6["idx_val"][:, :3].ravel()].reshape(n, 3, -1).astype(np.float32)
    w6 = np.maximum(d6["sim_val"][:, :3], 1e-7)
    sim_kmer_val = (labels * w6[:, :, None]).sum(axis=1) / w6.sum(axis=1)[:, None]
    f1_kmer = macro_f1_fixed(sim_kmer_val, y_val)
    auc_kmer = macro_auc(y_val, sim_kmer_val)
    corr = float(np.corrcoef(sim_emb_val.ravel(), sim_kmer_val.ravel())[0, 1])
    print(f"  kmer_sim 独立: F1(0.5)={f1_kmer:.4f} AUC={auc_kmer:.4f}")
    print(f"  corr(sim_emb, kmer_sim) = {corr:.4f}")
    fusion = {}
    for we in [0.0, 0.2, 0.5, 0.8, 1.0]:
        sim_f = we * sim_emb_val + (1 - we) * sim_kmer_val
        fusion[we] = round(macro_f1_fixed(sim_f, y_val), 6)
    print(f"  融合网格 (w_emb): {fusion}")
    best_we = max(fusion, key=fusion.get)
    print(f"  最优 w_emb={best_we} (F1={fusion[best_we]:.4f}, kmer alone={f1_kmer:.4f})")

    report = {"device": dev,
              "sim_emb_alone": {"f1_05": round(f1_emb, 6), "auc": round(auc_emb, 6)},
              "kmer_sim_alone": {"f1_05": round(f1_kmer, 6), "auc": round(auc_kmer, 6)},
              "corr_emb_kmer": corr, "fusion_w_emb": fusion, "best_w_emb": best_we,
              "hyperparams": {"proj_dims": PROJ_DIMS, "temperature": TEMPERATURE,
                              "jaccard_tau": JACCARD_TAU, "batch": BATCH_SIZE,
                              "epochs": EPOCHS, "lr": LR, "jitter_std": JITTER_STD}}

    if not gen_sub:
        np.savez_compressed(PROJ_PATH, proj_train=proj_train, proj_val=proj_val)
        report["submission"] = "none"
        with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  投影嵌入缓存: {PROJ_PATH}")
        print(f"  报告: {REPORT_PATH}")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    if fusion[best_we] <= f1_kmer + 0.001:
        report["submission"] = "none (融合无增益)"
        with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("  融合无显著增益 → 不生成提交")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    print("[5/7] 全量空间 (113,796) test 检索 + 集成...")
    with torch.no_grad():
        proj_full = model(torch.from_numpy(emb_all.astype(np.float32)).to(dev)).cpu().numpy()
        proj_test = model(torch.from_numpy(emb_test.astype(np.float32)).to(dev)).cpu().numpy()
    sim_emb_test = knn_aggregate(proj_test, proj_full, y_full, TOP_K)
    np.savez_compressed(PROJ_PATH, proj_train=proj_train, proj_val=proj_val,
                        proj_test=proj_test)

    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, cnn_test = vp["cnn"], tp["cnn"]
    t1 = np.load(T001A_PROBS_PATH)
    t001a_val, t001a_test = t1["t001a_val"], t1["t001a_test"]
    t2v = np.load(T002_VAL_PROBS_PATH)
    t2t = np.load(T002_TEST_PROBS_PATH)
    t002_val, t002_test = t2v["t002"], t2t["t002"]
    with open(TCK002_CONFIG_PATH, "r", encoding="utf-8") as f:
        w = json.load(f)["best_w"]

    # 重建 test kmer_sim (TCK002a 全量缓存)
    dfull = np.load(f"{REL_DIR}/TCK002a_neigh_full.npz")
    n2 = dfull["idx_test"].shape[0]
    lab = y_full[dfull["idx_test"][:, :3].ravel()].reshape(n2, 3, -1).astype(np.float32)
    ww = np.maximum(dfull["sim_test"][:, :3], 1e-7)
    sim_kmer_test = (lab * ww[:, :, None]).sum(axis=1) / ww.sum(axis=1)[:, None]
    sim_test = best_we * sim_emb_test + (1 - best_we) * sim_kmer_test
    sim_val = best_we * sim_emb_val + (1 - best_we) * sim_kmer_val

    ens_val = (w["t001a"] * t001a_val + w["t002"] * t002_val
               + w["cnn"] * cnn_val + 0.7 * sim_val)
    ens_test = (w["t001a"] * t001a_test + w["t002"] * t002_test
                + w["cnn"] * cnn_test + 0.7 * sim_test)

    print("[6/7] 100 档阈值 + 提交...")
    cands = np.linspace(0.01, 0.99, 100)
    best_t = np.full(500, 0.5)
    for i in tqdm(range(500), desc="    阈值搜索", unit="label"):
        yt = y_val[:, i]
        if yt.sum() == 0:
            continue
        p = ens_val[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (p & ytm).sum(0); fp = (p & ~ytm).sum(0); fn = (~p & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_t[i] = cands[f1t.argmax()]
    y_pred = np.zeros_like(ens_test, dtype=np.int8)
    for i, t in enumerate(best_t):
        y_pred[:, i] = (ens_test[:, i] >= t).astype(np.int8)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    report["submission"] = SUB_PATH
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")

    print("[7/7] 报告...")
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
