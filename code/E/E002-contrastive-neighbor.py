"""
E002-contrastive-neighbor — 监督对比学习嵌入 → 功能邻居特征
================================================================
问题:
  - E001 已证: 原始 mean-pooled/CLS 嵌入退化 (余弦 0.99+ 无功能区分度),
    朴素嵌入邻居无效 → 需功能感知的对比投影
依据:
  - FuncSeek (Cloete & Patterton, bioRxiv 2026.08.07, doi:10.64898/2026.08.07.738656):
    多 PLM 拼接 + 监督对比头 → 256-D 功能感知空间 → 最近邻检索大幅改善
    (OOD<30% 同源性 EC4 64.6% vs CLEAN 55.1% / BLASTp 51.4%)
  - ESM2: Lin et al., Science 2023, doi:10.1126/science.ade2574
方案:
  - 嵌入: ESM2-150M mean-pooled 640-D (缓存 parquet, 免重新推理)
  - 投影头: MLP(640→512→256) + L2 归一化
  - 训练: batch 内标签 Jaccard ≥ τ 为正对, InfoNCE 对比损失 (温度 T)
  - 检索: 投影后嵌入 top-3 相似度加权聚合 sim_emb (val 91,036 空间, 防泄漏)
  - 评估: sim_emb 独立 F1/AUC、与 kmer_sim 相关性、融合网格
  - 若融合有效: 全量空间 (113,796) test 检索 → 双 sim (kmer+emb) 进 TCK 框架 → 提交
输出:
  rel/E002_report.json
  rel/E002_emb_proj.npz (投影后嵌入, 免重复训练)
  submit/submission_E002_ensemble.csv (若有效)
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
REPORT_PATH = f"{REL_DIR}/E002_report.json"
SUB_PATH = f"{SUB_DIR}/submission_E002_ensemble.csv"
PROJ_PATH = f"{REL_DIR}/E002_emb_proj.npz"

ESM_TRAIN = f"{REL_DIR}/train_esm2_embeddings.parquet"
ESM_TEST = f"{REL_DIR}/test_esm2_embeddings.parquet"
NEIGH_K6_PATH = f"{REL_DIR}/CK002_neigh_k6.npz"
TCK002A_FULL_PATH = f"{REL_DIR}/TCK002a_neigh_full.npz"

TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
T001A_PROBS_PATH = f"{REL_DIR}/TCK001_t001a_probs.npz"
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"
TCK002_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

# 对比学习超参
PROJ_DIMS = [640, 512, 256]
TEMPERATURE = 0.5      # 0.1 在 ESM2 高相似嵌入下 exp 饱和, 0.5 更稳
JACCARD_TAU = 0.3
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3
SEED = 42

TOP_K = 3
VAL_SIZE = 0.2
RANDOM_STATE = 42


class ProjectionHead(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.nn.functional.normalize(self.net(x), dim=1)


def infonce_loss(z, y_batch, tau, jac_tau):
    """batch 内标签 Jaccard ≥ τ 为正对的 InfoNCE。
    注意: 只对正对位置计算 log, 非正对位置必须排除 (否则 log(clamp(1e-9)) 噪声项
    会淹没正对梯度, 导致 loss 恒定不降)。"""
    n = z.shape[0]
    sim = z @ z.T / tau                       # (n, n)
    sim.fill_diagonal_(-1e9)
    # Jaccard: |A∩B| / |A∪B|
    inter = y_batch @ y_batch.T
    sums = y_batch.sum(1, keepdim=True)
    union = sums + sums.T - inter
    jac = inter / union.clamp(min=1e-9)
    pos = (jac >= jac_tau) & (inter > 0)
    if pos.sum() == 0:
        return torch.tensor(0.0, device=z.device), 0.0
    denom = torch.exp(sim).sum(1, keepdim=True) + 1e-9
    pos_idx = pos.nonzero()                   # (n_pos, 2): 每正对的 (i, j)
    pos_exp = torch.exp(sim)[pos_idx[:, 0], pos_idx[:, 1]]
    denom_pos = denom[pos_idx[:, 0], 0]
    ratio = (pos_exp / denom_pos).clamp(min=1e-9)
    loss = -ratio.log().sum() / pos.sum()
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
    """GPU 检索 top-k + 相似度加权聚合邻居标签。"""
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
        sim = q[s:e] @ db.T                                    # (block, n_db)
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

    print("[2/7] 训练对比投影头 (InfoNCE, Jaccard≥0.3 正对)...")
    model = ProjectionHead(PROJ_DIMS).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    Xt = torch.from_numpy(emb_train.astype(np.float32)).to(dev)
    Yt = torch.from_numpy(y_train.astype(np.float32)).to(dev)
    n_tr = len(idx_train)
    for ep in range(EPOCHS):
        perm = torch.randperm(n_tr)
        losses, pos_rates = [], []
        for s in tqdm(range(0, n_tr, BATCH_SIZE), desc=f"epoch {ep+1}/{EPOCHS}",
                      unit="batch", leave=False):
            b = perm[s:s + BATCH_SIZE]
            z = model(Xt[b])
            loss, pr = infonce_loss(z, Yt[b], TEMPERATURE, JACCARD_TAU)
            if loss.item() == 0.0:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item()); pos_rates.append(pr)
        sched.step()
        if losses:
            print(f"  epoch {ep+1}: loss={np.mean(losses):.4f} 正对率={np.mean(pos_rates):.3f}")
    torch.save(model.state_dict(), f"{REL_DIR}/E002_proj_head.pt")

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

    report = {"device": dev, "sim_emb_alone": {"f1_05": round(f1_emb, 6), "auc": round(auc_emb, 6)},
              "kmer_sim_alone": {"f1_05": round(f1_kmer, 6), "auc": round(auc_kmer, 6)},
              "corr_emb_kmer": corr, "fusion_w_emb": fusion, "best_w_emb": best_we,
              "hyperparams": {"proj_dims": PROJ_DIMS, "temperature": TEMPERATURE,
                              "jaccard_tau": JACCARD_TAU, "batch": BATCH_SIZE,
                              "epochs": EPOCHS, "lr": LR}}

    if not gen_sub:
        np.savez_compressed(PROJ_PATH, proj_train=proj_train, proj_val=proj_val)
        report["submission"] = "none"
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  投影嵌入缓存: {PROJ_PATH}")
        print(f"  报告: {REPORT_PATH}")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    if fusion[best_we] <= f1_kmer + 0.001:
        report["submission"] = "none (融合无增益)"
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("  融合无显著增益 → 不生成提交")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    print("[5/7] 全量空间 (113,796) test 检索...")
    with torch.no_grad():
        proj_full = model(torch.from_numpy(emb_all.astype(np.float32)).to(dev)).cpu().numpy()
        proj_test = model(torch.from_numpy(emb_test.astype(np.float32)).to(dev)).cpu().numpy()
    sim_emb_test = knn_aggregate(proj_test, proj_full, y_full, TOP_K)
    np.savez_compressed(PROJ_PATH, proj_train=proj_train, proj_val=proj_val,
                        proj_test=proj_test)

    print("[6/7] 构建集成 (双 sim + TCK 四路)...")
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

    d_full = np.load(TCK002A_FULL_PATH)
    sim_kmer_test = knn_aggregate_probs(d_full, y_full)
    print("  (test 端 kmer_sim 复用 TCK002a 全量缓存)")

    sim_test = best_we * sim_emb_test + (1 - best_we) * sim_kmer_test
    sim_val = best_we * sim_emb_val + (1 - best_we) * sim_kmer_val
    ens_val = (w["t001a"] * t001a_val + w["t002"] * t002_val
               + w["cnn"] * cnn_val + 0.7 * sim_val)
    ens_test = (w["t001a"] * t001a_test + w["t002"] * t002_test
                + w["cnn"] * cnn_test + 0.7 * sim_test)

    print("[7/7] 100 档阈值 + 提交...")
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
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


def knn_aggregate_probs(d, y_space):
    n2 = d["idx_test"].shape[0]
    labels = y_space[d["idx_test"][:, :3].ravel()].reshape(n2, 3, -1).astype(np.float32)
    ww = np.maximum(d["sim_test"][:, :3], 1e-7)
    return (labels * ww[:, :, None]).sum(axis=1) / ww.sum(axis=1)[:, None]


if __name__ == "__main__":
    main()
