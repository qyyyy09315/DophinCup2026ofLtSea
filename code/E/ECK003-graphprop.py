"""
ECK003-graphprop — 二跳邻居图传播（方向 B）
=============================================
背景:
  - ECK001 三路融合 = 0.4558（新最佳）；延用其 sim 融合结构
  - 网络传播思想 (NetGO3.0 10.1007/978-1-0716-4662-5_7、DeepGraphGO
    10.1093/bioinformatics/btab270): 一跳只聚合直接邻居, 二跳传播"邻居的
    邻居"标签, 捕获同源网络的更广结构
方案:
  - 一跳聚合: sim_1hop (已有, val→train 的 kmer/align/pt5 融合值)
  - 二跳传播: 对每个蛋白, 取其邻居(已聚合)的 again 邻居做加权传播
      sim_2hop = A · sim_1hop   (A = 归一化邻居加权矩阵, train→train)
  - 构建 train→train 邻居: 复用 HashingVectorizer kmer 检索 (同 CK002),
    耗时 ~分钟级; 缓存 rel/ECK003_neigh_train{k}.npz
  - 融合: val 上 sim_final = (1-λ)·sim_base + λ·sim_2hop, 进 ECK001 四路集成
  - 网格: λ ∈ {0.2,0.4,0.6,0.8}, k ∈ {6}
  - 评估: 100 档阈值 val Macro F1 + R/P/AUC, 对比 ECK001 基线 0.4558
用法:
  python code/E/ECK003-graphprop.py [--gen-sub]
输出:
  rel/ECK003_report.json
  submit/submission_ECK003.csv (若有效)
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from scipy import sparse
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")

MAX_TOP_K = 5
N_FEATURES = 2 ** 20
BLOCK = 2048

def macro_rp(ens, y, t):
    pred = (ens >= t)
    rs, ps = [], []
    for i in range(500):
        yt = y[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
        rs.append(tp / (tp + fn) if tp + fn else 0.0)
        ps.append(tp / (tp + fp) if tp + fp else 0.0)
    return float(np.mean(rs)), float(np.mean(ps))


def macro_auc(y, ens):
    aucs = []
    for i in range(500):
        yt = y[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, ens[:, i]))
    return float(np.mean(aucs))


def train_neighbors_k(seqs, seqs_q, k, cache):
    """HashingVectorizer kmer → train 空间 top-5 邻居 (缓存)。"""
    if os.path.exists(cache):
        d = np.load(cache)
        return d["idx"], d["sim"]
    vec = HashingVectorizer(analyzer="char", ngram_range=(k, k),
                            n_features=N_FEATURES, dtype=np.float32)
    X_tr = normalize(sparse.csr_matrix(vec.transform(seqs)))
    X_q = normalize(sparse.csr_matrix(vec.transform(seqs_q)))
    n_q = X_q.shape[0]
    idx = np.zeros((n_q, MAX_TOP_K), dtype=np.int64)
    sims = np.zeros((n_q, MAX_TOP_K), dtype=np.float32)
    for s in tqdm(range(0, n_q, BLOCK), desc=f"train 邻居 k={k}"):
        e = min(s + BLOCK, n_q)
        sim = (X_q[s:e] @ X_tr.T).toarray()
        part = np.argpartition(-sim, MAX_TOP_K, axis=1)[:, :MAX_TOP_K]
        idx[s:e] = part
        sims[s:e] = np.take_along_axis(sim, part, axis=1)
    np.savez_compressed(cache, idx=idx, sim=sims)
    return idx, sims


def build_adj(neigh_idx, neigh_sim, n_nodes):
    """归一化邻接矩阵 A (n_nodes x n_nodes): 每行 top-5 邻居 sim 加权, 行归一。"""
    rows = np.repeat(np.arange(neigh_idx.shape[0]), MAX_TOP_K)
    cols = neigh_idx.ravel()
    val = np.maximum(neigh_sim.ravel(), 1e-7)
    A = sparse.csr_matrix((val, (rows, cols)), shape=(n_nodes, n_nodes))
    A = normalize(A, norm="l1", axis=1)  # 行归一 → 传播概率
    return A


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print(f"ECK003 方向B: 二跳邻居图传播 | 模式: {'含提交' if gen_sub else '评估'}")

    print("[1/6] 数据...")
    train_df = pd.read_csv(f"{PROJECT_ROOT}/data/train.csv")
    test_df = pd.read_csv(f"{PROJECT_ROOT}/data/test.csv")
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=0.2, random_state=42)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    seqs = train_df["sequence"].values
    print(f"  train {len(idx_train)} / val {len(idx_val)} / test {len(test_df)}")

    # ----- 复用 ECK001 的一跳 sim (val) -----
    print("[2/6] 构建一跳 sim (ECK001 三路融合 val)...")
    # 文件名为 ECK001-fusion.py (含连字符不可直接 import), 用 importlib 按路径加载
    import importlib.util
    mod_path = os.path.join(PROJECT_ROOT, "code", "E", "ECK001-fusion.py")
    spec = importlib.util.spec_from_file_location("eck001", mod_path)
    eck001 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eck001)
    parse_aln = eck001.parse_aln
    agg_labels = eck001.agg_labels
    knn_agg_cos = eck001.knn_agg_cos
    build_kmer = eck001.build_kmer
    ALIGN_W = eck001.ALIGN_W
    TOP_K = eck001.TOP_K
    SIM_W = eck001.SIM_W

    lens = np.array([len(s) for s in seqs])
    val_short = lens[idx_val] <= 256
    d = np.load(f"{REL_DIR}/E003_pt5_emb_short.npz")
    emb_s_tr = d["emb_s_tr"]
    short_tr = np.where(lens <= 256)[0]
    pos = np.searchsorted(short_tr, idx_val[val_short])
    sim_pt5_s = knn_agg_cos(emb_s_tr[pos], emb_s_tr, y_full[short_tr], 3)
    sim_kmer = build_kmer(idx_train, idx_val, y_train, "val")
    val_ids = train_df["protein_id"].values[idx_val]
    tdmap = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    idx_v, bits_v, _ = parse_aln(f"{REL_DIR}/ck005_tmp/align_val.tsv", val_ids, tdmap)
    sim_bit = agg_labels(idx_v, bits_v, y_train, 3)
    sim_base = ALIGN_W * sim_bit + (1 - ALIGN_W) * sim_kmer
    sim_base_f = sim_base.copy()
    sim_base_f[val_short] = 0.6 * sim_pt5_s + 0.4 * sim_base[val_short]
    print(f"  一跳 sim_base 全局 F1(0.5)={np.mean([f1_score(y_val[:,i].astype(bool), sim_base_f[:,i]>=0.5, zero_division=0) for i in range(500) if y_val[:,i].sum()>0]):.4f} (ECK001 sim路)")

    # ----- 二跳: train→train 邻居 + 传播 -----
    print("[3/6] train→train kmer 邻居 (k=6, 91,036 空间)...")
    k = 6
    # 只在 idx_train(91,036) 子集内算邻居, 索引对齐 y_train(0..91035)
    seqs_tt = seqs[idx_train]
    tr_idx, tr_sim = train_neighbors_k(seqs_tt, seqs_tt, k,
                                       f"{REL_DIR}/ECK003_neigh_train_k{k}.npz")
    # tr_idx 已是 0..91035 (train 行), 直接用于 y_train 索引

    # 二跳传播: val 的一跳邻居(train 行) → 这些邻居的邻居(仍在 train 空间)
    print("[4/6] 二跳聚合 (邻居的邻居 kmer 标签)...")
    dd = np.load(f"{REL_DIR}/CK002_neigh_k6.npz")
    neigh1 = dd["idx_val"]  # (22760,5) val→train(91,036 行)
    n_val = len(idx_val)
    two_hop_idx = tr_idx[neigh1]  # (n_val,5,5) 均为 train 行
    two_hop_sim = tr_sim[neigh1]
    # 聚合二跳邻居标签 kmer
    n_top = 3
    idx2 = two_hop_idx[:, :, :n_top].reshape(n_val, -1)  # (n_val, 15)
    w2 = (two_hop_sim[:, :, :n_top] ** 2).reshape(n_val, -1)
    lab2 = y_train[idx2.ravel()].reshape(n_val, -1, 500).astype(np.float32)
    sim_2hop = (lab2 * w2[:, :, None]).sum(axis=1) / (w2.sum(axis=1)[:, None] + 1e-7)
    f1_2hop = np.mean([f1_score(y_val[:, i].astype(bool), sim_2hop[:, i] >= 0.5, zero_division=0)
                       for i in range(500) if y_val[:, i].sum() > 0])
    corr2 = float(np.corrcoef(sim_2hop.ravel(), sim_kmer.ravel())[0, 1])
    print(f"  二跳 sim 独立 F1(0.5)={f1_2hop:.4f} | corr(2hop, kmer1hop)={corr2:.4f}")

    print("[5/6] 融合 λ 网格 (val)...")
    results = {}
    for lam in [0.2, 0.4, 0.6, 0.8]:
        sim_f = (1 - lam) * sim_base_f + lam * sim_2hop
        # 四路集成
        vp = np.load(f"{REL_DIR}/TC003_val_probs.npz"); cnn = vp["cnn"]
        t1 = np.load(f"{REL_DIR}/TCK001_t001a_probs.npz"); t1a = t1["t001a_val"]
        t2v = np.load(f"{REL_DIR}/T002_val_probs.npz"); t2 = t2v["t002"]
        w = json.load(open(f"{PROJECT_ROOT}/models/tck002-ensemble.json"))["best_w"]
        ens = w["t001a"] * t1a + w["t002"] * t2 + w["cnn"] * cnn + 0.75 * sim_f
        # 100 档阈值
        cands = np.linspace(0.01, 0.99, 100)
        bt = np.full(500, 0.5)
        for i in range(500):
            yt = y_val[:, i]
            if yt.sum() == 0:
                continue
            p = ens[:, i][:, None] >= cands[None, :]
            ytm = yt[:, None].astype(bool)
            tp = (p & ytm).sum(0); fp = (p & ~ytm).sum(0); fn = (~p & ytm).sum(0)
            prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
            f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
            bt[i] = cands[f1t.argmax()]
        r, p = macro_rp(ens, y_val, bt)
        auc = macro_auc(y_val, ens)
        f1 = np.mean([f1_score(y_val[:, i].astype(bool), ens[:, i] >= bt[i], zero_division=0)
                      for i in range(500) if y_val[:, i].sum() > 0])
        results[f"{lam}"] = {"f1": round(f1, 4), "r": round(r, 4), "p": round(p, 4),
                             "auc": round(auc, 4)}
    print(f"  λ 网格: {json.dumps(results, ensure_ascii=False)}")
    best_lam = max(results, key=lambda x: results[x]["f1"])
    f1_base = np.mean([f1_score(y_val[:, i].astype(bool), sim_base_f[:, i] >= 0.5, zero_division=0)
                       for i in range(500) if y_val[:, i].sum() > 0])  # 占位(下方用集成基线)
    # 重新精确算集成基线 (λ=0)
    vp = np.load(f"{REL_DIR}/TC003_val_probs.npz"); cnn = vp["cnn"]
    t1 = np.load(f"{REL_DIR}/TCK001_t001a_probs.npz"); t1a = t1["t001a_val"]
    t2v = np.load(f"{REL_DIR}/T002_val_probs.npz"); t2 = t2v["t002"]
    w = json.load(open(f"{PROJECT_ROOT}/models/tck002-ensemble.json"))["best_w"]
    ens_base = w["t001a"] * t1a + w["t002"] * t2 + w["cnn"] * cnn + 0.75 * sim_base_f
    cands = np.linspace(0.01, 0.99, 100)
    bt0 = np.full(500, 0.5)
    for i in range(500):
        yt = y_val[:, i]
        if yt.sum() == 0:
            continue
        p = ens_base[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (p & ytm).sum(0); fp = (p & ~ytm).sum(0); fn = (~p & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        bt0[i] = cands[f1t.argmax()]
    base_f1 = np.mean([f1_score(y_val[:, i].astype(bool), ens_base[:, i] >= bt0[i], zero_division=0)
                       for i in range(500) if y_val[:, i].sum() > 0])
    best_f1 = results[best_lam]["f1"]
    gain = best_f1 - base_f1
    print(f"  基线(集成 λ=0) F1={base_f1:.4f} | 最优 λ={best_lam} F1={best_f1:.4f} Δ={gain:+.4f}")

    report = {"sim_2hop_f1_05": round(f1_2hop, 4), "corr_2hop_1hop": round(corr2, 4),
              "lambda_grid": results, "best_lambda": best_lam,
              "base_ensemble_f1": round(base_f1, 4), "best_ensemble_f1": round(best_f1, 4),
              "gain": round(gain, 4)}

    if not gen_sub or gain <= 0.001:
        if gain <= 0.001:
            report["submission"] = "none (无增益)"
        with Path(f"{REL_DIR}/ECK003_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  报告: {REL_DIR}/ECK003_report.json")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    print("[6/6] test 端二跳 + 提交...")
    print("  (未实现 test 端, 需全量 train 邻居传播)")
    report["submission"] = "test 端待补"
    with Path(f"{REL_DIR}/ECK003_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
