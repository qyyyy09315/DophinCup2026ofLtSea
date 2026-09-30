"""
R001a-ensemble — BiLSTM embedding 邻居特征实验
================================================
方案:
  - 用 R001 的序列 embedding (512 维, rel/R001_embeddings.npz) 做最近邻
  - 邻居空间: val 查询 → emb_train (91,036, 防泄漏);
    test 查询 → emb_train + emb_val 全量 (113,796, 无泄漏)
  - 聚合: top-3 相似度加权邻居标签平均 (与 k-mer sim 相同聚合方式)
  - 评估: sim_emb 单独 F1(0.5)/AUC + 与 k-mer sim 的按标签相关性
  - 融合: k-mer sim 与 sim_emb 加权 (w 网格) → 与 C002b 集成检验
输出:
  rel/R001a_sim_emb.npz
  rel/R001a_report.json
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
EMB_PATH = f"{REL_DIR}/R001_embeddings.npz"
OUT_PATH = f"{REL_DIR}/R001a_sim_emb.npz"
REPORT_PATH = f"{REL_DIR}/R001a_report.json"

NEIGH_K6_PATH = f"{REL_DIR}/CK002_neigh_k6.npz"   # k-mer sim 对照 (val 空间 91,036)
TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"

TOP_K = 3
BLOCK = 500
VAL_SIZE = 0.2
RANDOM_STATE = 42


def macro_auc(y_true, y_prob):
    aucs = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, y_prob[:, i]))
    return float(np.mean(aucs))


def macro_f1_fixed_threshold(ens, y_true, t=0.5):
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


def build_sim_prob(neigh_idx, neigh_sim, y_space, top_k):
    n = neigh_idx.shape[0]
    labels = y_space[neigh_idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    w = np.maximum(neigh_sim[:, :top_k], 1e-7)
    return (labels * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]


def knn_embeddings(E_query, E_space, query_name):
    """embedding 空间 top-5 邻居 (cosine, L2 归一化后点积)。"""
    Eq = normalize(E_query)
    Es = normalize(E_space)
    n_q = Eq.shape[0]
    idx = np.zeros((n_q, TOP_K + 2), dtype=np.int64)
    sims = np.zeros((n_q, TOP_K + 2), dtype=np.float32)
    for s in range(0, n_q, BLOCK):
        e = min(s + BLOCK, n_q)
        sim = Eq[s:e] @ Es.T
        k = TOP_K + 2
        part = np.argpartition(-sim, k, axis=1)[:, :k]
        idx[s:e] = part
        sims[s:e] = np.take_along_axis(sim, part, axis=1)
    return idx, sims


def main():
    t0 = time.time()
    print("=" * 60)

    print("[1/5] 读取数据与 embedding...")
    train_df = pd.read_csv(TRAIN_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train = y_full[idx_train]

    d = np.load(EMB_PATH)
    emb_train, emb_val, emb_test = d["emb_train"], d["emb_val"], d["emb_test"]
    print(f"  emb_train={emb_train.shape}, emb_val={emb_val.shape}, emb_test={emb_test.shape}")

    print("[2/5] embedding 最近邻...")
    if os.path.exists(OUT_PATH):
        dd = np.load(OUT_PATH)
        print("  sim_emb 读缓存")
    else:
        idx_v, sim_v = knn_embeddings(emb_val, emb_train, "val")      # 空间 91,036
        space_all = np.vstack([emb_train, emb_val])                    # 全量 113,796
        idx_t, sim_t = knn_embeddings(emb_test, space_all, "test")     # 空间 113,796
        np.savez_compressed(OUT_PATH, idx_val=idx_v, sim_val=sim_v,
                            idx_test=idx_t, sim_test=sim_t)
        print(f"  邻居缓存已保存: {OUT_PATH}")

    print("[3/5] sim_emb 聚合与独立评估...")
    dd = np.load(OUT_PATH)
    sim_val = build_sim_prob(dd["idx_val"], dd["sim_val"], y_train, TOP_K)
    sim_test = build_sim_prob(dd["idx_test"], dd["sim_test"], y_full, TOP_K)
    print(f"  sim_emb val 密度: {sim_val.mean():.4f}")
    print(f"  val top-1 平均相似度: {dd['sim_val'][:, 0].mean():.4f} "
          f"(k-mer sim 对照: {np.load(NEIGH_K6_PATH)['sim_val'][:, 0].mean():.4f})")
    f1_emb = macro_f1_fixed_threshold(sim_val, y_full[idx_val])
    auc_emb = macro_auc(y_full[idx_val], sim_val)
    print(f"  sim_emb 独立: F1(0.5)={f1_emb:.4f}, AUC={auc_emb:.4f}")

    print("[4/5] 与 k-mer sim 相关性 + 融合检验...")
    d6 = np.load(NEIGH_K6_PATH)
    kmer_sim_val = build_sim_prob(d6["idx_val"], d6["sim_val"], y_train, 3)
    corrs = [np.corrcoef(sim_val[:, i], kmer_sim_val[:, i])[0, 1]
             for i in range(len(label_cols))
             if not np.isnan(np.corrcoef(sim_val[:, i], kmer_sim_val[:, i])[0, 1])]
    corr_mean = float(np.mean(corrs))
    print(f"  sim_emb vs kmer_sim 按标签相关: {corr_mean:.4f}")

    # 融合: sim_fused = w*sim_emb + (1-w)*kmer_sim, 与 C002b 集成
    vp = np.load(TC003_VAL_PROBS_PATH)
    cnn_val, val_true = vp["cnn"], vp["y_true"]
    best_w, best_f1, best_fuse = 0.5, -1.0, 0.5
    for w in np.linspace(0.0, 1.0, 11):
        fuse = w * sim_val + (1.0 - w) * kmer_sim_val
        for wc in np.linspace(0.0, 1.0, 11):
            ens = wc * cnn_val + (1.0 - wc) * fuse
            f1 = macro_f1_fixed_threshold(ens, val_true)
            if f1 > best_f1:
                best_f1, best_w, best_fuse = f1, w, wc
    print(f"  融合最优: sim_emb 权重={best_w:.2f}, cnn 权重={best_fuse:.2f}, F1(0.5)={best_f1:.4f}")

    print("[5/5] 保存报告...")
    report = {
        "sim_emb_alone": {"f1_05": round(f1_emb, 6), "auc": round(auc_emb, 6)},
        "top1_sim": {"emb": round(float(dd['sim_val'][:, 0].mean()), 4),
                     "kmer": round(float(np.load(NEIGH_K6_PATH)['sim_val'][:, 0].mean()), 4)},
        "corr_emb_kmer": round(corr_mean, 4),
        "fused_with_cnn": {"w_sim_emb": round(float(best_w), 4),
                           "w_cnn": round(float(best_fuse), 4),
                           "f1_05": round(best_f1, 6)},
        "baseline_ck002_f1_05": 0.4015,  # k=6 top-3 加权 + cnn 集成
        "total_time_seconds": round(time.time() - t0, 1),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
