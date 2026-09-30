"""
E001-pretrained-neighbor — 预训练嵌入邻居特征实验 (ESM2 / ProtBERT)
====================================================================
方案:
  - 用预训练蛋白语言模型的序列嵌入做最近邻邻居特征 (TALE 2021 /
    E-SNPs&GO 2022 思路: 语义空间邻居)
  - ESM2 150M mean-pooled 640 维 / ProtBERT CLS 1024 维 (均已缓存 parquet)
  - 邻居空间: val 查询 → 训练部分 (91,036, 防泄漏);
    test 查询 → 全量 (113,796, 无泄漏)
  - 聚合: top-3 相似度加权邻居标签平均 (与 k-mer sim 同聚合)
  - 评估: 单独 F1(0.5)/AUC + 与 k-mer sim 相关性 + 与 C002b 集成 (w 网格)
输出:
  rel/E001_report.json
  rel/E001_neigh_esm.npz / rel/E001_neigh_ptb.npz
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
ESM_TRAIN = f"{REL_DIR}/train_esm2_embeddings.parquet"
ESM_TEST = f"{REL_DIR}/test_esm2_embeddings.parquet"
PTB_TRAIN = f"{REL_DIR}/train_embeddings.parquet"
PTB_TEST = f"{REL_DIR}/test_embeddings.parquet"
REPORT_PATH = f"{REL_DIR}/E001_report.json"

NEIGH_K6_PATH = f"{REL_DIR}/CK002_neigh_k6.npz"
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


def knn_embeddings(E_query, E_space, name):
    Eq = E_query / (np.linalg.norm(E_query, axis=1, keepdims=True) + 1e-9)
    Es = E_space / (np.linalg.norm(E_space, axis=1, keepdims=True) + 1e-9)
    n_q = Eq.shape[0]
    k = TOP_K + 2
    idx = np.zeros((n_q, k), dtype=np.int64)
    sims = np.zeros((n_q, k), dtype=np.float32)
    for s in range(0, n_q, BLOCK):
        e = min(s + BLOCK, n_q)
        sim = Eq[s:e] @ Es.T
        part = np.argpartition(-sim, k, axis=1)[:, :k]
        idx[s:e] = part
        sims[s:e] = np.take_along_axis(sim, part, axis=1)
    return idx, sims


def load_emb(parquet_path, prot_ids):
    """按 protein_id 对齐加载嵌入矩阵。"""
    df = pd.read_parquet(parquet_path)
    df = df.set_index("protein_id").loc[prot_ids]
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    return df[emb_cols].values.astype(np.float32)


def evaluate_one(name, emb_train_all, emb_test, y_full, idx_train, idx_val,
                 y_val, kmer_sim_val, cnn_val, cache_pref):
    print(f"\n===== {name} =====")
    cache = f"{REL_DIR}/{cache_pref}.npz"
    if os.path.exists(cache):
        dd = np.load(cache)
        print("  邻居缓存读取")
    else:
        # val 查询 → 训练部分空间; test 查询 → 全量空间
        idx_v, sim_v = knn_embeddings(emb_train_all[idx_val], emb_train_all[idx_train], "val")
        idx_t, sim_t = knn_embeddings(emb_test, emb_train_all, "test")
        np.savez_compressed(cache, idx_val=idx_v, sim_val=sim_v,
                            idx_test=idx_t, sim_test=sim_t)
        print(f"  邻居缓存已保存: {cache}")
    dd = np.load(cache)
    sim_val = build_sim_prob(dd["idx_val"], dd["sim_val"], y_full[idx_train], TOP_K)
    sim_test = build_sim_prob(dd["idx_test"], dd["sim_test"], y_full, TOP_K)
    print(f"  top-1 平均相似度: {dd['sim_val'][:, 0].mean():.4f}")
    f1_alone = macro_f1_fixed_threshold(sim_val, y_val)
    auc_alone = macro_auc(y_val, sim_val)
    print(f"  单独: F1(0.5)={f1_alone:.4f}, AUC={auc_alone:.4f}")

    corrs = [np.corrcoef(sim_val[:, i], kmer_sim_val[:, i])[0, 1]
             for i in range(y_val.shape[1])
             if not np.isnan(np.corrcoef(sim_val[:, i], kmer_sim_val[:, i])[0, 1])]
    corr_mean = float(np.mean(corrs))
    print(f"  与 kmer_sim 相关: {corr_mean:.4f}")

    best_w, best_f1, best_wc = 0.5, -1.0, 0.5
    for w in np.linspace(0.0, 1.0, 11):
        fuse = w * sim_val + (1.0 - w) * kmer_sim_val
        for wc in np.linspace(0.0, 1.0, 11):
            ens = wc * cnn_val + (1.0 - wc) * fuse
            f1 = macro_f1_fixed_threshold(ens, y_val)
            if f1 > best_f1:
                best_f1, best_w, best_wc = f1, w, wc
    print(f"  与 kmer_sim+cnn 融合最优: w_sim={best_w:.2f}, w_cnn={best_wc:.2f}, F1(0.5)={best_f1:.4f}")
    return {"name": name, "f1_alone": round(f1_alone, 6), "auc_alone": round(auc_alone, 6),
            "corr_kmer": round(corr_mean, 4),
            "fused": {"w_sim": round(float(best_w), 4), "w_cnn": round(float(best_wc), 4),
                      "f1_05": round(best_f1, 6)},
            "sim_test_shape": list(sim_test.shape)}


def main():
    t0 = time.time()
    print("=" * 60)
    print("[1/5] 读取数据与索引...")
    train_df = pd.read_csv(TRAIN_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_val = y_full[idx_val]
    prot_ids = train_df["protein_id"].values

    vp = np.load(TC003_VAL_PROBS_PATH)
    cnn_val, val_true = vp["cnn"], vp["y_true"]
    d6 = np.load(NEIGH_K6_PATH)
    kmer_sim_val = build_sim_prob(d6["idx_val"], d6["sim_val"], y_full[idx_train], 3)
    print(f"  标签 {len(label_cols)}, val {len(idx_val)}")

    print("[2/5] 加载 ESM2 嵌入...")
    esm_all = load_emb(ESM_TRAIN, prot_ids)
    esm_test = load_emb(ESM_TEST, pd.read_parquet(ESM_TEST)["protein_id"].values)
    print(f"  esm_train_all {esm_all.shape}, esm_test {esm_test.shape}")
    res_esm = evaluate_one("ESM2-150M", esm_all, esm_test, y_full, idx_train, idx_val,
                           y_val, kmer_sim_val, cnn_val, "E001_neigh_esm")

    print("\n[3/5] 加载 ProtBERT 嵌入...")
    ptb_all = load_emb(PTB_TRAIN, prot_ids)
    ptb_test = load_emb(PTB_TEST, pd.read_parquet(PTB_TEST)["protein_id"].values)
    print(f"  ptb_train_all {ptb_all.shape}, ptb_test {ptb_test.shape}")
    res_ptb = evaluate_one("ProtBERT", ptb_all, ptb_test, y_full, idx_train, idx_val,
                           y_val, kmer_sim_val, cnn_val, "E001_neigh_ptb")

    print("\n[4/5] 保存报告...")
    report = {
        "baseline": {"kmer_sim_alone_f1_05": round(macro_f1_fixed_threshold(kmer_sim_val, y_val), 6),
                     "ck002_f1_05": 0.4015},
        "results": [res_esm, res_ptb],
        "total_time_seconds": round(time.time() - t0, 1),
    }
    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
