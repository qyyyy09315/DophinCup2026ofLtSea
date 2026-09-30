"""
ECK002-labelcorr — 标签相关性后处理（稀疏条件依赖增强）
=========================================================
编号: ECK = Embedding + CK（融合链上的后处理），ECK002（原 E003b）。
背景:
  - ECK001 三路融合 = 0.4558（新最佳）
  - TCK002b 标签共现已失败: p' = (1-α)p + α(p@P) 一次传播 → F1 0.4136→0.3422
    (共现 58.7% 非零、P 过密 → 抹平概率)
  - 本实验差异: 不在概率层传播, 而在**决策后的标签级重排**, 且用**稀疏强关联**:
    · 全量阈值决策后, 对每个蛋白取被判阳性的标签集合 S
    · 用条件概率 P(j|i∈S) 的 top-K 强关联, 把集合 S 之外的“高度伴随标签”补入
    · (分层思想: 强伴随 ≈ 功能层级中的子标签/关联标签, 参考 Barutcuoglu 2006
      层级约束、Cerri 2016 层级多标签)
方案:
  - 输入: ECK001 四路集成概率 ens (复用 ECK001-fusion 构建逻辑)
  - C[j][i] = P(i|j) 从训练集 (91,036) 算, 行归一化, 只保留 top-K 强关联
  - 后处理: pred' = pred ∪ { j∉S : max_{i∈S} P(j|i) > τ }
  - 网格: K ∈ {3,5,10}, τ ∈ {0.3,0.5,0.7}
  - 评估: 100 档阈值 (同 ECK001) 下 val Macro F1, 对比 ECK001 基线 0.4558
用法:
  python code/E/ECK002-labelcorr.py [--gen-sub]
输出:
  rel/E003b_report.json
  submit/submission_E003b.csv (若有效)
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
# 自包含: 内联 E003-fusion 核心函数 (文件名含连字符无法 import)

MAX_AA = 256
ALIGN_W = 0.4
TOP_K = 3
SIM_W = 0.75
MAX_ACCEPT = 5
EVALUE = 0.001


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


def macro_rp(ens, y_true, t):
    pred = (ens >= t)
    rs, ps = [], []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
        rs.append(tp / (tp + fn) if tp + fn else 0.0)
        ps.append(tp / (tp + fp) if tp + fp else 0.0)
    return float(np.mean(rs)), float(np.mean(ps))


def parse_aln(tsv_path, query_ids, id_map):
    idx = np.full((len(query_ids), MAX_ACCEPT), -1, dtype=np.int64)
    bits = np.zeros((len(query_ids), MAX_ACCEPT), dtype=np.float32)
    pident = np.zeros((len(query_ids), MAX_ACCEPT), dtype=np.float32)
    qpos = {q: i for i, q in enumerate(query_ids)}
    with open(tsv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            q, t = parts[0], parts[1]
            if q not in qpos or t not in id_map:
                continue
            i = qpos[q]
            try:
                bs, ev, pid = float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            if ev > EVALUE:
                continue
            j = 0
            while j < MAX_ACCEPT and bits[i, j] > bs:
                j += 1
            if j < MAX_ACCEPT:
                idx[i, j:] = np.roll(idx[i, j:], 1)
                bits[i, j:] = np.roll(bits[i, j:], 1)
                pident[i, j:] = np.roll(pident[i, j:], 1)
                idx[i, j] = id_map[t]
                bits[i, j] = bs
                pident[i, j] = pid
    return idx, bits, pident


def agg_labels(idx, w, y_space, top_k):
    n = idx.shape[0]
    valid = idx[:, :top_k] >= 0
    lab = np.zeros((n, top_k, y_space.shape[1]), dtype=np.float32)
    for j in range(top_k):
        lab[:, j] = y_space[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :top_k], 1e-7)
    ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(axis=1) / (ww.sum(axis=1)[:, None] + 1e-7)


def knn_agg_cos(q_emb, db_emb, db_y, top_k=3):
    q = torch.from_numpy(q_emb.astype(np.float32))
    db = torch.from_numpy(db_emb.astype(np.float32))
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    db = db / db.norm(dim=1, keepdim=True).clamp(min=1e-8)
    q, db = q.to("cuda"), db.to("cuda")
    n = q.shape[0]
    idx = np.zeros((n, 5), dtype=np.int64)
    sims = np.zeros((n, 5), dtype=np.float32)
    for s in range(0, n, 512):
        e = min(s + 512, n)
        top = torch.topk(q[s:e] @ db.T, 5, dim=1)
        idx[s:e] = top.indices.cpu().numpy()
        sims[s:e] = top.values.cpu().numpy()
    lab = db_y[idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    w = np.maximum(sims[:, :top_k], 1e-7)
    return (lab * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]


def build_kmer(idx_train, idx_val, y_train, split="val", y_full=None):
    K_RANGE = [3, 4, 5, 6, 7, 8]
    FUSION = [0, 0.1, 0.2, 0.4, 0.2, 0.1]
    if split == "val":
        n = len(idx_val)
        sim = np.zeros((n, 500), dtype=np.float32)
        for k, wt in zip(K_RANGE, FUSION):
            if wt <= 0:
                continue
            prefix = "CK003" if k >= 7 else "CK002"
            dd = np.load(f"{REL_DIR}/{prefix}_neigh_k{k}.npz")
            lab = y_train[dd["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
            s = dd["sim_val"][:, :5].astype(np.float32)
            ww = np.maximum(s, 1e-7) ** 2
            sim += wt * (lab * ww[:, :, None]).sum(axis=1) / ww.sum(axis=1)[:, None]
        return sim
    else:
        test_df = pd.read_csv(f"{PROJECT_ROOT}/data/test.csv")
        n2 = len(test_df)
        y_space = y_full
        sim = np.zeros((n2, 500), dtype=np.float32)
        for k, wt in zip(K_RANGE, FUSION):
            if wt <= 0:
                continue
            dd = np.load(f"{REL_DIR}/TCK002a_neigh_full.npz" if k == 6
                         else f"{REL_DIR}/CK004_neigh_k{k}_full.npz")
            lab = y_space[dd["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
            s = dd["sim_test"][:, :5].astype(np.float32)
            ww = np.maximum(s, 1e-7) ** 2
            sim += wt * (lab * ww[:, :, None]).sum(axis=1) / ww.sum(axis=1)[:, None]
        return sim


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print(f"ECK002-labelcorr: 标签条件依赖稀疏增强 | 模式: {'含提交' if gen_sub else '评估'}")

    print("[1/5] 数据 + 构建 ECK001 集成概率 (val)...")
    train_df = pd.read_csv(f"{PROJECT_ROOT}/data/train.csv")
    test_df = pd.read_csv(f"{PROJECT_ROOT}/data/test.csv")
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    # 必须与 E003-fusion 完全一致: train_test_split 返回的 idx_train 顺序
    # (kmer/align 邻居缓存按此顺序建索引; setdiff1d 会错位)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=0.2, random_state=42)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    lens = train_df["sequence"].str.len().values
    val_short = lens[idx_val] <= MAX_AA
    d = np.load(f"{REL_DIR}/E003_pt5_emb_short.npz")
    emb_s_tr = d["emb_s_tr"]
    short_tr = np.where(lens <= MAX_AA)[0]
    pos = np.searchsorted(short_tr, idx_val[val_short])
    sim_pt5_s = knn_agg_cos(emb_s_tr[pos], emb_s_tr, y_full[short_tr], 3)
    sim_kmer = build_kmer(idx_train, idx_val, y_train, "val")
    val_ids = train_df["protein_id"].values[idx_val]
    train_db_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    idx_v, bits_v, _ = parse_aln(f"{REL_DIR}/ck005_tmp/align_val.tsv", val_ids, train_db_map)
    sim_bit = agg_labels(idx_v, bits_v, y_train, TOP_K)
    sim_base = ALIGN_W * sim_bit + (1 - ALIGN_W) * sim_kmer
    sim_f = sim_base.copy()
    sim_f[val_short] = 0.6 * sim_pt5_s + 0.4 * sim_base[val_short]
    vp = np.load(f"{REL_DIR}/TC003_val_probs.npz"); cnn = vp["cnn"]
    t1 = np.load(f"{REL_DIR}/TCK001_t001a_probs.npz"); t1a = t1["t001a_val"]
    t2v = np.load(f"{REL_DIR}/T002_val_probs.npz"); t2 = t2v["t002"]
    w = json.load(open(f"{PROJECT_ROOT}/models/tck002-ensemble.json"))["best_w"]
    ens = w["t001a"] * t1a + w["t002"] * t2 + w["cnn"] * cnn + SIM_W * sim_f
    cands = np.linspace(0.01, 0.99, 100)
    best_t = np.full(500, 0.5)
    for i in range(500):
        yt = y_val[:, i]
        if yt.sum() == 0:
            continue
        p = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (p & ytm).sum(0); fp = (p & ~ytm).sum(0); fn = (~p & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_t[i] = cands[f1t.argmax()]
    pred_base = ens >= best_t[None, :]
    f1_base = np.mean([f1_score(y_val[:, i].astype(bool), pred_base[:, i], zero_division=0)
                       for i in range(500) if y_val[:, i].sum() > 0])
    print(f"  ECK001 基线 pred: F1={f1_base:.4f} (应≈0.4558)")

    print("[2/5] 训练集条件概率矩阵 (稀疏 top-K)...")
    C = y_train.T @ y_train  # (500,500) 共现计数
    # 条件概率 P(row|col): 行 = 伴随标签, 列 = 已判阳性标签
    P = C / np.maximum(C.sum(axis=0), 1)[None, :]  # P[i|j]
    np.fill_diagonal(P, 0)

    print("[3/5] 决策后稀疏增强网格 (K, τ)...")
    results = {}
    for K in [3, 5, 10]:
        for tau in [0.30, 0.50, 0.70]:
            # 每列(阳性标签)只保留 top-K 强伴随
            Pk = np.zeros_like(P)
            for j in range(500):
                nn = np.argsort(P[:, j])[::-1]
                keep = nn[:K]
                Pk[keep, j] = P[keep, j]
                Pk[keep, j] /= Pk[keep, j].sum() + 1e-9  # 重归一
            # 增强: 对每个蛋白, 阳性标签集合 S → 补入 max P(j|i∈S) > τ 的伴随标签
            S_mask = pred_base  # n x 500
            n = len(y_val)
            add_score = np.zeros((n, 500), dtype=np.float32)
            # 向量化: add_score[j] = max_{i in S} Pk[j,i]
            for i in range(n):
                si = S_mask[i]
                if si.sum() == 0:
                    continue
                add_score[i] = Pk[:, si].max(axis=1)
            pred_new = pred_base | (add_score > tau)
            f1 = np.mean([f1_score(y_val[:, i].astype(bool), pred_new[:, i], zero_division=0)
                          for i in range(500) if y_val[:, i].sum() > 0])
            results[f"K{K}_t{tau}"] = round(f1, 6)
    print(f"  网格: {results}")
    print(f"  基线: {f1_base:.4f}")

    # 最优配置
    best_key = max(results, key=results.get)
    best_f1 = results[best_key]
    gain = best_f1 - f1_base
    print(f"  最优 {best_key} F1={best_f1:.4f} (Δ={gain:+.4f})")

    report = {"baseline_f1": round(f1_base, 6), "grid": results,
              "best": best_key, "best_f1": round(best_f1, 6), "gain": round(gain, 6)}

    if not gen_sub or gain <= 0.001:
        if gain <= 0.001:
            report["submission"] = "none (无增益)"
        with Path(f"{REL_DIR}/E003b_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  报告: {REL_DIR}/E003b_report.json")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    print("[4/5] test 端...")
    # test 端 E003a 概率
    d2 = np.load(f"{REL_DIR}/E003_pt5_emb_short.npz")
    emb_s_te = d2["emb_s_te"]
    short_te = np.where(test_df["sequence"].str.len().values <= MAX_AA)[0]
    sim_pt5_test = knn_agg_cos(emb_s_te, emb_s_tr, y_full[short_tr], 3)
    sim_kmer_test = build_kmer(idx_train, idx_val, y_train, "test", y_full=y_full)
    test_df2 = test_df
    test_ids = test_df2["protein_id"].values
    full_ids = train_df["protein_id"].values
    full_db_map = {pid: i for i, pid in enumerate(full_ids)}
    idx_t, bits_t, _ = parse_aln(f"{REL_DIR}/ck005_tmp/align_test.tsv", test_ids, full_db_map)
    sim_bit_test = agg_labels(idx_t, bits_t, y_full, TOP_K)
    sim_base_test = ALIGN_W * sim_bit_test + (1 - ALIGN_W) * sim_kmer_test
    sim_test = sim_base_test.copy()
    sim_test[short_te] = 0.6 * sim_pt5_test + 0.4 * sim_base_test[short_te]
    tp = np.load(f"{REL_DIR}/TC003_test_probs.npz"); cnn_t = tp["cnn"]
    t1t = np.load(f"{REL_DIR}/TCK001_t001a_probs.npz"); t1a_t = t1t["t001a_test"]
    t2t = np.load(f"{REL_DIR}/T002_test_probs.npz"); t2_t = t2t["t002"]
    ens_test = w["t001a"] * t1a_t + w["t002"] * t2_t + w["cnn"] * cnn_t + SIM_W * sim_test
    pred_test_base = ens_test >= best_t[None, :]
    # 增强
    K, tau = int(best_key.split("K")[1].split("_")[0]), float(best_key.split("t")[1])
    Pk = np.zeros_like(P)
    for j in range(500):
        nn = np.argsort(P[:, j])[::-1]
        keep = nn[:K]
        Pk[keep, j] = P[keep, j]
        Pk[keep, j] /= Pk[keep, j].sum() + 1e-9
    add_score = np.zeros((len(ens_test), 500), dtype=np.float32)
    for i in range(len(ens_test)):
        si = pred_test_base[i]
        if si.sum() == 0:
            continue
        add_score[i] = Pk[:, si].max(axis=1)
    pred_test = pred_test_base | (add_score > tau)

    print("[5/5] 提交...")
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df2[["protein_id"]].copy()
    label_df = pd.DataFrame(pred_test.astype(np.int8), columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    sub_path = f"{SUB_DIR}/submission_E003b.csv"
    submission.to_csv(sub_path, index=False)
    report["submission"] = sub_path
    with Path(f"{REL_DIR}/E003b_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  提交: {sub_path} ({len(submission)} 行)")
    print(f"  报告: {REL_DIR}/E003b_report.json")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
