"""
generate_soft_prob_submission.py — 生成基于 C011a 架构的连续软概率提交文件
===========================================================================
依据与背景:
  1. 榜单第一名 (52.8565 / AUC 0.875519) 仅提交 5 次；
  2. C011a 在验证集上的连续概率 Macro ROC-AUC 高达 0.860850，而二值化后线上 AUC 仅为 0.713251；
  3. 本脚本直接提取 C011a 在测试集上的连续混合概率 (经 RRF 秩融合与 Sinkhorn 双边际最优传输正则化后的全局概率)，
     将其归一化并保留 4 位浮点数，生成 submission_C011a_prob.csv，供验证平台连续概率打分机制。
用法:
  python code/tools/generate_soft_prob_submission.py
"""

import json, os, time
import numpy as np
import pandas as pd

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

print("1. Loading datasets and probabilities...")
t0 = time.time()
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
label_cols = [c for c in train_df.columns if c.startswith("label_")]
n_labels = len(label_cols)
n_test = len(test_df)
y_full = train_df[label_cols].values.astype(np.int8)

c005_test = np.load(os.path.join(REL_DIR, "C005_test_probs.npz"))["test"]
cnn_test = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
mix_te = 0.5 * cnn_test + 0.5 * c005_test

t1_test = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"]
t2_test = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]

with open(TCK002_CONFIG, encoding="utf-8") as f:
    w = json.load(f)["best_w"]

deep_t = w["t001a"] * t1_test + w["t002"] * t2_test + w["cnn"] * mix_te

def parse_aln(tsv, query_ids, id_map, evalue=0.001, max_accept=5):
    idx = np.full((len(query_ids), max_accept), -1, np.int64)
    bits = np.zeros((len(query_ids), max_accept), np.float32)
    qpos = {q: i for i, q in enumerate(query_ids)}
    with open(tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            q, t = p[0], p[1]
            if q not in qpos or t not in id_map: continue
            i = qpos[q]
            try: bs, ev = float(p[2]), float(p[3])
            except ValueError: continue
            if ev > evalue: continue
            j = 0
            while j < max_accept and bits[i, j] > bs: j += 1
            if j < max_accept:
                idx[i, j:] = np.roll(idx[i, j:], 1)
                bits[i, j:] = np.roll(bits[i, j:], 1)
                idx[i, j] = id_map[t]
                bits[i, j] = bs
    return idx, bits

tsv_te = os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv")
full_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
idx_t, bits_t = parse_aln(tsv_te, test_df["protein_id"].values, full_map)

def agg_labels(idx, w, y_space, top_k=3):
    n = idx.shape[0]
    valid = idx[:, :top_k] >= 0
    lab = np.zeros((n, top_k, y_space.shape[1]), np.float32)
    for j in range(top_k): lab[:, j] = y_space[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :top_k], 1e-7)
    ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)

sim_bit_t = agg_labels(idx_t, bits_t, y_full)

sim_kmer_t = 0.0
for kk, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
    if wt <= 0: continue
    d = np.load(os.path.join(
        REL_DIR, "TCK002a_neigh_full.npz" if kk == 6
        else f"CK004_neigh_k{kk}_full.npz"))
    n2 = d["idx_test"].shape[0]
    lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
    s = d["sim_test"][:, :5].astype(np.float32)
    ww = np.maximum(s, 1e-7) ** 2
    sim_kmer_t = sim_kmer_t + wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]

sim_t = 0.4 * sim_bit_t + 0.6 * sim_kmer_t

def ranks(score):
    return np.argsort(np.argsort(-score, axis=1, kind="mergesort"), axis=1)

def rrf_pair(deep, sim, k=30):
    return 1.0 / (k + ranks(deep) + 1) + 1.0 / (k + ranks(sim) + 1)

def col_norm(x):
    lo = x.min(axis=0)
    hi = x.max(axis=0)
    return (x - lo) / np.maximum(hi - lo, 1e-9)

def sinkhorn(K, row_m, col_m, max_iter=100, tol=1e-5):
    u = np.ones_like(row_m)
    v = np.ones_like(col_m)
    for _ in range(max_iter):
        u_new = row_m / np.maximum(K @ v, 1e-12)
        v_new = col_m / np.maximum(K.T @ u_new, 1e-12)
        if np.max(np.abs(u_new - u)) < tol and np.max(np.abs(v_new - v)) < tol:
            break
        u, v = u_new, v_new
    P = u[:, None] * K * v[None, :]
    lo = P.min(axis=0)
    hi = P.max(axis=0)
    return (P - lo) / np.maximum(hi - lo, 1e-9)

print("2. Computing RRF and Sinkhorn Optimal Transport...")
s011_t = deep_t + 0.75 * sim_t + 0.50 * col_norm(rrf_pair(deep_t, sim_t, 30))

pi = y_full.mean(0).astype(np.float64)
row_t = np.maximum((s011_t >= 0.35).sum(1).astype(np.float64), 1.0)
col_t = np.maximum(pi * n_test * 1.0, 1e-6)
col_t *= row_t.sum() / col_t.sum()

ker_t = np.exp((s011_t - s011_t.max(axis=1, keepdims=True)) / 0.20)
p_t = sinkhorn(ker_t, row_t, col_t)

# 连续概率混合矩阵 (0.4 * s011 + 0.6 * p_t)
mixed_t = 0.40 * s011_t + 0.60 * p_t

# 严格按列归一化到 [0.0, 1.0] 纯净概率区间
prob_matrix = np.clip(col_norm(mixed_t), 0.0, 1.0)

print(f"Continuous Probability matrix computed: shape={prob_matrix.shape}")
print(f"  Range: [{prob_matrix.min():.4f}, {prob_matrix.max():.4f}], Mean: {prob_matrix.mean():.4f}")

# 生成连续浮点数提交文件 (保留 4 位小数)
out_csv = os.path.join(SUB_DIR, "submission_C011a_prob.csv")
sub_df = pd.concat([test_df[["protein_id"]], pd.DataFrame(np.round(prob_matrix, 4), columns=label_cols)], axis=1)
sub_df.to_csv(out_csv, index=False)
print(f"Written continuous probability submission: {out_csv} ({len(sub_df)} rows)")

# 复制到桌面备用
desktop_path = r"C:\Users\YKSHb\Desktop\submission_C011a_prob.csv"
sub_df.to_csv(desktop_path, index=False)
print(f"Copied to Desktop: {desktop_path}")
print(f"Total time: {time.time()-t0:.1f}s")