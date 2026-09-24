import os, sys, time, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

print("1. Loading datasets and 1-hop / 2-hop alignments...")
t0 = time.time()
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
label_cols = [c for c in train_df.columns if c.startswith("label_")]
n_labels = len(label_cols)
y_full = train_df[label_cols].values.astype(np.int8)

idx_train, idx_val = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
y_train, y_val = y_full[idx_train], y_full[idx_val]
n_val = len(idx_val)
val_pids = train_df["protein_id"].values[idx_val]
train_pids = train_df["protein_id"].values[idx_train]
train_map = {pid: i for i, pid in enumerate(train_pids)}
val_map = {pid: i for i, pid in enumerate(val_pids)}

# 1-hop: val -> train
tsv_val_tr = os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
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

idx_v_1hop, bits_v_1hop = parse_aln(tsv_val_tr, val_pids, train_map)
hit_1hop = idx_v_1hop[:, 0] >= 0
miss_1hop = ~hit_1hop
print(f"  1-hop Val Hit: {hit_1hop.sum()}, Miss: {miss_1hop.sum()} (out of {n_val})")

# 2-hop: val -> val (自比对)
tsv_val_val = os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv")
val_to_val_hits = {}
with open(tsv_val_val, 'r') as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 5: continue
        q, t = p[0], p[1]
        if q == t: continue
        if q in val_map and t in val_map:
            qi, ti = val_map[q], val_map[t]
            bs = float(p[2])
            if qi not in val_to_val_hits:
                val_to_val_hits[qi] = []
            if len(val_to_val_hits[qi]) < 5:
                val_to_val_hits[qi].append((ti, bs))

rescued_miss_count = 0
for qi in range(n_val):
    if miss_1hop[qi] and qi in val_to_val_hits:
        for ti, bs in val_to_val_hits[qi]:
            if hit_1hop[ti]:
                rescued_miss_count += 1
                break

print(f"  2-hop Rescued Miss sequences: {rescued_miss_count} / {miss_1hop.sum()} ({rescued_miss_count/miss_1hop.sum()*100:.1f}%)")

def agg_labels(idx, w, y_space, top_k=3):
    n = idx.shape[0]
    valid = idx[:, :top_k] >= 0
    lab = np.zeros((n, top_k, y_space.shape[1]), np.float32)
    for j in range(top_k): lab[:, j] = y_space[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :top_k], 1e-7)
    ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)

sim_bit_1hop = agg_labels(idx_v_1hop, bits_v_1hop, y_train)

# 构造 2-hop 补充特征
sim_bit_2hop = sim_bit_1hop.copy()
for qi in range(n_val):
    if miss_1hop[qi] and qi in val_to_val_hits:
        accum_lab = np.zeros(n_labels, dtype=np.float32)
        accum_w = 0.0
        for ti, bs1 in val_to_val_hits[qi]:
            if hit_1hop[ti]:
                lab_ti = sim_bit_1hop[ti]
                w_hop = bs1
                accum_lab += lab_ti * w_hop
                accum_w += w_hop
        if accum_w > 0:
            sim_bit_2hop[qi] = accum_lab / accum_w

print("2. Comparing 1-hop Sim vs 1-hop + 2-hop Sim...")
def build_kmer_val(y_train):
    sim = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0: continue
        prefix = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{prefix}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_train[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        s = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        sim += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return sim

kmer_v = build_kmer_val(y_train)

cnn_val = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
c005_val = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
mix_val = 0.5 * cnn_val + 0.5 * c005_val
t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
with open(TCK002_CONFIG, encoding="utf-8") as f:
    w_cfg = json.load(f)["best_w"]
deep_v = w_cfg["t001a"] * t1 + w_cfg["t002"] * t2 + w_cfg["cnn"] * mix_val

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

def search_scut(ens, y):
    cands = np.linspace(0.01, 0.99, 100)
    best = np.full(n_labels, 0.5, dtype=np.float64)
    for i in range(n_labels):
        yt = y[:, i]
        if yt.sum() == 0: continue
        pr = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0)
        fp = (pr & ~ytm).sum(0)
        fn = (~pr & ytm).sum(0)
        f1t = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
        best[i] = cands[f1t.argmax()]
    return best

def macro_f1(probs, targets, ths):
    preds = (probs >= ths).astype(np.int8)
    tp = (preds & targets).sum(0)
    fp = (preds & ~targets).sum(0)
    fn = (~preds & targets).sum(0)
    p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return float(np.mean(f))

# 评估 1: 原版 1-hop
sim_v_1hop = 0.4 * sim_bit_1hop + 0.6 * kmer_v
s011_v1 = deep_v + 0.75 * sim_v_1hop + 0.50 * col_norm(rrf_pair(deep_v, sim_v_1hop, 30))
t_tmp1 = search_scut(s011_v1, y_val)
row_m1 = np.maximum((s011_v1 >= t_tmp1).sum(1).astype(np.float64), 1.0)
col_m1 = np.maximum(y_train.mean(0) * n_val * 1.0, 1e-6)
col_m1 *= row_m1.sum() / col_m1.sum()
ker1 = np.exp((s011_v1 - s011_v1.max(axis=1, keepdims=True)) / 0.20)
p_v1 = sinkhorn(ker1, row_m1, col_m1)
mixed1 = 0.40 * s011_v1 + 0.60 * p_v1
ths1 = search_scut(mixed1, y_val)
f1_1hop = macro_f1(mixed1, y_val, ths1)

# 评估 2: 2-hop 增强
sim_v_2hop = 0.4 * sim_bit_2hop + 0.6 * kmer_v
s011_v2 = deep_v + 0.75 * sim_v_2hop + 0.50 * col_norm(rrf_pair(deep_v, sim_v_2hop, 30))
t_tmp2 = search_scut(s011_v2, y_val)
row_m2 = np.maximum((s011_v2 >= t_tmp2).sum(1).astype(np.float64), 1.0)
col_m2 = np.maximum(y_train.mean(0) * n_val * 1.0, 1e-6)
col_m2 *= row_m2.sum() / col_m2.sum()
ker2 = np.exp((s011_v2 - s011_v2.max(axis=1, keepdims=True)) / 0.20)
p_v2 = sinkhorn(ker2, row_m2, col_m2)
mixed2 = 0.40 * s011_v2 + 0.60 * p_v2
ths2 = search_scut(mixed2, y_val)
f1_2hop = macro_f1(mixed2, y_val, ths2)

# 计算 miss 子集上的 F1
p1_miss = (mixed1[miss_1hop] >= ths1).astype(np.int8)
y_miss = y_val[miss_1hop]
tp1 = (p1_miss & y_miss).sum(0)
fp1 = (p1_miss & ~y_miss).sum(0)
fn1 = (~p1_miss & y_miss).sum(0)
f1_miss_1hop = np.mean(np.where(2*tp1 + fp1 + fn1 > 0, 2*tp1 / (2*tp1 + fp1 + fn1), 0.0))

p2_miss = (mixed2[miss_1hop] >= ths2).astype(np.int8)
tp2 = (p2_miss & y_miss).sum(0)
fp2 = (p2_miss & ~y_miss).sum(0)
fn2 = (~p2_miss & y_miss).sum(0)
f1_miss_2hop = np.mean(np.where(2*tp2 + fp2 + fn2 > 0, 2*tp2 / (2*tp2 + fp2 + fn2), 0.0))

print("=" * 60)
print(f"1-Hop Baseline (C011a) Macro F1 : {f1_1hop:.4f} | Miss F1: {f1_miss_1hop:.4f}")
print(f"2-Hop Transitive Homology F1    : {f1_2hop:.4f} | Miss F1: {f1_miss_2hop:.4f}")
print(f"Delta Global F1                 : {f1_2hop - f1_1hop:+.4f}")
print(f"Delta Miss Subset F1            : {f1_miss_2hop - f1_miss_1hop:+.4f}")
print("=" * 60)
PY