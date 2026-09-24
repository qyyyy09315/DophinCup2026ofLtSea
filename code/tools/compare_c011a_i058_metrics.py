import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
label_cols = [c for c in train_df.columns if c.startswith("label_")]
n_labels = len(label_cols)
y_full = train_df[label_cols].values.astype(np.int8)

idx_train, idx_val = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
y_train, y_val = y_full[idx_train], y_full[idx_val]
n_val = len(idx_val)

cnn_val = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
c005_val = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
mix_val = 0.5 * cnn_val + 0.5 * c005_val
t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
with open(TCK002_CONFIG, encoding="utf-8") as f:
    w = json.load(f)["best_w"]
deep_v = w["t001a"] * t1 + w["t002"] * t2 + w["cnn"] * mix_val

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

val_ids = train_df["protein_id"].values[idx_val]
train_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
tsv = os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv")
idx_v, bits_v = parse_aln(tsv, val_ids, train_map)
hit_mask = idx_v[:, 0] >= 0
miss_mask = ~hit_mask

def agg_labels(idx, w, y_space, top_k=3):
    n = idx.shape[0]
    valid = idx[:, :top_k] >= 0
    lab = np.zeros((n, top_k, y_space.shape[1]), np.float32)
    for j in range(top_k): lab[:, j] = y_space[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :top_k], 1e-7)
    ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)

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

sim_v = 0.4 * agg_labels(idx_v, bits_v, y_train) + 0.6 * build_kmer_val(y_train)

def ranks(score):
    return np.argsort(np.argsort(-score, axis=1, kind="mergesort"), axis=1)

def rrf_pair(deep, sim, k):
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

s011_v = deep_v + 0.75 * sim_v + 0.50 * col_norm(rrf_pair(deep_v, sim_v, 30))

cands = np.linspace(0.01, 0.99, 100)
best_t_i011 = np.full(500, 0.5, dtype=np.float64)
for i in range(500):
    yt = y_val[:, i]
    if yt.sum() == 0: continue
    pr = s011_v[:, i][:, None] >= cands[None, :]
    ytm = yt[:, None].astype(bool)
    tp = (pr & ytm).sum(0)
    fp = (pr & ~ytm).sum(0)
    fn = (~pr & ytm).sum(0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    best_t_i011[i] = cands[f1t.argmax()]

# --- 1. C011a 原版 (未校准) 在验证集上的指标 ---
pi_train = y_train.mean(0).astype(np.float64)
row_m_a = np.maximum((s011_v >= best_t_i011).sum(1).astype(np.float64), 1.0)
col_m_a = np.maximum(pi_train * n_val * 1.0, 1e-6)
col_m_a *= row_m_a.sum() / col_m_a.sum()
ker_a = np.exp((s011_v - s011_v.max(axis=1, keepdims=True)) / 0.20)
p_v_a = sinkhorn(ker_a, row_m_a, col_m_a)
s_mixed_a = 0.40 * s011_v + 0.60 * p_v_a

# 搜 100 档 SCut 阈值
t_sub_a = np.full(500, 0.5)
for i in range(500):
    yt = y_val[:, i]
    if yt.sum() == 0: continue
    pr = s_mixed_a[:, i][:, None] >= cands[None, :]
    ytm = yt[:, None].astype(bool)
    tp = (pr & ytm).sum(0)
    fp = (pr & ~ytm).sum(0)
    fn = (~pr & ytm).sum(0)
    f1t = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    t_sub_a[i] = cands[f1t.argmax()]

preds_a = (s_mixed_a >= t_sub_a).astype(np.int8)

# --- 2. I058 算法在验证集上的模拟 ---
target_mean_card = float(y_train.sum(1).mean())
target_freq = y_train.sum(0) / len(y_train)

scale_factor = target_mean_card / np.maximum(row_m_a.mean(), 1.0)
row_m_cal = np.maximum(row_m_a * scale_factor, 1.0)
row_m_cal *= (target_mean_card * n_val) / row_m_cal.sum()
col_m_cal = np.maximum(pi_train * n_val * 1.0, 1e-6)
col_m_cal *= row_m_cal.sum() / col_m_cal.sum()

p_v_cal = sinkhorn(ker_a, row_m_cal, col_m_cal)
s_mixed_cal = 0.40 * s011_v + 0.60 * p_v_cal

preds_i058 = np.zeros_like(s_mixed_cal, dtype=np.int8)
ths_i058 = np.zeros(500)
for j in range(n_labels):
    k_expected = int(np.round(target_freq[j] * n_val))
    scores_col = s_mixed_cal[:, j]
    cutoff_val = np.partition(scores_col, -k_expected)[-k_expected]
    effective_th = max(min(cutoff_val, 0.48), 0.22)
    ths_i058[j] = effective_th
    preds_i058[:, j] = (scores_col >= effective_th).astype(np.int8)

def calc_all(preds, y, scores):
    tp = (preds & y).sum(0)
    fp = (preds & ~y).sum(0)
    fn = (~preds & y).sum(0)
    p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    
    def sf(mask):
        sp, sy = preds[mask], y[mask]
        stp = (sp & sy).sum(0)
        sfp = (sp & ~sy).sum(0)
        sfn = (~sp & sy).sum(0)
        return np.mean(np.where(2*stp + sfp + sfn > 0, 2*stp / (2*stp + sfp + sfn), 0.0))
        
    auc = roc_auc_score(y, scores, average="macro")
    auc_bin = roc_auc_score(y, preds, average="macro")
    
    return {
        "macro_f1": float(np.mean(f)),
        "macro_p": float(np.mean(p)),
        "macro_r": float(np.mean(r)),
        "auc_continuous": float(auc),
        "auc_binary": float(auc_bin),
        "hit_f1": float(sf(hit_mask)),
        "miss_f1": float(sf(miss_mask)),
        "mean_card": float(preds.sum(1).mean()),
        "median_card": float(np.median(preds.sum(1))),
        "total_pos": int(preds.sum())
    }

res_a = calc_all(preds_a, y_val, s_mixed_a)
res_i058 = calc_all(preds_i058, y_val, s_mixed_cal)

print("=" * 65)
print(f"{'Metric':20s} | {'C011a (Original)':16s} | {'I058 (Calibrated)':16s} | {'Diff':10s}")
print("=" * 65)
for k in res_a.keys():
    va, vi = res_a[k], res_i058[k]
    if isinstance(va, float):
        print(f"{k:20s} | {va:16.4f} | {vi:16.4f} | {vi - va:+10.4f}")
    else:
        print(f"{k:20s} | {va:16d} | {vi:16d} | {vi - va:+10d}")
print("=" * 65)
PY