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

print("1. Loading validation features, existing models and new PLM-MLP...")
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

# 加载已有 4 路模型
cnn_val = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
c005_val = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
mix_val = 0.5 * cnn_val + 0.5 * c005_val
t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
with open(TCK002_CONFIG, encoding="utf-8") as f:
    w_cfg = json.load(f)["best_w"]
deep_v = w_cfg["t001a"] * t1 + w_cfg["t002"] * t2 + w_cfg["cnn"] * mix_val

# 加载新训练的高效残差 PLM-MLP (ESM2+ProtTrans)
plm_mlp_v = np.load(os.path.join(REL_DIR, "fast_plm_mlp_val_probs.npz"))["val"]

# 1-hop & 2-hop 比对
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

idx_v, bits_v = parse_aln(tsv_val_tr, val_pids, train_map)
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

sim_bit_v = agg_labels(idx_v, bits_v, y_train)

# 2-hop 传递同源对 miss 补充
val_map = {pid: i for i, pid in enumerate(val_pids)}
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
            if qi not in val_to_val_hits: val_to_val_hits[qi] = []
            if len(val_to_val_hits[qi]) < 5: val_to_val_hits[qi].append((ti, bs))

sim_bit_2hop = sim_bit_v.copy()
for qi in range(n_val):
    if miss_mask[qi] and qi in val_to_val_hits:
        accum_lab = np.zeros(n_labels, dtype=np.float32)
        accum_w = 0.0
        for ti, bs1 in val_to_val_hits[qi]:
            if hit_mask[ti]:
                lab_ti = sim_bit_v[ti]
                accum_lab += lab_ti * bs1
                accum_w += bs1
        if accum_w > 0:
            sim_bit_2hop[qi] = accum_lab / accum_w

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
sim_v_2hop = 0.4 * sim_bit_2hop + 0.6 * kmer_v

def ranks(score):
    return np.argsort(np.argsort(-score, axis=1, kind="mergesort"), axis=1)

def rrf_pair(deep, sim, k=30):
    return 1.0 / (k + ranks(deep) + 1) + 1.0 / (k + ranks(sim) + 1)

def rrf_trio(deep, sim, plm, k=30):
    return 1.0 / (k + ranks(deep) + 1) + 1.0 / (k + ranks(sim) + 1) + 1.0 / (k + ranks(plm) + 1)

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
    preds = (probs >= ths).astype(bool)
    targs = targets.astype(bool)
    tp = (preds & targs).sum(0)
    fp = (preds & ~targs).sum(0)
    fn = (~preds & targs).sum(0)
    p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return float(np.mean(f))

# ------------------------------------------------------------------------------
# 实验 1: 将 PLM-MLP 作为特征增强融入 Deep 分支
# ------------------------------------------------------------------------------
print("\n--- Strategy A: Fusing PLM-MLP into Deep Feature Branch ---")
for w_plm in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    deep_enhanced = (1.0 - w_plm) * deep_v + w_plm * plm_mlp_v
    s_rrf = deep_enhanced + 0.75 * sim_v_2hop + 0.50 * col_norm(rrf_pair(deep_enhanced, sim_v_2hop, 30))
    t_tmp = search_scut(s_rrf, y_val)
    row_m = np.maximum((s_rrf >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_train.mean(0) * n_val * 1.0, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s_rrf - s_rrf.max(axis=1, keepdims=True)) / 0.20)
    p_v = sinkhorn(ker, row_m, col_m)
    mixed = 0.40 * s_rrf + 0.60 * p_v
    ths = search_scut(mixed, y_val)
    f1 = macro_f1(mixed, y_val, ths)
    auc = roc_auc_score(y_val, mixed, average="macro")
    print(f"  w_plm={w_plm:.2f} -> Val Macro F1: {f1:.4f} | Continuous AUC: {auc:.4f}")

# ------------------------------------------------------------------------------
# 实验 2: 3-way RRF (Deep vs Sim vs PLM-MLP)
# ------------------------------------------------------------------------------
print("\n--- Strategy B: 3-way RRF Rank Fusion (Deep + Sim + PLM-MLP) ---")
for beta_plm in [0.10, 0.20, 0.30, 0.40]:
    rrf_3way = col_norm(rrf_trio(deep_v, sim_v_2hop, plm_mlp_v, 30))
    s_rrf3 = deep_v + 0.75 * sim_v_2hop + beta_plm * rrf_3way
    t_tmp = search_scut(s_rrf3, y_val)
    row_m = np.maximum((s_rrf3 >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_train.mean(0) * n_val * 1.0, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s_rrf3 - s_rrf3.max(axis=1, keepdims=True)) / 0.20)
    p_v = sinkhorn(ker, row_m, col_m)
    mixed = 0.40 * s_rrf3 + 0.60 * p_v
    ths = search_scut(mixed, y_val)
    f1 = macro_f1(mixed, y_val, ths)
    auc = roc_auc_score(y_val, mixed, average="macro")
    print(f"  beta_plm={beta_plm:.2f} -> Val Macro F1: {f1:.4f} | Continuous AUC: {auc:.4f}")

print("\nBaseline C011a was: Val Macro F1 = 0.4453 | Continuous AUC = 0.8609")
print(f"Completed evaluation in {time.time()-t0:.1f}s")