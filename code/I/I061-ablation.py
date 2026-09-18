"""
I061-ablation — I060 组件精确消融 (2-hop 同源 vs PLM-MLP vs 两者)
=================================================================
目的: 定位 I060 增益 (+0.0018) 究竟来自哪个组件，为后续投入确定方向。
对照:
  A. C011a 基线 (1-hop + 4路deep + RRF + Sinkhorn)
  B. 仅加 2-hop 同源
  C. 仅加 PLM-MLP (w_plm=0.25)
  D. I060 = 2-hop + PLM-MLP (两者)
"""
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
N_LABELS = 500


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


def agg_labels(idx, w, y_space, top_k=3):
    n = idx.shape[0]
    valid = idx[:, :top_k] >= 0
    lab = np.zeros((n, top_k, y_space.shape[1]), np.float32)
    for j in range(top_k):
        lab[:, j] = y_space[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :top_k], 1e-7)
    ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)


def ranks(s):
    return np.argsort(np.argsort(-s, axis=1, kind="mergesort"), axis=1)


def rrf_pair(d, s, k=30):
    return 1.0 / (k + ranks(d) + 1) + 1.0 / (k + ranks(s) + 1)


def col_norm(x):
    lo, hi = x.min(axis=0), x.max(axis=0)
    return (x - lo) / np.maximum(hi - lo, 1e-9)


def sinkhorn(K, r, c, it=100, tol=1e-5):
    u = np.ones_like(r); v = np.ones_like(c)
    for _ in range(it):
        un = r / np.maximum(K @ v, 1e-12)
        vn = c / np.maximum(K.T @ un, 1e-12)
        if np.max(np.abs(un - u)) < tol and np.max(np.abs(vn - v)) < tol: break
        u, v = un, vn
    P = u[:, None] * K * v[None, :]
    lo, hi = P.min(axis=0), P.max(axis=0)
    return (P - lo) / np.maximum(hi - lo, 1e-9)


def search_scut(ens, y):
    cands = np.linspace(0.01, 0.99, 100)
    best = np.full(N_LABELS, 0.5)
    for i in range(N_LABELS):
        yt = y[:, i]
        if yt.sum() == 0: continue
        pr = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0); fp = (pr & ~ytm).sum(0); fn = (~pr & ytm).sum(0)
        f1t = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
        best[i] = cands[f1t.argmax()]
    return best


def macro_f1(probs, y, ths):
    preds = (probs >= ths).astype(bool)
    yb = y.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
    p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return float(np.mean(f))


def full_pipeline(deep_v, sim_v, y_train, y_val, n_val):
    s = deep_v + 0.75 * sim_v + 0.50 * col_norm(rrf_pair(deep_v, sim_v, 30))
    t_tmp = search_scut(s, y_val)
    row_m = np.maximum((s >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_train.mean(0) * n_val * 1.0, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s - s.max(axis=1, keepdims=True)) / 0.20)
    p_v = sinkhorn(ker, row_m, col_m)
    mixed = 0.40 * s + 0.60 * p_v
    ths = search_scut(mixed, y_val)
    f1 = macro_f1(mixed, y_val, ths)
    auc = roc_auc_score(y_val, mixed, average="macro")
    return f1, auc, mixed, ths


def main():
    t0 = time.time()
    print("=" * 70)
    print("  I061: Precise Component Ablation for I060")
    print("=" * 70)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    n_val = len(idx_val)
    val_pids = train_df["protein_id"].values[idx_val]
    train_pids = train_df["protein_id"].values[idx_train]
    train_map = {pid: i for i, pid in enumerate(train_pids)}
    val_map = {pid: i for i, pid in enumerate(val_pids)}

    # 深度 4 路
    cnn_val = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_val = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    mix_val = 0.5 * cnn_val + 0.5 * c005_val
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        w_cfg = json.load(f)["best_w"]
    deep4 = w_cfg["t001a"] * t1 + w_cfg["t002"] * t2 + w_cfg["cnn"] * mix_val
    plm_mlp = np.load(os.path.join(REL_DIR, "fast_plm_mlp_val_probs.npz"))["val"]
    deep5 = 0.75 * deep4 + 0.25 * plm_mlp

    # 1-hop 同源
    idx_v, bits_v = parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                              val_pids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim1 = agg_labels(idx_v, bits_v, y_train)

    # 2-hop 传递同源
    sim2 = sim1.copy()
    v2v = {}
    with open(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            q, t = p[0], p[1]
            if q == t or q not in val_map or t not in val_map: continue
            qi, ti = val_map[q], val_map[t]
            v2v.setdefault(qi, [])
            if len(v2v[qi]) < 5: v2v[qi].append((ti, float(p[2])))
    rescued = 0
    for qi in range(n_val):
        if miss[qi] and qi in v2v:
            acc = np.zeros(N_LABELS, np.float32); aw = 0.0
            for ti, bs in v2v[qi]:
                if hit[ti]:
                    acc += sim1[ti] * bs; aw += bs
            if aw > 0:
                sim2[qi] = acc / aw
                rescued += 1
    print(f"  2-hop rescued {rescued} miss sequences")

    def build_kmer(y_tr):
        s = 0.0
        for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
            if wt <= 0: continue
            pre = "CK003" if k >= 7 else "CK002"
            d = np.load(os.path.join(REL_DIR, f"{pre}_neigh_k{k}.npz"))
            n = d["idx_val"].shape[0]
            lab = y_tr[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
            sm = d["sim_val"][:, :5].astype(np.float32)
            ww = np.maximum(sm, 1e-7) ** 2
            s += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
        return s

    kmer = build_kmer(y_train)
    sim1_full = 0.4 * sim1 + 0.6 * kmer
    sim2_full = 0.4 * sim2 + 0.6 * kmer

    print("\n" + "=" * 70)
    print(f"{'Config':42s} | {'Val F1':>8s} | {'AUC':>8s} | {'vs C011a':>9s}")
    print("=" * 70)

    configs = [
        ("A. C011a baseline (1-hop + 4-way deep)", deep4, sim1_full),
        ("B. + 2-hop homology only", deep4, sim2_full),
        ("C. + PLM-MLP only (w=0.25)", deep5, sim1_full),
        ("D. I060 = 2-hop + PLM-MLP", deep5, sim2_full),
    ]
    results = {}
    for name, dv, sv in configs:
        f1, auc, _, _ = full_pipeline(dv, sv, y_train, y_val, n_val)
        results[name] = (f1, auc)
        print(f"{name:42s} | {f1:8.4f} | {auc:8.4f} | {f1-0.4453:+9.4f}")

    # miss 子集切片
    print("\n--- Miss Subset F1 (orphan sequences) ---")
    for name, dv, sv in configs:
        f1, auc, mixed, ths = full_pipeline(dv, sv, y_train, y_val, n_val)
        preds = (mixed >= ths).astype(bool)
        pm, ym = preds[miss], y_val[miss].astype(bool)
        tp = (pm & ym).sum(0); fp = (pm & ~ym).sum(0); fn = (~pm & ym).sum(0)
        f1m = np.mean(np.where(2*tp+fp+fn > 0, 2*tp/np.maximum(2*tp+fp+fn,1), 0.0))
        print(f"{name:42s} | Miss F1 = {f1m:.4f}")

    with open(os.path.join(REL_DIR, "I061_report.json"), "w", encoding="utf-8") as f:
        json.dump({k: {"f1": v[0], "auc": v[1]} for k, v in results.items()}, f, indent=2)
    print(f"\nCompleted in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
