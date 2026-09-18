"""
I065 — 最优配置 (多种子 1664D PLM-MLP + 2-hop 同源 + 线性 w=0.35) 生成提交
=========================================================================
依据 I064: multi-seed 1664D + linear w=0.35 -> Val F1 0.4475 (C011a +0.0022)
"""
import os, sys, time, json
import numpy as np
import pandas as pd

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
W_PLM = 0.35


def parse_aln(tsv, qids, idmap, ev=0.001, mx=5):
    idx = np.full((len(qids), mx), -1, np.int64); bits = np.zeros((len(qids), mx), np.float32)
    qp = {q: i for i, q in enumerate(qids)}
    with open(tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            q, t = p[0], p[1]
            if q not in qp or t not in idmap: continue
            i = qp[q]
            try: bs, e = float(p[2]), float(p[3])
            except ValueError: continue
            if e > ev: continue
            j = 0
            while j < mx and bits[i, j] > bs: j += 1
            if j < mx:
                idx[i, j:] = np.roll(idx[i, j:], 1); bits[i, j:] = np.roll(bits[i, j:], 1)
                idx[i, j] = idmap[t]; bits[i, j] = bs
    return idx, bits


def agg(idx, w, ys, tk=3):
    n = idx.shape[0]; valid = idx[:, :tk] >= 0
    lab = np.zeros((n, tk, ys.shape[1]), np.float32)
    for j in range(tk): lab[:, j] = ys[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :tk], 1e-7); ww[~valid] = 0.0
    return (lab*ww[:,:,None]).sum(1)/(ww.sum(1)[:,None]+1e-7)


def ranks(s): return np.argsort(np.argsort(-s, axis=1, kind="mergesort"), axis=1)
def col_norm(x):
    lo, hi = x.min(axis=0), x.max(axis=0)
    return (x-lo)/np.maximum(hi-lo, 1e-9)


def sinkhorn(K, r, c, it=100, tol=1e-5):
    u = np.ones_like(r); v = np.ones_like(c)
    for _ in range(it):
        un = r/np.maximum(K@v, 1e-12); vn = c/np.maximum(K.T@un, 1e-12)
        if np.max(np.abs(un-u)) < tol and np.max(np.abs(vn-v)) < tol: break
        u, v = un, vn
    P = u[:,None]*K*v[None,:]
    lo, hi = P.min(axis=0), P.max(axis=0)
    return (P-lo)/np.maximum(hi-lo, 1e-9)


def main():
    t0 = time.time()
    print("=" * 70)
    print(f"  I065: Optimal Config Submission (multi-seed 1664D, w={W_PLM})")
    print("=" * 70)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    n_te = len(test_df)

    c005_t = np.load(os.path.join(REL_DIR, "C005_test_probs.npz"))["test"]
    cnn_t = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
    t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"]
    t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4_t = wc["t001a"]*t1t + wc["t002"]*t2t + wc["cnn"]*(0.5*cnn_t+0.5*c005_t)

    # 多种子 1664D 测试集概率
    print("  Loading multi-seed 1664D test probabilities...")
    plm_t = np.load(os.path.join(REL_DIR, "plm1664_multiseed_test.npz"))["test"]
    deep5_t = (1-W_PLM)*deep4_t + W_PLM*plm_t

    print("  1-hop + 2-hop transitive homology on test set...")
    full_map = {p: i for i, p in enumerate(train_df["protein_id"].values)}
    idx_t, bits_t = parse_aln(os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv"),
                              test_df["protein_id"].values, full_map)
    hit_t = idx_t[:, 0] >= 0; miss_t = ~hit_t
    sim1_t = agg(idx_t, bits_t, y_full)

    test_pids = test_df["protein_id"].values
    test_map = {p: i for i, p in enumerate(test_pids)}
    te2te = {}
    with open(os.path.join(CK005A_TMP, "align_test_to_test_e0.001.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            q, t = p[0], p[1]
            if q == t or q not in test_map or t not in test_map: continue
            qi, ti = test_map[q], test_map[t]
            te2te.setdefault(qi, [])
            if len(te2te[qi]) < 5: te2te[qi].append((ti, float(p[2])))
    sim2_t = sim1_t.copy()
    rescued = 0
    for qi in range(n_te):
        if miss_t[qi] and qi in te2te:
            acc = np.zeros(N_LABELS, np.float32); aw = 0.0
            for ti, bs in te2te[qi]:
                if hit_t[ti]: acc += sim1_t[ti]*bs; aw += bs
            if aw > 0:
                sim2_t[qi] = acc/aw; rescued += 1
    print(f"  2-hop rescued {rescued} test orphans")

    kmer_t = 0.0
    for kk, wt in zip([3,4,5,6,7,8], [0,0.1,0.2,0.4,0.2,0.1]):
        if wt <= 0: continue
        d = np.load(os.path.join(REL_DIR,
            "TCK002a_neigh_full.npz" if kk == 6 else f"CK004_neigh_k{kk}_full.npz"))
        n2 = d["idx_test"].shape[0]
        lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
        sm = d["sim_test"][:, :5].astype(np.float32)
        ww = np.maximum(sm, 1e-7)**2
        kmer_t += wt*(lab*ww[:,:,None]).sum(1)/ww.sum(1)[:,None]
    sim_t = 0.4*sim2_t + 0.6*kmer_t

    s_t = deep5_t + 0.75*sim_t + 0.50*col_norm(1.0/(30+ranks(deep5_t)+1) + 1.0/(30+ranks(sim_t)+1))

    # 阈值：复用 I064 最优配置在验证集上搜到的 SCut
    ths_path = os.path.join(REL_DIR, "i064_best_ths.npy")
    if os.path.exists(ths_path):
        ths = np.load(ths_path)
        print(f"  Loaded SCut thresholds from {ths_path}")
    else:
        raise FileNotFoundError("Run I064 threshold export first")

    row_t = np.maximum((s_t >= ths).sum(1).astype(np.float64), 1.0)
    col_t = np.maximum(y_full.mean(0)*n_te, 1e-6); col_t *= row_t.sum()/col_t.sum()
    ker_t = np.exp((s_t - s_t.max(axis=1, keepdims=True))/0.20)
    p_t = sinkhorn(ker_t, row_t, col_t)
    mixed_t = 0.40*s_t + 0.60*p_t
    y_pred = (mixed_t >= ths).astype(np.int8)

    out = os.path.join(SUB_DIR, "submission_I065_ensemble.csv")
    sub = pd.concat([test_df[["protein_id"]], pd.DataFrame(y_pred, columns=label_cols)], axis=1)
    sub.to_csv(out, index=False)
    sub.to_csv(r"C:\Users\YKSHb\Desktop\submission_I065_ensemble.csv", index=False)
    print(f"\n  Written: {out}")
    print(f"  Desktop: submission_I065_ensemble.csv")
    print(f"  Pred mean card: {y_pred.sum(1).mean():.2f} (train truth 25.42)")
    print(f"  Total positives: {y_pred.sum():,d}")
    print(f"\nCompleted in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
