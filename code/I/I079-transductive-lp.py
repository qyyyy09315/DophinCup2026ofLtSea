"""
I079 — 转导标签传播作第三支, 带「去掉验证节点」反例对照
======================================================
信息来源: ESM2 嵌入图 (与 DIAMOND 的 E 值截断比对不是同一张图).
必做对照: 归纳式 (验证只检索训练) vs 转导式 (训练+验证一起构图, 含验证-验证边).
若两者融合分接近, 说明收益来自 kNN 检索而非转导, 立即关闭.

构图: L2 归一化后 GPU 批余弦 top-k (k=20), 边权 = max(cos, 0).
传播: Zhou 2003  F ← α S F + (1-α) Y, S 对称归一化, 30 次, 验证行再逐列 z-score.
融合: 接入 I064 full_pipe, 替换 sim 或与 sim 混合.
"""
import os, io, json, time
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
REPORT = os.path.join(REL_DIR, "I079_report.json")
N_LABELS = 500
W_PLM = 0.35
BASE_F1 = 0.4475
K = 20
ITERS = 30
ALPHAS = [0.5, 0.8, 0.99]
BATCH = 1024

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe


def gpu_knn(Q, Kmat, k, exclude_self=False, q_offset=0, k_offset=0):
    """Q (nq,d) Kmat (nk,d) both L2-normalized torch cuda -> idx, sim cpu int64/float32"""
    nq = Q.shape[0]
    idx = np.zeros((nq, k), np.int64)
    sim = np.zeros((nq, k), np.float32)
    for s in range(0, nq, BATCH):
        e = min(s + BATCH, nq)
        sc = Q[s:e] @ Kmat.T
        if exclude_self:
            for i in range(e - s):
                j = s + i + q_offset - k_offset
                if 0 <= j < Kmat.shape[0]:
                    sc[i, j] = -1.0
        sv, ix = torch.topk(sc, k, dim=1)
        idx[s:e] = ix.cpu().numpy()
        sim[s:e] = sv.cpu().numpy()
    return idx, sim


def knn_to_sparse(idx, sim, n):
    rows, cols, data = [], [], []
    for i in range(idx.shape[0]):
        for t, w in zip(idx[i], sim[i]):
            if w <= 0:
                continue
            b = int(t)
            if i == b:
                continue
            rows.append(i); cols.append(b); data.append(float(w))
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    return 0.5 * (W + W.T)


def spread(W, Y, alpha, iters=ITERS):
    d = np.array(W.sum(1)).ravel()
    dinv = np.power(np.maximum(d, 1e-12), -0.5)
    D = sparse.diags(dinv)
    S = D @ W @ D
    F = Y.copy()
    a, b = float(alpha), 1.0 - float(alpha)
    for _ in range(iters):
        F = a * (S @ F) + b * Y
    return F


def col_zscore(M):
    mu = M.mean(0, keepdims=True)
    sd = np.maximum(M.std(0, keepdims=True), 1e-9)
    return ((M - mu) / sd).astype(np.float32)


def weighted_knn_label(idx, sim, y_src):
    n, k = idx.shape
    lab = y_src[idx.ravel()].reshape(n, k, -1).astype(np.float32)
    ww = np.maximum(sim, 1e-7)
    return (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]


def load_sim_old(train_df, idx_tr, idx_va, y_tr, n_va, val_pids, train_map, val_map):
    parse_aln, agg = None, None
    # local copies of I064 helpers
    def parse_aln(tsv, qids, idmap, ev=0.001, mx=5):
        idx = np.full((len(qids), mx), -1, np.int64)
        bits = np.zeros((len(qids), mx), np.float32)
        qp = {q: i for i, q in enumerate(qids)}
        with open(tsv) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 4:
                    continue
                q, t = p[0], p[1]
                if q not in qp or t not in idmap:
                    continue
                i = qp[q]
                try:
                    bs, e = float(p[2]), float(p[3])
                except ValueError:
                    continue
                if e > ev:
                    continue
                j = 0
                while j < mx and bits[i, j] > bs:
                    j += 1
                if j < mx:
                    idx[i, j:] = np.roll(idx[i, j:], 1)
                    bits[i, j:] = np.roll(bits[i, j:], 1)
                    idx[i, j] = idmap[t]
                    bits[i, j] = bs
        return idx, bits

    def agg(idx, w, ys, tk=3):
        n = idx.shape[0]
        valid = idx[:, :tk] >= 0
        lab = np.zeros((n, tk, ys.shape[1]), np.float32)
        for j in range(tk):
            lab[:, j] = ys[idx[:, j].clip(min=0)]
        lab[~valid] = 0.0
        ww = np.maximum(w[:, :tk], 1e-7)
        ww[~valid] = 0.0
        return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)

    idx_v, bits_v = parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                              val_pids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim1 = agg(idx_v, bits_v, y_tr)
    v2v = {}
    with open(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            q, t = p[0], p[1]
            if q == t or q not in val_map or t not in val_map:
                continue
            qi, ti = val_map[q], val_map[t]
            v2v.setdefault(qi, [])
            if len(v2v[qi]) < 5:
                v2v[qi].append((ti, float(p[2])))
    sim2 = sim1.copy()
    for qi in range(n_va):
        if miss[qi] and qi in v2v:
            acc = np.zeros(N_LABELS, np.float32)
            aw = 0.0
            for ti, bs in v2v[qi]:
                if hit[ti]:
                    acc += sim1[ti] * bs
                    aw += bs
            if aw > 0:
                sim2[qi] = acc / aw
    kmer = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        pre = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{pre}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_tr[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        sm = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(sm, 1e-7) ** 2
        kmer += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return 0.4 * sim2 + 0.6 * kmer


def main():
    t0 = time.time()
    print("=" * 92)
    print("  I079: Transductive LP vs inductive kNN (ESM2 graph)")
    print("=" * 92)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_tr, n_va = len(idx_tr), len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}

    cnn = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    plm = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn + 0.5 * c005)
    sim_old = load_sim_old(train_df, idx_tr, idx_va, y_tr, n_va, val_pids, train_map, val_map)

    print("  loading ESM2 embeddings ...", flush=True)
    df_e = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    pid = train_df["protein_id"].values
    # align rows to train.csv order
    if not np.array_equal(df_e["protein_id"].values, pid):
        df_e = df_e.set_index("protein_id").loc[pid].reset_index()
    ec = [c for c in df_e.columns if c.startswith("emb_")]
    E = df_e[ec].values.astype(np.float32)
    Etr, Eva = E[idx_tr], E[idx_va]
    def l2(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, 1e-9)
    Etr, Eva = l2(Etr), l2(Eva)
    n_all = n_tr + n_va
    Eall = np.concatenate([Etr, Eva], axis=0)

    device = "cuda"
    Qt = torch.from_numpy(Eall).to(device)
    print(f"  GPU kNN transductive k={K} on {n_all} nodes ...", flush=True)
    idx_all, sim_all = gpu_knn(Qt, Qt, K, exclude_self=True)
    print(f"  mean edge cos={sim_all.mean():.4f}")

    print("  GPU kNN inductive val->train ...", flush=True)
    Qv = torch.from_numpy(Eva).to(device)
    Kt = torch.from_numpy(Etr).to(device)
    idx_ind, sim_ind = gpu_knn(Qv, Kt, K, exclude_self=False)
    knn_ind = weighted_knn_label(idx_ind, sim_ind, y_tr)
    knn_ind = col_zscore(knn_ind)
    print(f"  inductive standalone F1={macro_f1(knn_ind, y_va, search_scut(knn_ind, y_va)):.4f}")

    W = knn_to_sparse(idx_all, sim_all, n_all)
    Y = np.zeros((n_all, N_LABELS), np.float64)
    Y[:n_tr] = y_tr.astype(np.float64)
    print(f"  graph nnz={W.nnz} ... spreading", flush=True)

    lp_scores = {}
    for a in ALPHAS:
        F = spread(W, Y, a)
        sc = col_zscore(F[n_tr:])
        sf1 = macro_f1(sc, y_va, search_scut(sc, y_va))
        lp_scores[a] = sc
        print(f"  LP α={a:.2f} standalone F1={sf1:.4f}", flush=True)

    variants = [("V0 现状同源支", sim_old),
                ("V1 归纳ESM-kNN 替换", knn_ind),
                ("V2 转导LP α=0.80 替换", lp_scores[0.8]),
                ("V3 转导LP α=0.50 替换", lp_scores[0.5]),
                ("V4 转导LP α=0.99 替换", lp_scores[0.99]),
                ("V5 0.5现状+0.5转导α0.80", 0.5 * sim_old + 0.5 * lp_scores[0.8]),
                ("V6 0.5现状+0.5归纳kNN", 0.5 * sim_old + 0.5 * knn_ind)]

    print("\n" + "=" * 92)
    print(f"{'配置':32s} | {'独立F1':>8s} | {'融合F1':>8s} | {'AUC':>8s} | {'d融合':>8s}")
    print("-" * 92)
    res = {}
    for name, sim in variants:
        sf1 = macro_f1(sim, y_va, search_scut(sim, y_va))
        f1, auc, _, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=plm, w=W_PLM)
        res[name] = {"standalone_f1": float(sf1), "fused_f1": float(f1),
                     "auc": float(auc), "d": float(f1 - BASE_F1)}
        print(f"{name:32s} | {sf1:8.4f} | {f1:8.4f} | {auc:8.4f} | {f1 - BASE_F1:+8.4f}", flush=True)

    d_lp = res["V2 转导LP α=0.80 替换"]["fused_f1"] - res["V1 归纳ESM-kNN 替换"]["fused_f1"]
    print(f"\n  转导 − 归纳 (替换融合) = {d_lp:+.4f}")
    if abs(d_lp) < 0.001 and res["V5 0.5现状+0.5转导α0.80"]["fused_f1"] < BASE_F1 + 0.0005:
        verdict = "转导无新信息, 关闭"
    elif res["V5 0.5现状+0.5转导α0.80"]["fused_f1"] > BASE_F1 + 0.0005:
        verdict = "转导混合有增益"
    else:
        verdict = "转导相对归纳无实质差异, 关闭"
    print(f"  判定: {verdict}")

    np.savez_compressed(os.path.join(REL_DIR, "i079_lp_a080.npz"), val=lp_scores[0.8])
    np.savez_compressed(os.path.join(REL_DIR, "i079_ind_knn.npz"), val=knn_ind)
    out = {"results": res, "d_transductive_minus_inductive": float(d_lp),
           "verdict": verdict, "k": K, "iters": ITERS, "alphas": ALPHAS}
    with io.open(REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
