"""
I063 — 2176D 多种子 PLM-MLP 接入流水线 + 权重扫描 + 提交生成
=============================================================
I062 分支: 2176D (ESM2-640 + ProtTrans-1024 + R001-BiLSTM-512) 三种子残差 MLP 集成
  独立 SCut F1 = 0.2867 (较 I060 的 1664D 单种子 0.2590 提升 +0.0277), AUC 0.7857
本实验: 2-hop 同源 + 该分支 接入 deep 路 w 扫描 -> RRF -> Sinkhorn -> 生成提交
"""
import os, sys, time, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
REPORT = os.path.join(REL_DIR, "I063_report.json")
N_LABELS = 500


def parse_aln(tsv, qids, idmap, evalue=0.001, max_accept=5):
    idx = np.full((len(qids), max_accept), -1, np.int64)
    bits = np.zeros((len(qids), max_accept), np.float32)
    qpos = {q: i for i, q in enumerate(qids)}
    with open(tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            q, t = p[0], p[1]
            if q not in qpos or t not in idmap: continue
            i = qpos[q]
            try: bs, ev = float(p[2]), float(p[3])
            except ValueError: continue
            if ev > evalue: continue
            j = 0
            while j < max_accept and bits[i, j] > bs: j += 1
            if j < max_accept:
                idx[i, j:] = np.roll(idx[i, j:], 1)
                bits[i, j:] = np.roll(bits[i, j:], 1)
                idx[i, j] = idmap[t]
                bits[i, j] = bs
    return idx, bits


def agg(idx, w, yspace, top_k=3):
    n = idx.shape[0]
    valid = idx[:, :top_k] >= 0
    lab = np.zeros((n, top_k, yspace.shape[1]), np.float32)
    for j in range(top_k): lab[:, j] = yspace[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :top_k], 1e-7); ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)


def transitive_2hop(sim1, hit, miss, v2v, n):
    sim2 = sim1.copy()
    for qi in range(n):
        if miss[qi] and qi in v2v:
            acc = np.zeros(N_LABELS, np.float32); aw = 0.0
            for ti, bs in v2v[qi]:
                if hit[ti]:
                    acc += sim1[ti] * bs; aw += bs
            if aw > 0: sim2[qi] = acc / aw
    return sim2


def build_kmer(y_tr):
    s = 0.0
    for k, wt in zip([3,4,5,6,7,8], [0,0.1,0.2,0.4,0.2,0.1]):
        if wt <= 0: continue
        pre = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{pre}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_tr[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        sm = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(sm, 1e-7) ** 2
        s += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return s


def ranks(s): return np.argsort(np.argsort(-s, axis=1, kind="mergesort"), axis=1)
def rrf_pair(d, s, k=30): return 1.0/(k+ranks(d)+1) + 1.0/(k+ranks(s)+1)
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


def search_scut(ens, y):
    cands = np.linspace(0.01, 0.99, 100)
    best = np.full(N_LABELS, 0.5)
    for i in range(N_LABELS):
        yt = y[:, i]
        if yt.sum() == 0: continue
        pr = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0); fp = (pr & ~ytm).sum(0); fn = (~pr & ytm).sum(0)
        best[i] = cands[(2*tp/np.maximum(2*tp+fp+fn,1e-9)).argmax()]
    return best


def macro_f1(probs, y, ths):
    preds = (probs >= ths).astype(bool); yb = y.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
    p = np.where(tp+fp>0, tp/(tp+fp), 0.0); r = np.where(tp+fn>0, tp/(tp+fn), 0.0)
    f = np.where(p+r>0, 2*p*r/(p+r), 0.0)
    return float(np.mean(f))


def main():
    gen_sub = True
    t0 = time.time()
    print("=" * 70)
    print("  I063: 2176D Multi-Seed PLM-MLP into Full Pipeline")
    print("=" * 70)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va, n_te = len(idx_va), len(test_df)
    val_pids = train_df["protein_id"].values[idx_va]
    train_pids = train_df["protein_id"].values[idx_tr]
    train_map = {p: i for i, p in enumerate(train_pids)}
    val_map = {p: i for i, p in enumerate(val_pids)}

    # 深度 4 路
    cnn_v = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        w_cfg = json.load(f)["best_w"]
    deep4 = w_cfg["t001a"]*t1 + w_cfg["t002"]*t2 + w_cfg["cnn"]*(0.5*cnn_v+0.5*c005_v)

    plm_v = np.load(os.path.join(REL_DIR, "multi_plm_mlp_val_probs.npz"))["val"]

    # 同源
    idx_v, bits_v = parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                              val_pids, train_map)
    hit = idx_v[:, 0] >= 0; miss = ~hit
    sim1 = agg(idx_v, bits_v, y_tr)

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
    sim2 = transitive_2hop(sim1, hit, miss, v2v, n_va)
    kmer = build_kmer(y_tr)
    sim_full = 0.4*sim2 + 0.6*kmer

    def pipeline(deep_v):
        s = deep_v + 0.75*sim_full + 0.50*col_norm(rrf_pair(deep_v, sim_full, 30))
        t_tmp = search_scut(s, y_va)
        row_m = np.maximum((s >= t_tmp).sum(1).astype(np.float64), 1.0)
        col_m = np.maximum(y_tr.mean(0)*n_va, 1e-6); col_m *= row_m.sum()/col_m.sum()
        ker = np.exp((s - s.max(axis=1, keepdims=True))/0.20)
        p_v = sinkhorn(ker, row_m, col_m)
        mixed = 0.40*s + 0.60*p_v
        ths = search_scut(mixed, y_va)
        return macro_f1(mixed, y_va, ths), roc_auc_score(y_va, mixed, average="macro"), ths

    print(f"\n{'w_plm':>6s} | {'Val F1':>8s} | {'AUC':>8s} | {'vs C011a':>9s} | {'vs I060':>9s}")
    print("-" * 60)
    best_f1, best_w = -1, 0.25
    for w in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        dv = (1-w)*deep4 + w*plm_v
        f1, auc, _ = pipeline(dv)
        print(f"{w:6.2f} | {f1:8.4f} | {auc:8.4f} | {f1-0.4453:+9.4f} | {f1-0.4471:+9.4f}")
        if f1 > best_f1:
            best_f1, best_w = f1, w

    print(f"\n  BEST: w_plm={best_w:.2f} -> Val F1={best_f1:.4f} (C011a 0.4453, I060 0.4471)")

    # 生成提交
    if gen_sub:
        print(f"\n--- Generating submission with w_plm={best_w:.2f} ---")
        c005_t = np.load(os.path.join(REL_DIR, "C005_test_probs.npz"))["test"]
        cnn_t = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
        t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"]
        t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
        deep4_t = w_cfg["t001a"]*t1t + w_cfg["t002"]*t2t + w_cfg["cnn"]*(0.5*cnn_t+0.5*c005_t)
        plm_t = np.load(os.path.join(REL_DIR, "multi_plm_mlp_test_probs.npz"))["test"]
        deep5_t = (1-best_w)*deep4_t + best_w*plm_t

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
        sim2_t = transitive_2hop(sim1_t, hit_t, miss_t, te2te, n_te)

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

        s_t = deep5_t + 0.75*sim_t + 0.50*col_norm(rrf_pair(deep5_t, sim_t, 30))
        row_t = np.maximum((s_t >= t_tmp).sum(1).astype(np.float64), 1.0)
        col_t = np.maximum(y_full.mean(0)*n_te, 1e-6); col_t *= row_t.sum()/col_t.sum()
        ker_t = np.exp((s_t - s_t.max(axis=1, keepdims=True))/0.20)
        p_t = sinkhorn(ker_t, row_t, col_t)
        mixed_t = 0.40*s_t + 0.60*p_t

        # 复用最优阈值
        _, _, ths_best = pipeline((1-best_w)*deep4 + best_w*plm_v)
        y_pred = (mixed_t >= ths_best).astype(np.int8)

        out = os.path.join(SUB_DIR, "submission_I063_ensemble.csv")
        pd.concat([test_df[["protein_id"]], pd.DataFrame(y_pred, columns=label_cols)],
                  axis=1).to_csv(out, index=False)
        pd.concat([test_df[["protein_id"]], pd.DataFrame(y_pred, columns=label_cols)],
                  axis=1).to_csv(r"C:\Users\YKSHb\Desktop\submission_I063_ensemble.csv", index=False)
        print(f"  Written: {out}")
        print(f"  Copied to Desktop: submission_I063_ensemble.csv")
        print(f"  Pred mean cardinality: {y_pred.sum(1).mean():.2f}")

    with Path(REPORT).open("w", encoding="utf-8") as f:
        json.dump({"best_w_plm": best_w, "val_f1": best_f1,
                   "delta_vs_c011a": best_f1-0.4453, "delta_vs_i060": best_f1-0.4471}, f, indent=2)
    print(f"\nCompleted in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
