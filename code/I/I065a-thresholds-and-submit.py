"""
I065a — 生成 I064 最优配置阈值并运行 I065 提交
==============================================
配置: multi-seed 1664D PLM-MLP (seeds 11/22/33) + 2-hop 传递同源 + RRF + Sinkhorn, w_plm=0.35
资产:
  rel/plm1664_multiseed_val.npz   (已有, I064 产出)
  rel/plm1664_multiseed_test.npz  (已有, 多种子测试集概率)
  rel/i064_best_ths.npy           (本脚本产出)
输出:
  submit/submission_I065_ensemble.csv + 桌面副本
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
N_LABELS = 500
W_PLM = 0.35


def parse_aln(tsv, qids, idmap, ev=0.001, mx=5):
    idx = np.full((len(qids), mx), -1, np.int64)
    bits = np.zeros((len(qids), mx), np.float32)
    qp = {q: i for i, q in enumerate(qids)}
    with open(tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
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


def transitive_2hop(sim1, hit, miss, nbr, n):
    out = sim1.copy()
    for qi in range(n):
        if miss[qi] and qi in nbr:
            acc = np.zeros(N_LABELS, np.float32)
            aw = 0.0
            for ti, bs in nbr[qi]:
                if hit[ti]:
                    acc += sim1[ti] * bs
                    aw += bs
            if aw > 0:
                out[qi] = acc / aw
    return out


def build_kmers(y_tr, split):
    """split='val' 或 'test'"""
    s = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        pre = "CK003" if k >= 7 else "CK002"
        if split == "val":
            d = np.load(os.path.join(REL_DIR, f"{pre}_neigh_k{k}.npz"))
            idx = d["idx_val"][:, :5]
            sm = d["sim_val"][:, :5].astype(np.float32)
        else:
            f = "TCK002a_neigh_full.npz" if k == 6 else f"CK004_neigh_k{k}_full.npz"
            d = np.load(os.path.join(REL_DIR, f))
            idx = d["idx_test"][:, :5]
            sm = d["sim_test"][:, :5].astype(np.float32)
        n = idx.shape[0]
        lab = y_tr[idx.ravel()].reshape(n, 5, -1).astype(np.float32)
        ww = np.maximum(sm, 1e-7) ** 2
        s += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return s


def ranks(s):
    return np.argsort(np.argsort(-s, axis=1, kind="mergesort"), axis=1)


def col_norm(x):
    lo, hi = x.min(axis=0), x.max(axis=0)
    return (x - lo) / np.maximum(hi - lo, 1e-9)


def sinkhorn(K, r, c, it=100, tol=1e-5):
    u = np.ones_like(r)
    v = np.ones_like(c)
    for _ in range(it):
        un = r / np.maximum(K @ v, 1e-12)
        vn = c / np.maximum(K.T @ un, 1e-12)
        if np.max(np.abs(un - u)) < tol and np.max(np.abs(vn - v)) < tol:
            break
        u, v = un, vn
    P = u[:, None] * K * v[None, :]
    lo, hi = P.min(axis=0), P.max(axis=0)
    return (P - lo) / np.maximum(hi - lo, 1e-9)


def search_scut(ens, y, n=100):
    cands = np.linspace(0.01, 0.99, n)
    best = np.full(N_LABELS, 0.5)
    for i in range(N_LABELS):
        yt = y[:, i]
        if yt.sum() == 0:
            continue
        pr = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0)
        fp = (pr & ~ytm).sum(0)
        fn = (~pr & ytm).sum(0)
        best[i] = cands[(2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)).argmax()]
    return best


def macro_f1(probs, y, ths):
    preds = (probs >= ths).astype(bool)
    yb = y.astype(bool)
    tp = (preds & yb).sum(0)
    fp = (preds & ~yb).sum(0)
    fn = (~preds & yb).sum(0)
    p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    return float(np.mean(np.where(p + r > 0, 2 * p * r / (p + r), 0.0)))


def rrf_terms(a, b, k=30):
    return 1.0 / (k + ranks(a) + 1) + 1.0 / (k + ranks(b) + 1)


def main():
    t0 = time.time()
    print("=" * 70)
    print(f"  I065: multi-seed 1664D PLM-MLP + 2-hop, w={W_PLM}")
    print("=" * 70)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va, n_te = len(idx_va), len(test_df)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    full_map = {p: i for i, p in enumerate(train_df["protein_id"].values)}

    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]

    # ---------- 验证集: 求阈值 ----------
    print("\n[1/2] Validation: computing SCut thresholds...")
    cnn_v = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1v = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2v = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    deep4_v = wc["t001a"] * t1v + wc["t002"] * t2v + wc["cnn"] * (0.5 * cnn_v + 0.5 * c005_v)
    multi_v = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    deep5_v = (1 - W_PLM) * deep4_v + W_PLM * multi_v

    idx_v, bits_v = parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                              val_pids, train_map)
    hit_v = idx_v[:, 0] >= 0
    miss_v = ~hit_v
    sim1_v = agg(idx_v, bits_v, y_tr)
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
    sim2_v = transitive_2hop(sim1_v, hit_v, miss_v, v2v, n_va)
    sim_v = 0.4 * sim2_v + 0.6 * build_kmers(y_tr, "val")

    s_v = deep5_v + 0.75 * sim_v + 0.50 * col_norm(rrf_terms(deep5_v, sim_v, 30))

    # 复算: 求 SCut 阈值与验证指标
    t_tmp = search_scut(s_v, y_va)
    row_m = np.maximum((s_v >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_tr.mean(0) * n_va, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker_v = np.exp((s_v - s_v.max(axis=1, keepdims=True)) / 0.20)
    p_v = sinkhorn(ker_v, row_m, col_m)
    mixed_v = 0.40 * s_v + 0.60 * p_v
    ths = search_scut(mixed_v, y_va)
    val_f1 = macro_f1(mixed_v, y_va, ths)
    val_auc = roc_auc_score(y_va, mixed_v, average="macro")
    np.save(os.path.join(REL_DIR, "i064_best_ths.npy"), ths)
    print(f"  Val F1 = {val_f1:.4f} | AUC = {val_auc:.4f} | (C011a 0.4453, I060 0.4471)")
    print(f"  Thresholds saved (mean {ths.mean():.4f})")

    # ---------- 测试集: 生成提交 ----------
    print("\n[2/2] Test: building submission...")
    c005_t = np.load(os.path.join(REL_DIR, "C005_test_probs.npz"))["test"]
    cnn_t = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
    t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"]
    t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
    deep4_t = wc["t001a"] * t1t + wc["t002"] * t2t + wc["cnn"] * (0.5 * cnn_t + 0.5 * c005_t)
    multi_t = np.load(os.path.join(REL_DIR, "plm1664_multiseed_test.npz"))["test"]
    deep5_t = (1 - W_PLM) * deep4_t + W_PLM * multi_t

    idx_t, bits_t = parse_aln(os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv"),
                              test_df["protein_id"].values, full_map)
    hit_t = idx_t[:, 0] >= 0
    miss_t = ~hit_t
    sim1_t = agg(idx_t, bits_t, y_full)
    test_map = {p: i for i, p in enumerate(test_df["protein_id"].values)}
    te2te = {}
    with open(os.path.join(CK005A_TMP, "align_test_to_test_e0.001.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            q, t = p[0], p[1]
            if q == t or q not in test_map or t not in test_map:
                continue
            qi, ti = test_map[q], test_map[t]
            te2te.setdefault(qi, [])
            if len(te2te[qi]) < 5:
                te2te[qi].append((ti, float(p[2])))
    sim2_t = transitive_2hop(sim1_t, hit_t, miss_t, te2te, n_te)
    n_rescued = int(np.sum(np.any(sim2_t != sim1_t, axis=1)))
    print(f"  Test 2-hop rescued {n_rescued} orphans (of {int(miss_t.sum())})")
    sim_t = 0.4 * sim2_t + 0.6 * build_kmers(y_full, "test")

    s_t = deep5_t + 0.75 * sim_t + 0.50 * col_norm(rrf_terms(deep5_t, sim_t, 30))
    row_t = np.maximum((s_t >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_t = np.maximum(y_full.mean(0) * n_te, 1e-6)
    col_t *= row_t.sum() / col_t.sum()
    ker_t = np.exp((s_t - s_t.max(axis=1, keepdims=True)) / 0.20)
    p_t = sinkhorn(ker_t, row_t, col_t)
    mixed_t = 0.40 * s_t + 0.60 * p_t
    y_pred = (mixed_t >= ths).astype(np.int8)

    sub = pd.concat([test_df[["protein_id"]], pd.DataFrame(y_pred, columns=label_cols)], axis=1)
    out = os.path.join(SUB_DIR, "submission_I065_ensemble.csv")
    sub.to_csv(out, index=False)
    sub.to_csv(r"C:\Users\YKSHb\Desktop\submission_I065_ensemble.csv", index=False)
    print(f"  Written: {out}")
    print(f"  Desktop: submission_I065_ensemble.csv")
    print(f"  Pred mean card: {y_pred.sum(1).mean():.2f} (train truth 25.42)")
    print(f"  Total positives: {y_pred.sum():,d}")

    rep = {"w_plm": W_PLM, "val_f1": val_f1, "val_auc": val_auc,
           "delta_vs_c011a": val_f1 - 0.4453, "delta_vs_i060": val_f1 - 0.4471,
           "test_rescued_orphans": n_rescued}
    with Path(os.path.join(REL_DIR, "I065_report.json")).open("w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"\nCompleted in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
