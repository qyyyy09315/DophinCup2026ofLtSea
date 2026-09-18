"""
I066b — 阈值居中校准的连续概率提交 (双保险格式)
================================================
背景推论 (已用公开榜数据反解验证):
  平台 AUC 由提交值的排序直接计算。二值提交时 AUC 被压缩为 0.5+(TPR-FPR)/2 ≈ 0.713；
  榜首 0.875519 与我们本地连续概率 AUC 0.8609~0.8647 同量级 => 榜首提交的是浮点概率。
风险: 若平台用固定 0.5 卡阈值算 Macro F1，而我们原始分数整体偏小(均值~0.1)，
      则 (s>=0.5) 会严重欠预测 (实测 F1 仅 0.2426)。
方案: 逐标签单调校准 (Threshold-Centered Calibration)
      p'_ij = 1 / (1 + exp(-(s_ij - t_j)/w_j))
      其中 t_j 为该标签的 SCut 最优阈值，w_j 为该标签分数的稳健尺度。
  => 保序: 逐标签 AUC 不变 => 宏平均 AUC 不变(直通 0.86+)
  => 0.5 恰好对应最优阈值 => 若平台卡 0.5，Macro F1 即取到最优
用法: python code/I/I066b-calibrated-prob.py
输出: submit/submission_I066b_prob_calibrated.csv + 桌面副本
      rel/I066b_report.json
"""
import os, sys, time, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
REPORT = os.path.join(REL_DIR, "I066b_report.json")
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
        tp = (pr & ytm).sum(0); fp = (pr & ~ytm).sum(0); fn = (~pr & ytm).sum(0)
        best[i] = cands[(2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)).argmax()]
    return best


def macro_f1(probs, y, ths):
    preds = (probs >= ths).astype(bool)
    yb = y.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
    p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    return float(np.mean(np.where(p + r > 0, 2 * p * r / (p + r), 0.0)))


def calibrate_center(S, ths, scale_frac=0.35):
    """逐标签单调校准: 使 S==ths 映射到 0.5，保序。"""
    out = np.empty_like(S, dtype=np.float64)
    for j in range(S.shape[1]):
        col = S[:, j].astype(np.float64)
        # 稳健尺度: 该列 5%-95% 分位跨度
        q05, q95 = np.percentile(col, [5, 95])
        w = max((q95 - q05) * scale_frac, 1e-6)
        out[:, j] = 1.0 / (1.0 + np.exp(-np.clip((col - ths[j]) / w, -30, 30)))
    return out


def main():
    t0 = time.time()
    print("=" * 72)
    print("  I066b: Threshold-Centered Calibrated Probability Submission")
    print("=" * 72)

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

    # ---------- 验证集 ----------
    print("\n[1/3] Validation: computing calibrated probabilities...")
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

    s_v = deep5_v + 0.75 * sim_v + 0.50 * col_norm(1.0/(30+ranks(deep5_v)+1) + 1.0/(30+ranks(sim_v)+1))
    t_tmp = search_scut(s_v, y_va)
    row_m = np.maximum((s_v >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_tr.mean(0) * n_va, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker_v = np.exp((s_v - s_v.max(axis=1, keepdims=True)) / 0.20)
    p_v = sinkhorn(ker_v, row_m, col_m)
    mixed_v = 0.40 * s_v + 0.60 * p_v

    ths = search_scut(mixed_v, y_va)
    f1_scut = macro_f1(mixed_v, y_va, ths)
    auc_raw = roc_auc_score(y_va, mixed_v, average="macro")
    f1_at_half_raw = macro_f1(mixed_v, y_va, np.full(N_LABELS, 0.5))
    print(f"  Raw scores : SCut F1={f1_scut:.4f} | AUC={auc_raw:.4f} | F1@0.5={f1_at_half_raw:.4f}")

    cal_v = calibrate_center(mixed_v, ths)
    auc_cal = roc_auc_score(y_va, cal_v, average="macro")
    f1_cal_half = macro_f1(cal_v, y_va, np.full(N_LABELS, 0.5))
    f1_cal_scut = macro_f1(cal_v, y_va, search_scut(cal_v, y_va))
    print(f"  Calibrated : SCut F1={f1_cal_scut:.4f} | AUC={auc_cal:.4f} | F1@0.5={f1_cal_half:.4f}")
    print(f"  => 保序检验: AUC 差 {abs(auc_cal-auc_raw):.6f} (应为 0)")
    print(f"  => 0.5 卡阈值 F1 从 {f1_at_half_raw:.4f} 提升到 {f1_cal_half:.4f}")

    # ---------- 测试集 ----------
    print("\n[2/3] Test: building calibrated probability matrix...")
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
    sim_t = 0.4 * sim2_t + 0.6 * build_kmers(y_full, "test")

    s_t = deep5_t + 0.75 * sim_t + 0.50 * col_norm(1.0/(30+ranks(deep5_t)+1) + 1.0/(30+ranks(sim_t)+1))
    row_t = np.maximum((s_t >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_t = np.maximum(y_full.mean(0) * n_te, 1e-6)
    col_t *= row_t.sum() / col_t.sum()
    ker_t = np.exp((s_t - s_t.max(axis=1, keepdims=True)) / 0.20)
    p_t = sinkhorn(ker_t, row_t, col_t)
    mixed_t = 0.40 * s_t + 0.60 * p_t

    cal_t = calibrate_center(mixed_t, ths)
    print(f"  Calibrated test matrix: {cal_t.shape}, range [{cal_t.min():.4f}, {cal_t.max():.4f}], mean {cal_t.mean():.4f}")

    # ---------- 写出 ----------
    print("\n[3/3] Writing submissions...")
    # A. 校准概率 (双保险)
    sub_prob = pd.concat([test_df[["protein_id"]],
                          pd.DataFrame(np.round(cal_t, 4), columns=label_cols)], axis=1)
    out_p = os.path.join(SUB_DIR, "submission_I066b_prob_calibrated.csv")
    sub_prob.to_csv(out_p, index=False)
    sub_prob.to_csv(r"C:\Users\YKSHb\Desktop\submission_I066b_prob_calibrated.csv", index=False)
    print(f"  [双保险] {out_p}")

    # B. 二值 (对照)
    y_bin = (cal_t >= 0.5).astype(np.int8)
    sub_bin = pd.concat([test_df[["protein_id"]],
                         pd.DataFrame(y_bin, columns=label_cols)], axis=1)
    out_b = os.path.join(SUB_DIR, "submission_I066b_binary.csv")
    sub_bin.to_csv(out_b, index=False)
    sub_bin.to_csv(r"C:\Users\YKSHb\Desktop\submission_I066b_binary.csv", index=False)
    print(f"  [二值]   {out_b}")
    print(f"  Pred mean card (binary): {y_bin.sum(1).mean():.2f} (train truth 25.42)")

    rep = {
        "raw_scut_f1": f1_scut, "raw_auc": auc_raw, "raw_f1_at_0.5": f1_at_half_raw,
        "calibrated_auc": auc_cal, "calibrated_f1_at_0.5": f1_cal_half,
        "calibrated_scut_f1": f1_cal_scut,
        "auc_preserved": bool(abs(auc_cal - auc_raw) < 1e-6),
        "test_mean_prob": float(cal_t.mean()), "test_mean_card_binary": float(y_bin.sum(1).mean()),
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
