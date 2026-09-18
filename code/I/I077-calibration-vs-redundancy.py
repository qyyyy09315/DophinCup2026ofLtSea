"""
I077 — 判定「标定错位 vs 信息冗余」: 分支分数标定对融合的影响
==============================================================
背景矛盾: I073 倾向加权让 PLM 头独立 F1 上升 0.0088, I074 却让融合 F1 下降 0.0014。
I075 给出的"头部集合冗余"机制效应量很小且 I076 在标签级被否决 (dF1_l vs overlap_l r=+0.3014,
符号与冗余假设相反; 只有 rho_l 的 r=-0.1273 是弱正确符号)。
I076 节 4 还发现: 融合增益随标签频率单调上升 (+0.0113 -> +0.0211), 与冗余也同步上升,
说明**增益由分支绝对强度驱动, 不是由去相关驱动**。

新假设 (本实验判定): 失败原因不是信息冗余, 而是**分支分数标定错位**。
融合链路里 d = (1-w)*deep4 + w*plm 是**原始分值的线性混合**。deep4 与 plm 的分值尺度不同
(概率分布形状不同), 而下游有两处对尺度敏感:
  ① RRF 项用 col_norm(逐标签 min-max), 对异常值敏感;
  ② Sinkhorn 核 ker = exp((s - rowmax(s))/0.20) —— **逐行归一化**,
     若 plm 抬高了稀有标签分数, 该行的 rowmax 与会占据核质量的标签都会改变,
     传输计划的边际被别的标签吸走, 融合秩序变差。
若假设成立, 则**用秩融合 (RRF, 尺度无关) 或逐标签标准化后再混合**, 倾向加权的独立增益应当传导。

6 个变体 (全用 I073 存的 L0/L1, 同种子42/同架构/仅损失不同, 受控对照):
  V1 线性原始分 + plm_asl        (基线口径)
  V2 线性原始分 + plm_prop       (复现 I074 负结果)
  V3 逐标签 min-max 后线性混合 + plm_asl
  V4 逐标签 min-max 后线性混合 + plm_prop    <- 若 V4>V3, 标定错位假设成立
  V5 逐标签 z-score 后线性混合 + plm_prop    <- 另一种标定
  V6 纯 RRF 秩融合 + plm_prop                <- 尺度无关的极端情形
判据: 融合 Macro F1 与 R@FPR0.069。差异应集中在头部 (对比 R@FPR 列)。
"""
import os, io, json, time
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc as sk_auc

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT = os.path.join(REL_DIR, "I077_report.json")
N_LABELS = 500
FPR_GRID = [0.02, 0.05, 0.069, 0.10]
W_PLM = 0.35

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
macro_f1, search_scut, ranks, col_norm, sinkhorn = (
    i064.macro_f1, i064.search_scut, i064.ranks, i064.col_norm, i064.sinkhorn)


def zscore_cols(M):
    return (M - M.mean(0, keepdims=True)) / np.maximum(M.std(0, keepdims=True), 1e-9)


def head_metrics(S, y):
    aucs, paucs = [], []
    rec = {f: [] for f in FPR_GRID}
    for j in range(y.shape[1]):
        yt = y[:, j]
        if yt.min() == yt.max(): continue
        fpr, tpr, _ = roc_curve(yt, S[:, j])
        if len(fpr) < 2: continue
        aucs.append(sk_auc(fpr, tpr))
        m = fpr <= 0.10
        paucs.append(sk_auc(fpr[m], tpr[m]) / 0.10 if m.sum() >= 2 else 0.0)
        for f in FPR_GRID:
            k = np.searchsorted(fpr, f, side="right") - 1
            rec[f].append(float(tpr[max(k, 0)]))
    return {"auc": float(np.mean(aucs)), "pauc": float(np.mean(paucs)),
            **{f"r@{f}": float(np.mean(rec[f])) for f in FPR_GRID}}


def blend(deep4, plm, mode, w=W_PLM):
    """返回注入 PLM 后的 deep 分支。mode 决定标定方式。"""
    if mode == "raw":
        return (1 - w) * deep4 + w * plm
    if mode == "minmax":
        return (1 - w) * col_norm(deep4) + w * col_norm(plm)
    if mode == "zscore":
        return (1 - w) * zscore_cols(deep4) + w * zscore_cols(plm)
    if mode == "rrf":
        return col_norm(deep4) + w * col_norm(1.0 / (30 + ranks(plm) + 1))
    raise ValueError(mode)


def run(d, sim, y_tr, y_va, n_va):
    """I064 full_pipe 的后半段, 逐行对齐"""
    s = d + 0.75 * sim + 0.50 * col_norm(1.0 / (30 + ranks(d) + 1) + 1.0 / (30 + ranks(sim) + 1))
    t_tmp = search_scut(s, y_va)
    row_m = np.maximum((s >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_tr.mean(0) * n_va, 1e-6); col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s - s.max(axis=1, keepdims=True)) / 0.20)
    p_v = sinkhorn(ker, row_m, col_m)
    mixed = 0.40 * s + 0.60 * p_v
    return macro_f1(mixed, y_va, search_scut(mixed, y_va)), mixed


def main():
    t0 = time.time()
    print("=" * 108)
    print("  I077: Calibration-Misalignment vs Information-Redundancy")
    print("=" * 108)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}

    cnn = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    plm_asl = np.load(os.path.join(REL_DIR, "i073_L0_asl_val.npz"))["val"]
    plm_prop = np.load(os.path.join(REL_DIR, "i073_L1_asl_prop_val.npz"))["val"]
    with open(os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json"), encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn + 0.5 * c005)

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
        return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)

    ck = os.path.join(REL_DIR, "ck005a_tmp")
    idx_v, bits_v = parse_aln(os.path.join(ck, "align_val_very-sensitive_e0.001.tsv"), val_pids, train_map)
    hit = idx_v[:, 0] >= 0; miss = ~hit
    sim1 = agg(idx_v, bits_v, y_tr)
    v2v = {}
    with open(os.path.join(ck, "align_val_to_val_e0.001.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            q, t = p[0], p[1]
            if q == t or q not in val_map or t not in val_map: continue
            qi, ti = val_map[q], val_map[t]
            v2v.setdefault(qi, [])
            if len(v2v[qi]) < 5: v2v[qi].append((ti, float(p[2])))
    sim2 = sim1.copy()
    for qi in range(n_va):
        if miss[qi] and qi in v2v:
            acc = np.zeros(N_LABELS, np.float32); aw = 0.0
            for ti, bs in v2v[qi]:
                if hit[ti]: acc += sim1[ti] * bs; aw += bs
            if aw > 0: sim2[qi] = acc / aw
    kmer = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0: continue
        pre = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{pre}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_tr[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        sm = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(sm, 1e-7) ** 2
        kmer += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    sim = 0.4 * sim2 + 0.6 * kmer
    print(f"  val {n_va} x {N_LABELS} | deep4 / sim / plm 就绪")

    # 标定诊断: 两版 PLM 头的逐标签分数分布差异
    print("\n  标定诊断 (PLM 头):")
    for nm, P in [("plm_asl", plm_asl), ("plm_prop", plm_prop)]:
        print(f"    {nm:9s}: 全局 mean={P.mean():.5f} p99={np.quantile(P,0.99):.5f} max={P.max():.5f} | "
              f"逐标签行均值 mean={P.mean(1).mean():.5f} std={P.mean(1).std():.5f}")

    plan = [
        ("V1 线性原始分 + asl",        "raw",    plm_asl),
        ("V2 线性原始分 + prop",       "raw",    plm_prop),
        ("V3 min-max标定 + asl",       "minmax", plm_asl),
        ("V4 min-max标定 + prop",      "minmax", plm_prop),
        ("V5 z-score标定 + prop",      "zscore", plm_prop),
        ("V6 RRF秩融合 + prop",        "rrf",    plm_prop),
    ]
    print("\n" + "=" * 108)
    print(f"{'配置':26s} | {'融合F1':>7s} | {'dF1(V1)':>8s} | {'AUC':>7s} | {'pAUC':>7s} | " +
          " | ".join([f"R@F{f:<5}" for f in FPR_GRID]))
    print("-" * 108)
    res = {}
    base = None
    for name, mode, P in plan:
        d = blend(deep4, P, mode)
        f1, mixed = run(d, sim, y_tr, y_va, n_va)
        hm = head_metrics(mixed, y_va)
        if base is None: base = f1
        res[name] = {"mode": mode, "f1": f1, "dF1_vs_V1": f1 - base, **hm}
        print(f"{name:26s} | {f1:7.4f} | {f1-base:+8.4f} | {hm['auc']:7.4f} | {hm['pauc']:7.4f} | " +
              " | ".join([f"{hm[f'r@{f}']:8.4f}" for f in FPR_GRID]), flush=True)

    print("\n" + "=" * 108)
    print("  判定")
    print("=" * 108)
    v4, v3 = res["V4 min-max标定 + prop"]["f1"], res["V3 min-max标定 + asl"]["f1"]
    v2, v1 = res["V2 线性原始分 + prop"]["f1"], res["V1 线性原始分 + asl"]["f1"]
    print(f"  V1(基线) = {v1:.4f} | V2(原始分+prop) = {v2:.4f}  -> dF1 = {v2-v1:+.4f}  (I074 负结果复现)")
    print(f"  V3(minmax+asl) = {v3:.4f} | V4(minmax+prop) = {v4:.4f} -> dF1 = {v4-v3:+.4f}")
    if v4 > v3 + 0.0005 and v2 < v1:
        verdict = "标定错位假设成立: 换标定后倾向加权的增益传导"
    elif v2 < v1 and v4 <= v3 + 0.0005:
        verdict = "标定错位假设不成立: 换标定也无法传导 -> 回到信息冗余解释"
    else:
        verdict = "结论不明确, 需检查 V1/V2 是否复现 I074 的负结果"
    print(f"  {verdict}")

    out = {"results": res, "V1": v1, "verdict": verdict,
           "d_raw": v2 - v1, "d_minmax": v4 - v3,
           "d_zscore": res["V5 z-score标定 + prop"]["f1"] - v3,
           "d_rrf": res["V6 RRF秩融合 + prop"]["f1"] - v3,
           "w_plm": W_PLM, "fpr_grid": FPR_GRID}
    with io.open(REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
