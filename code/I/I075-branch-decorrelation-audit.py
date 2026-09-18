"""
I075 — 分支去相关性审计: 把「独立越强融合越差」从现象变成可测量的设计准则
==========================================================================
背景 (I063 / I074 三次独立复现的法则):
  独立分支指标上升 -> 融合指标下降, 因为分支间误差相关性上升, 互补增益被吃掉。

本实验不训练任何模型, 全部基于已有 val 概率:
  - 真实分路: cnn(C002b) / c005(CSRA) / t001a / t002 / sim(DIAMOND+2hop+kmer)
  - 受控对照对 (I073 产出, 同种子42/同架构, 仅损失不同):
      plm_asl  = i073_L0_asl_val.npz
      plm_prop = i073_L1_asl_prop_val.npz   (正例 1/freq 加权)
    这一对是检验机制的干净探针: I074 已测出 plm_prop 独立更强、但融合更差。

四节输出:
  1. 各分路独立指标 (F1/AUC/AP/pAUC/R@FPR)
  2. 按标签频率分带的 AP 专家表 -> 谁在哪个频带最强 (互补结构可视化)
  3. 两两「头部一致性」 vs 「配对融合增益」-> 量化负相关 (设计准则的实证)
  4. 受控对照: plm_asl vs plm_prop 与 sim 的头部一致性差异 -> 直接验证机制
"""
import os, io, json, time
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc as sk_auc, average_precision_score

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT = os.path.join(REL_DIR, "I075_report.json")
N_LABELS = 500
FPR_GRID = [0.02, 0.05, 0.069, 0.10]
TOPQ = 0.99          # 头部定义: 逐标签分数分位 >= 99%

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
macro_f1, search_scut, col_norm = i064.macro_f1, i064.search_scut, i064.col_norm


def branch_metrics(S, y):
    aucs, ap, paucs = [], [], []
    rec = {f: [] for f in FPR_GRID}
    for j in range(y.shape[1]):
        yt = y[:, j]
        if yt.min() == yt.max(): continue
        fpr, tpr, _ = roc_curve(yt, S[:, j])
        if len(fpr) < 2: continue
        aucs.append(sk_auc(fpr, tpr))
        ap.append(average_precision_score(yt, S[:, j]))
        m = fpr <= 0.10
        paucs.append(sk_auc(fpr[m], tpr[m]) / 0.10 if m.sum() >= 2 else 0.0)
        for f in FPR_GRID:
            k = np.searchsorted(fpr, f, side="right") - 1
            rec[f].append(float(tpr[max(k, 0)]))
    return {"auc": float(np.mean(aucs)), "ap": float(np.mean(ap)), "pauc": float(np.mean(paucs)),
            **{f"r@{f}": float(np.mean(rec[f])) for f in FPR_GRID}}


def per_label_ap(S, y):
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        yt = y[:, j]
        if yt.min() == yt.max(): continue
        out[j] = average_precision_score(yt, S[:, j])
    return out


def corr_all_pairs(mats, names):
    """逐标签 Pearson (全样本) 的解析式: 对每列标准化后取内积"""
    Z = {}
    for n, M in zip(names, mats):
        Zc = M - M.mean(0, keepdims=True)
        Zc /= np.maximum(M.std(0, keepdims=True), 1e-9)
        Z[n] = Zc
    n = mats[0].shape[0]
    out = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            out[(names[a], names[b])] = float(np.mean((Z[names[a]] * Z[names[b]]).sum(0) / n))
    return out


def head_agreement(A, B, y, q=TOPQ):
    """逐标签: 取 A 或 B 处于头部 (>= q 分位) 的行, 在该子集上算两路分数的 Pearson, 再对标签平均。
    度量「两路在排序头部是否彼此复制」。"""
    n, L = A.shape
    vals = []
    for j in range(L):
        yt = y[:, j]
        if yt.min() == yt.max(): continue
        ta = np.quantile(A[:, j], q); tb = np.quantile(B[:, j], q)
        s = (A[:, j] >= ta) | (B[:, j] >= tb)
        if s.sum() < 30: continue
        a, b = A[s, j], B[s, j]
        if a.std() < 1e-12 or b.std() < 1e-12: continue
        vals.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(vals)), len(vals)


def head_jaccard(A, B, y, q=TOPQ):
    js = []
    for j in range(y.shape[1]):
        if y[:, j].min() == y[:, j].max(): continue
        sa = A[:, j] >= np.quantile(A[:, j], q)
        sb = B[:, j] >= np.quantile(B[:, j], q)
        u = (sa | sb).sum()
        if u == 0: continue
        js.append((sa & sb).sum() / u)
    return float(np.mean(js))


def main():
    t0 = time.time()
    print("=" * 100)
    print("  I075: Branch Decorrelation Audit — 'standalone stronger != fused stronger' as a design rule")
    print("=" * 100)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    freq = y_tr.mean(0)

    # ---- 分路装载 ----
    print("\n  装载分路 (val) ...", flush=True)
    cnn = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    plm_asl = np.load(os.path.join(REL_DIR, "i073_L0_asl_val.npz"))["val"]
    plm_prop = np.load(os.path.join(REL_DIR, "i073_L1_asl_prop_val.npz"))["val"]
    for nm, M in [("cnn", cnn), ("c005", c005), ("t001a", t1), ("t002", t2),
                  ("plm_asl", plm_asl), ("plm_prop", plm_prop)]:
        assert M.shape == (n_va, N_LABELS), f"{nm} {M.shape}"

    deep4 = 0.5 * cnn + 0.5 * c005 + t1 + t2     # 与 I064 同形的粗混合, 仅用于参照
    with open(os.path.join(REL_DIR, "ck005a_tmp", "..", "..", "models", "tck002-ensemble.json")
              if False else os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json"),
              encoding="utf-8") as f:
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
    print(f"  分路就绪: 6 支 + sim | val {n_va} x {N_LABELS}")

    # ---- 节 1: 独立指标 ----
    print("\n" + "=" * 100)
    print("  [节 1] 各分路独立指标 (val) — 注意 AUC 排名与 R@FPR0.069 排名并不一致")
    print("=" * 100)
    allb = [("cnn(C002b)", cnn), ("c005(CSRA)", c005), ("t001a", t1), ("t002", t2),
            ("plm_asl", plm_asl), ("plm_prop", plm_prop), ("sim", sim), ("deep4", deep4)]
    print(f"{'分路':14s} | {'F1':>7s} | {'AUC':>7s} | {'AP':>7s} | {'pAUC':>7s} | " +
          " | ".join([f"R@F{f:<5}" for f in FPR_GRID]))
    print("-" * 100)
    S1 = {}
    for nm, M in allb:
        f1 = macro_f1(M, y_va, search_scut(M, y_va))
        bm = branch_metrics(M, y_va)
        S1[nm] = {"f1": f1, **bm}
        print(f"{nm:14s} | {f1:7.4f} | {bm['auc']:7.4f} | {bm['ap']:7.4f} | {bm['pauc']:7.4f} | " +
              " | ".join([f"{bm[f'r@{f}']:8.4f}" for f in FPR_GRID]), flush=True)
    S1["cnn"] = S1["cnn(C002b)"]; S1["c005"] = S1["c005(CSRA)"]

    # ---- 节 2: 频率分带专家表 ----
    print("\n" + "=" * 100)
    print("  [节 2] 按训练集标签频率分带的 AP — 谁在哪个频带是专家")
    print("=" * 100)
    qs = np.quantile(freq, [0.25, 0.5, 0.75])
    bands = {"极稀有 <Q1": freq <= qs[0], "稀有 Q1-Q2": (freq > qs[0]) & (freq <= qs[1]),
             "中等 Q2-Q3": (freq > qs[1]) & (freq <= qs[2]), "高频 >Q3": freq > qs[2]}
    aps = {nm: per_label_ap(M, y_va) for nm, M in allb}
    hdr = " | ".join([f"{b:>13s}" for b in bands])
    print(f"{'分路':14s} | {hdr}")
    print("-" * 100)
    band_tab = {}
    for nm, _ in allb:
        row = {b: float(np.nanmean(aps[nm][m])) for b, m in bands.items()}
        band_tab[nm] = row
        print(f"{nm:14s} | " + " | ".join([f"{row[b]:13.4f}" for b in bands]))
    print(f"\n  (各带标签数: " + ", ".join([f"{b}={int(m.sum())}" for b, m in bands.items()]) + ")")
    for b in bands:
        best = max(band_tab, key=lambda k: band_tab[k][b])
        print(f"  {b:13s} 带内 AP 最高: {best} ({band_tab[best][b]:.4f})")

    # ---- 节 3: 头部一致性 vs 配对融合增益 ----
    print("\n" + "=" * 100)
    print("  [节 3] 两两「头部一致性」 vs 「配对融合增益」 — 设计准则的实证")
    print("=" * 100)
    pair_b = [("cnn", cnn), ("c005", c005), ("t001a", t1), ("t002", t2),
              ("plm_asl", plm_asl), ("sim", sim)]
    ca = corr_all_pairs([M for _, M in pair_b], [n for n, _ in pair_b])
    rows = []
    for a in range(len(pair_b)):
        for b in range(a + 1, len(pair_b)):
            na, Ma = pair_b[a]; nb, Mb = pair_b[b]
            F = 0.5 * col_norm(Ma) + 0.5 * col_norm(Mb)
            f_fuse = macro_f1(F, y_va, search_scut(F, y_va))
            f_a = S1[na]["f1"]; f_b = S1[nb]["f1"]
            ha, _ = head_agreement(Ma, Mb, y_va)
            rows.append({"A": na, "B": nb, "corr_all": ca[(na, nb)], "head_agree": ha,
                         "jaccard": head_jaccard(Ma, Mb, y_va),
                         "f1_fuse": f_fuse, "f1_best": max(f_a, f_b),
                         "f1_weak": min(f_a, f_b),
                         "gain": f_fuse - max(f_a, f_b)})
    rows.sort(key=lambda r: -r["head_agree"])
    print(f"{'A':9s} | {'B':9s} | {'头部一致性':>10s} | {'全样本相关':>10s} | {'头部Jaccard':>11s} | "
          f"{'配对融合F1':>10s} | {'最优单支F1':>10s} | {'增益':>8s}")
    print("-" * 100)
    for r in rows:
        print(f"{r['A']:9s} | {r['B']:9s} | {r['head_agree']:10.4f} | {r['corr_all']:10.4f} | "
              f"{r['jaccard']:11.4f} | {r['f1_fuse']:10.4f} | {r['f1_best']:10.4f} | {r['gain']:+8.4f}")
    hs = np.array([r["head_agree"] for r in rows]); gs = np.array([r["gain"] for r in rows])
    cs = np.array([r["corr_all"] for r in rows]); js = np.array([r["jaccard"] for r in rows])
    ws = np.array([r["f1_weak"] for r in rows])
    r_hg = float(np.corrcoef(hs, gs)[0, 1]); r_cg = float(np.corrcoef(cs, gs)[0, 1])
    r_jg = float(np.corrcoef(js, gs)[0, 1]); r_wg = float(np.corrcoef(ws, gs)[0, 1])
    print(f"\n  跨 {len(rows)} 个配对的 Pearson 相关:")
    print(f"    头部一致性  vs 融合增益: r = {r_hg:+.4f}")
    print(f"    全样本相关  vs 融合增益: r = {r_cg:+.4f}")
    print(f"    头部 Jaccard vs 融合增益: r = {r_jg:+.4f}")
    print(f"    [混淆对照] 弱支 F1  vs 融合增益: r = {r_wg:+.4f}   <- 若此项绝对值很大, 说明上面的相关被强度混淆")

    # 二元最小二乘: gain ~ b1*head_agree + b2*f1_weak, 暴露强度混淆
    Xd = np.column_stack([np.ones_like(hs), hs, ws])
    coef, *_ = np.linalg.lstsq(Xd, gs, rcond=None)
    pred = Xd @ coef
    ss_res = ((gs - pred) ** 2).sum(); ss_tot = ((gs - gs.mean()) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    print(f"    二元拟合 gain ~ a*head_agree + b*f1_weak:")
    print(f"      a(head_agree) = {coef[1]:+.4f}  b(f1_weak) = {coef[2]:+.4f}  R^2 = {r2:.4f}")
    print(f"    -> 若 a 的符号在控制强度后翻转/消失, 则节 3 的相关不可作为去相关法则的证据。")

    # ---- 节 4: 受控对照 (同种子42/同架构, 仅损失不同) ----
    print("\n" + "=" * 100)
    print("  [节 4] 受控对照: plm_asl vs plm_prop (I073 同种子42, 仅损失不同)")
    print("=" * 100)
    for nm, M in [("plm_asl", plm_asl), ("plm_prop", plm_prop)]:
        ha, nlab = head_agreement(M, sim, y_va)
        print(f"  {nm:9s}: 独立 F1={S1[nm]['f1']:.4f} | AP={S1[nm]['ap']:.4f} | "
              f"R@0.069={S1[nm][f'r@{FPR_GRID[2]}']:.4f} || "
              f"与 sim 头部一致性={ha:.4f} (n={nlab}) | Jaccard={head_jaccard(M, sim, y_va):.4f}")
    # 机制探针: 与「其余分支组成的集成」的相关性, 比与单一分支的相关性更接近真实问题
    print("\n  机制探针 — 与「集成(deep4 + sim)」的相关性 (受控: 同种子42/同架构, 仅损失不同):")
    rest = deep4 + sim
    for nm, M in [("plm_asl", plm_asl), ("plm_prop", plm_prop)]:
        h_rest, _ = head_agreement(M, rest, y_va)
        c_rest = float(np.mean([np.corrcoef(M[:, j], rest[:, j])[0, 1]
                                for j in range(N_LABELS) if M[:, j].std() > 1e-12 and rest[:, j].std() > 1e-12]))
        print(f"    {nm:9s}: 头部一致性={h_rest:.4f} | 全样本相关={c_rest:.4f} | "
              f"Jaccard={head_jaccard(M, rest, y_va):.4f}")
    for nm, M in [("plm_asl", plm_asl), ("plm_prop", plm_prop)]:
        F = 0.5 * col_norm(M) + 0.5 * col_norm(sim)
        f1f = macro_f1(F, y_va, search_scut(F, y_va))
        print(f"  {nm:9s} x sim 配对融合 F1 = {f1f:.4f} "
              f"(较 plm 单支 {S1[nm]['f1']:+.4f}, 较 sim 单支 {S1['sim']['f1']:+.4f})")

    out = {"section1_standalone": S1, "section2_band_ap": band_tab,
           "section2_band_sizes": {b: int(m.sum()) for b, m in bands.items()},
           "section3_pairs": rows,
           "section3_corr_headagree_vs_gain": r_hg,
           "section3_corr_corrall_vs_gain": r_cg,
           "section3_corr_jaccard_vs_gain": r_jg,
           "fpr_grid": FPR_GRID, "topq": TOPQ}
    with io.open(REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
