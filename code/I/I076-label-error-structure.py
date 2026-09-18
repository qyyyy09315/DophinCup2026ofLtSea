"""
I076 — 标签级误差结构诊断: 定位融合收益真正所在的标签子集
==========================================================
I075 已在「分支级」量化了头部冗余 (全局 Jaccard)。本实验下推到「标签级」, 回答:
  在 500 个标签里, 哪些标签的融合增益最大? 这些标签上两支的头部假阳性重叠是否更低?
  逐标签秩相关 rho_l 与逐标签增益 dF1_l 的关系是否比分支级更干净 (无强度混淆)?

定义 (全部在 val 上, 逐标签):
  - headFP_l  : 负例中分数排在前 6.9% 的样本集合 (低 FPR 区的假阳性)
  - overlap_l : 两支 headFP_l 的 Jaccard  (越低 = 冗余越少)
  - rho_l     : 两支分数在「该标签正例」上的 Spearman 秩相关
  - dF1_l     : 融合后的逐标签 F1 - 两支逐标签 F1 的最大值 (融合的实际收益)
  - winner_l  : 逐标签 F1 最高的分支 (谁在该标签最强)

输出:
  1. 三支 + 融合的逐标签 F1 分布
  2. dF1_l 与 overlap_l / rho_l 的相关性 (标签级, n=500, 强度混淆已被 F1 差值部分抵消)
  3. 增益最大的 TOP-30 标签清单 -> 后续新信号应该攻的方向 (存 CSV)
  4. 按标签频率分带的增益结构
"""
import os, io, json, time
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORT = os.path.join(REL_DIR, "I076_report.json")
CSV_OUT = os.path.join(REL_DIR, "I076_label_targets.csv")
N_LABELS = 500
HEAD_FPR = 0.069

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
macro_f1, search_scut, col_norm = i064.macro_f1, i064.search_scut, i064.col_norm


def per_label_f1(S, y):
    """逐标签用自身最优阈值 (SCut 口径) 的 F1"""
    out = np.full(y.shape[1], np.nan)
    cands = np.linspace(0.01, 0.99, 100)
    for j in range(y.shape[1]):
        yt = y[:, j]
        if yt.sum() == 0: continue
        pr = S[:, j][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0); fp = (pr & ~ytm).sum(0); fn = (~pr & ytm).sum(0)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
        out[j] = f1.max()
    return out


def head_fp_set(S, y, j, fpr=HEAD_FPR):
    """**负例中**分数最高的前 fpr 比例 (真正的低 FPR 假阳性集合, 不含真阳性)"""
    negidx = np.where(y[:, j] == 0)[0]
    if len(negidx) == 0: return np.zeros(S.shape[0], bool)
    m = max(int(len(negidx) * fpr), 1)
    sel = negidx[np.argsort(-S[negidx, j])][:m]
    mask = np.zeros(S.shape[0], bool); mask[sel] = True
    return mask


def main():
    t0 = time.time()
    print("=" * 104)
    print("  I076: Per-Label Error-Structure Diagnosis — where does fusion actually earn?")
    print("=" * 104)

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

    cnn = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    plm = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
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

    # 真实融合分 (与 I064 full_pipe 同形, w=0.35)
    f1_fused, auc_fused, mixed, ths, s = i064.full_pipe(deep4, sim, y_tr, y_va, n_va, plm=plm, w=0.35)
    print(f"  融合分校验: Macro F1={f1_fused:.4f} (基线 0.4475) AUC={auc_fused:.4f}")

    # ---- 逐标签 F1 ----
    print("\n  计算逐标签 F1 (4 个分数场) ...", flush=True)
    F_fused = per_label_f1(mixed, y_va)
    F_sim = per_label_f1(sim, y_va)
    F_deep = per_label_f1(deep4 + 0.35 * plm, y_va)     # deep5
    F_plm = per_label_f1(plm, y_va)
    print("=" * 104)
    print("  [节 1] 逐标签 F1 分布 (500 标签)")
    print("=" * 104)
    print(f"{'场':18s} | {'均值':>7s} | {'中位':>7s} | {'Q1':>7s} | {'Q3':>7s} | {'零分标签数':>9s} | {'>0.5 标签数':>10s}")
    print("-" * 104)
    for nm, F in [("融合 (最终)", F_fused), ("sim 支", F_sim), ("deep5 支", F_deep), ("plm 支", F_plm)]:
        v = F[~np.isnan(F)]
        print(f"{nm:18s} | {v.mean():7.4f} | {np.median(v):7.4f} | {np.quantile(v,0.25):7.4f} | "
              f"{np.quantile(v,0.75):7.4f} | {int((v==0).sum()):9d} | {int((v>0.5).sum()):10d}")

    # ---- 节 2: 逐标签增益 vs 冗余 ----
    print("\n" + "=" * 104)
    print("  [节 2] 逐标签融合增益 dF1_l vs 头部假阳性重叠 overlap_l / 正例秩相关 rho_l")
    print("=" * 104)
    best_single = np.maximum(np.maximum(np.nan_to_num(F_sim, nan=-1), np.nan_to_num(F_deep, nan=-1)),
                             np.nan_to_num(F_plm, nan=-1))
    dF1 = F_fused - np.where(best_single < 0, np.nan, best_single)

    ov = np.full(N_LABELS, np.nan); rho = np.full(N_LABELS, np.nan)
    for j in range(N_LABELS):
        yt = y_va[:, j]
        if yt.sum() == 0: continue
        ma = head_fp_set(sim, y_va, j)
        mb = head_fp_set(deep4 + 0.35 * plm, y_va, j)
        u = (ma | mb).sum()
        if u > 0: ov[j] = (ma & mb).sum() / u
        pos = yt.astype(bool)
        if pos.sum() >= 5 and sim[pos, j].std() > 1e-12 and (deep4 + 0.35 * plm)[pos, j].std() > 1e-12:
            rho[j] = spearmanr(sim[pos, j], (deep4 + 0.35 * plm)[pos, j]).statistic

    ok = ~np.isnan(dF1) & ~np.isnan(ov)
    print(f"  有效标签: dF1 {int((~np.isnan(dF1)).sum())} | overlap {int((~np.isnan(ov)).sum())} | rho {int((~np.isnan(rho)).sum())}")
    print(f"  dF1_l 均值={np.nanmean(dF1):+.4f} 中位={np.nanmedian(dF1):+.4f} "
          f"正增益标签数={int((dF1>0).sum())} 负增益标签数={int((dF1<0).sum())}")
    print(f"  overlap_l 均值={np.nanmean(ov):.4f} | rho_l 均值={np.nanmean(rho):.4f}")
    r_do = float(np.corrcoef(dF1[ok], ov[ok])[0, 1])
    ok2 = ~np.isnan(dF1) & ~np.isnan(rho)
    r_dr = float(np.corrcoef(dF1[ok2], rho[ok2])[0, 1])
    print(f"\n  标签级 Pearson 相关 (n={int(ok.sum())}):")
    print(f"    dF1_l  vs  overlap_l : r = {r_do:+.4f}   <- 期望为负 (冗余越低增益越大)")
    print(f"    dF1_l  vs  rho_l     : r = {r_dr:+.4f}   <- 期望为负 (越不相关增益越大)  n={int(ok2.sum())}")

    # ---- 节 3: TOP 增益标签 ----
    print("\n" + "=" * 104)
    print("  [节 3] 融合增益最大的 TOP-30 标签 (新信号应攻方向)")
    print("=" * 104)
    order = np.argsort(-np.nan_to_num(dF1, nan=-1))[:30]
    print(f"{'排名':>4s} | {'标签':>8s} | {'freq':>7s} | {'融合F1':>7s} | {'sim F1':>7s} | {'deep5 F1':>8s} | "
          f"{'dF1':>8s} | {'overlap':>8s} | {'rho':>7s}")
    print("-" * 104)
    rows = []
    for r, j in enumerate(order, 1):
        rows.append({"rank": r, "label_idx": int(j), "label": label_cols[j], "freq": float(freq[j]),
                     "f1_fused": float(F_fused[j]), "f1_sim": float(F_sim[j]), "f1_deep5": float(F_deep[j]),
                     "f1_plm": float(F_plm[j]), "dF1": float(dF1[j]),
                     "overlap": float(ov[j]) if not np.isnan(ov[j]) else None,
                     "rho": float(rho[j]) if not np.isnan(rho[j]) else None})
        print(f"{r:4d} | {label_cols[j]:>8s} | {freq[j]:7.4f} | {F_fused[j]:7.4f} | {F_sim[j]:7.4f} | "
              f"{F_deep[j]:8.4f} | {dF1[j]:+8.4f} | "
              f"{(ov[j] if not np.isnan(ov[j]) else float('nan')):8.4f} | "
              f"{(rho[j] if not np.isnan(rho[j]) else float('nan')):7.4f}")
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"\n  已存: {CSV_OUT}")

    # ---- 节 4: 频率分带增益结构 ----
    print("\n" + "=" * 104)
    print("  [节 4] 按标签频率分带的增益与冗余结构")
    print("=" * 104)
    qs = np.quantile(freq, [0.25, 0.5, 0.75])
    bands = {"极稀有 <Q1": freq <= qs[0], "稀有 Q1-Q2": (freq > qs[0]) & (freq <= qs[1]),
             "中等 Q2-Q3": (freq > qs[1]) & (freq <= qs[2]), "高频 >Q3": freq > qs[2]}
    print(f"{'频带':14s} | {'n':>4s} | {'融合F1':>7s} | {'sim F1':>7s} | {'deep5 F1':>8s} | "
          f"{'dF1':>8s} | {'overlap':>8s} | {'rho':>7s}")
    print("-" * 104)
    band_out = {}
    for b, m in bands.items():
        band_out[b] = {"n": int(m.sum()), "f1_fused": float(np.nanmean(F_fused[m])),
                       "f1_sim": float(np.nanmean(F_sim[m])), "f1_deep5": float(np.nanmean(F_deep[m])),
                       "dF1": float(np.nanmean(dF1[m])), "overlap": float(np.nanmean(ov[m])),
                       "rho": float(np.nanmean(rho[m]))}
        d = band_out[b]
        print(f"{b:14s} | {d['n']:4d} | {d['f1_fused']:7.4f} | {d['f1_sim']:7.4f} | {d['f1_deep5']:8.4f} | "
              f"{d['dF1']:+8.4f} | {d['overlap']:8.4f} | {d['rho']:7.4f}")

    out = {"fused_macro_f1": f1_fused, "fused_auc": auc_fused,
           "per_label_stats": {nm: {"mean": float(np.nanmean(F)), "median": float(np.nanmedian(F)),
                                    "zero_labels": int((np.nan_to_num(F) == 0).sum()),
                                    "gt05_labels": int((np.nan_to_num(F) > 0.5).sum())}
                               for nm, F in [("fused", F_fused), ("sim", F_sim), ("deep5", F_deep), ("plm", F_plm)]},
           "dF1_mean": float(np.nanmean(dF1)), "dF1_pos_labels": int((dF1 > 0).sum()),
           "dF1_neg_labels": int((dF1 < 0).sum()),
           "overlap_mean": float(np.nanmean(ov)), "rho_mean": float(np.nanmean(rho)),
           "corr_dF1_overlap": r_do, "corr_dF1_rho": r_dr,
           "band_structure": band_out, "top30": rows, "head_fpr": HEAD_FPR}
    with io.open(REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
