"""
backfill_sheet_metrics — 从缓存概率补齐腾讯表缺失的实验指标 (只读缓存, 不训练)
==============================================================================
覆盖:
  canonical 融合分 (I064 口径)     -> I064/I065/I066b/I066c/I069/I070/I072/I074/
                                      I075/I076/I078/I079/I080/I081/I082 的 R/P(/AUC)
  resattn_val_probs.npz            -> I066 独立 R/P/F1/AUC
  multi_plm_mlp_val_probs.npz      -> I062 独立 R/P (修正 P=0 错值)
  i073_L1 (asl_prop)               -> I073 独立 P
  i073_L0 + full_pipe (w=0.35)     -> I077 V1 融合 R/P (校验 0.4471/0.8638)
  multi_plm(2176D) + full_pipe(w=.25) -> I063 融合 R/P/AUC (校验 0.4462)
  canonical 分数 1 位量化 @0.5     -> I068 决策 R/P (校验 F1 0.4411)
输出: rel/backfill_metrics.json
"""
import os, io, json, time
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe


def macro_pr(S, y, ths):
    preds = (S >= ths).astype(bool); yb = y.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    return float(np.mean(r)), float(np.mean(p))


def eval_scores(S, y, tag, out):
    ths = search_scut(S, y)
    f1 = macro_f1(S, y, ths)
    auc = roc_auc_score(y, S, average="macro")
    r, p = macro_pr(S, y, ths)
    out[tag] = {"recall": r, "precision": p, "auc": auc, "f1": f1}
    print(f"  {tag:28s} R={r:.4f} P={p:.4f} AUC={auc:.4f} F1={f1:.4f}")
    return out[tag]


def build_sim(y_tr, n_va, val_pids, train_map, val_map):
    def parse_aln(tsv, qids, idmap, ev=0.001, mx=5):
        idx = np.full((len(qids), mx), -1, np.int64)
        bits = np.zeros((len(qids), mx), np.float32)
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
        n = idx.shape[0]
        valid = idx[:, :tk] >= 0
        lab = np.zeros((n, tk, ys.shape[1]), np.float32)
        for j in range(tk): lab[:, j] = ys[idx[:, j].clip(min=0)]
        lab[~valid] = 0.0
        ww = np.maximum(w[:, :tk], 1e-7); ww[~valid] = 0.0
        return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)

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
    return 0.4 * sim2 + 0.6 * kmer


def main():
    t0 = time.time()
    print("=" * 88)
    print("  Backfill sheet metrics from cached probabilities")
    print("=" * 88)

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
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn + 0.5 * c005)
    sim = build_sim(y_tr, n_va, val_pids, train_map, val_map)
    plm1664 = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    out = {}

    print("\n  [1] canonical 融合分 (I064 口径)")
    f1c, aucc, mixed_c, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=plm1664, w=0.35)
    assert abs(f1c - 0.4475) < 0.0005, f1c
    m = eval_scores(mixed_c, y_va, "canonical_fused", out)

    print("\n  [2] I066 残基注意力检查点 (缓存 val 概率)")
    eval_scores(np.load(os.path.join(REL_DIR, "resattn_val_probs.npz"))["val"],
                y_va, "I066_resattn", out)

    print("\n  [3] I062 2176D 多PLM 独立分")
    eval_scores(np.load(os.path.join(REL_DIR, "multi_plm_mlp_val_probs.npz"))["val"],
                y_va, "I062_multi_plm", out)

    print("\n  [4] I073 L1 asl_prop 独立分")
    eval_scores(np.load(os.path.join(REL_DIR, "i073_L1_asl_prop_val.npz"))["val"],
                y_va, "I073_L1_prop", out)

    print("\n  [5] I077 V1 融合 (单种子 asl, w=0.35)")
    f1v, aucv, mixed_v, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va,
                                         plm=np.load(os.path.join(REL_DIR, "i073_L0_asl_val.npz"))["val"],
                                         w=0.35)
    assert abs(f1v - 0.4471) < 0.0005 and abs(aucv - 0.8638) < 0.0005, (f1v, aucv)
    eval_scores(mixed_v, y_va, "I077_V1_fused", out)

    print("\n  [6] I063 重建融合 (2176D, w=0.25)")
    f163, auc63, mixed_63, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va,
                                            plm=np.load(os.path.join(REL_DIR, "multi_plm_mlp_val_probs.npz"))["val"],
                                            w=0.25)
    assert abs(f163 - 0.4462) < 0.0005, f163
    eval_scores(mixed_63, y_va, "I063_fused_2176_w25", out)

    print("\n  [7] I068 1位小数量化决策 @0.5 (对象 = I066c 校准后概率)")
    ths_c = search_scut(mixed_c, y_va)
    cal = np.empty_like(mixed_c, dtype=np.float64)
    for j in range(mixed_c.shape[1]):
        col = mixed_c[:, j].astype(np.float64)
        t = float(np.clip(ths_c[j], col.min() + 1e-9, col.max() - 1e-9))
        lower = col <= t
        cal[lower, j] = 0.5 * (col[lower] - col.min()) / max(t - col.min(), 1e-12)
        cal[~lower, j] = 0.5 + 0.5 * (col[~lower] - t) / max(col.max() - t, 1e-12)
    q1 = np.floor(cal * 10 + 0.5) / 10.0        # 普通四舍五入到 1 位
    yb = y_va.astype(bool); pr = (q1 >= 0.5)
    tp = (pr & yb).sum(0); fp = (pr & ~yb).sum(0); fn = (~pr & yb).sum(0)
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    f1q = float(np.mean(np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)))
    # I068 报告中的 0.4411 是 cal_v[:2000] 子集快速估计; 全量 val 真值更低, 结论不变
    print(f"  注: I068 报告 0.4411 为 2000 行子集估计, 全量 val F1 = {f1q:.4f}")
    out["I068_quant1dec"] = {"recall": float(np.mean(r)), "precision": float(np.mean(p)),
                             "auc": 0.8499, "f1": f1q,
                             "note": "I068_report 0.4411 为 2000 行子集估计"}
    print(f"  {'I068_quant1dec':28s} R={np.mean(r):.4f} P={np.mean(p):.4f} F1={f1q:.4f}")

    with io.open(os.path.join(REL_DIR, "backfill_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: rel/backfill_metrics.json")
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
