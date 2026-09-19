"""
I085 — DIAMOND top-100 门控聚合评估 (前置: 用户重跑比对 --max-target-seqs 100)
================================================================
背景: I082 证据分解显示 50.8% 的 sim 漏检格属于「top-5 命中都不带标签」,
      更深检索是唯一杠杆; 但 tk=5 零增益证明无门控堆命中会被弱同源噪声淹没
      (CK005k 教训)。本实验在 top-100 命中上测门控聚合。
输入: rel/ck005a_tmp/align_val_very-sensitive_e0.001_top100.tsv (qseqid sseqid bitscore evalue pident)
变体 (全部 bitscore 加权, kmer 部分不动 0.6 权重):
  top3     自检 (应复现现状 0.4316 独立)
  top10 / top20 / top50          纯 top-K
  gate50   top-100 + 累计 bitscore 份额 >= 50% 截断 (优势命中主导)
  gate70   同上, 份额 70%
判据: 独立 F1 + 融合 Macro F1 (w=0.35, 基线 0.4475), 通过线 +0.001。
"""
import os, io, json, time
import numpy as np
import pandas as pd
from collections import defaultdict
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
ALN100 = os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001_top100.tsv")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
REPORT = os.path.join(REL_DIR, "I085_report.json")
N_LABELS = 500
W_PLM = 0.35
BASE = 0.4475

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe


def parse_topk(tsv, qids, idmap, ev=0.001, kmax=100):
    qp = {q: i for i, q in enumerate(qids)}
    hits = defaultdict(list)
    with open(tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            q, t = p[0], p[1]
            if q not in qp or t not in idmap: continue
            try: bs, e = float(p[2]), float(p[3])
            except ValueError: continue
            if e > ev: continue
            hits[qp[q]].append((bs, idmap[t]))
    idx = np.full((len(qids), kmax), -1, np.int64)
    bits = np.zeros((len(qids), kmax), np.float32)
    for i, lst in hits.items():
        lst.sort(reverse=True)
        for j, (bs, ti) in enumerate(lst[:kmax]):
            idx[i, j] = ti; bits[i, j] = bs
    return idx, bits


def agg_topk(idx, bits, ys, tk):
    n = idx.shape[0]
    valid = idx[:, :tk] >= 0
    lab = np.zeros((n, tk, ys.shape[1]), np.float32)
    for j in range(tk):
        lab[:, j] = ys[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(bits[:, :tk], 1e-7); ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)


def agg_gated(idx, bits, ys, share):
    """按累计 bitscore 份额截断: 保留前缀直到累计 >= share x 该行总 bitscore"""
    n, kmax = idx.shape
    valid = idx >= 0
    ww = np.where(valid, np.maximum(bits, 0.0), 0.0)
    cum = np.cumsum(ww, axis=1)
    total = cum[:, -1:].copy()
    keep = (cum <= np.maximum(share * total, ww[:, :1])) & valid
    keep[:, 0] = valid[:, 0]
    lab = np.zeros((n, kmax, ys.shape[1]), np.float32)
    # 分块构造, 避免整段 (n,100,500) 过大
    out = np.zeros((n, ys.shape[1]), np.float32)
    wsum = ww.sum(1)
    for s in range(0, n, 2048):
        e = min(s + 2048, n)
        lab_s = np.zeros((e - s, kmax, ys.shape[1]), np.float32)
        for j in range(kmax):
            lab_s[:, j] = ys[idx[s:e, j].clip(min=0)]
        lab_s[~keep[s:e]] = 0.0
        w_s = ww[s:e][:, :, None]
        out[s:e] = (lab_s * w_s).sum(1) / np.maximum(w_s.sum(1), 1e-7)
    return out


def transitive_2hop(sim1, hit, miss, nbr, n):
    out = sim1.copy()
    for qi in range(n):
        if miss[qi] and qi in nbr:
            acc = np.zeros(N_LABELS, np.float32); aw = 0.0
            for ti, bs in nbr[qi]:
                if hit[ti]:
                    acc += sim1[ti] * bs; aw += bs
            if aw > 0: out[qi] = acc / aw
    return out


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


def main():
    t0 = time.time()
    print("=" * 96)
    print("  I085: DIAMOND top-100 gated aggregation")
    print("=" * 96)
    if not os.path.exists(ALN100):
        print(f"  缺少 {ALN100}")
        print("  先重跑比对 (手动长任务, 见待办): --max-target-seqs 100 版本的 val->train 表")
        return

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
    plm = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn + 0.5 * c005)

    print("  解析 top-100 比对 (约 1-2 分钟) ...", flush=True)
    idx100, bits100 = parse_topk(ALN100, val_pids, train_map)
    hit = idx100[:, 0] >= 0
    miss = ~hit
    print(f"  命中率 {hit.mean():.3f} | 每查询命中数中位 {int((idx100 >= 0).sum(1).clip(0, 100).astype(float).median())}")

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

    kmer = build_kmer(y_tr)
    variants = [
        ("top3(自检)", lambda: agg_topk(idx100, bits100, y_tr, 3)),
        ("top10", lambda: agg_topk(idx100, bits100, y_tr, 10)),
        ("top20", lambda: agg_topk(idx100, bits100, y_tr, 20)),
        ("top50", lambda: agg_topk(idx100, bits100, y_tr, 50)),
        ("gate50", lambda: agg_gated(idx100, bits100, y_tr, 0.5)),
        ("gate70", lambda: agg_gated(idx100, bits100, y_tr, 0.7)),
    ]
    res = {}
    for name, fn in variants:
        sim1 = fn()
        sim2 = transitive_2hop(sim1, hit, miss, v2v, n_va)
        sim = 0.4 * sim2 + 0.6 * kmer
        sf1 = macro_f1(sim, y_va, search_scut(sim, y_va))
        f1, auc, _, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=plm, w=W_PLM)
        res[name] = {"standalone_f1": sf1, "fused_f1": f1, "auc": auc}
        print(f"  {name:12s} 独立 {sf1:.4f} | 融合 {f1:.4f} ({f1 - BASE:+.4f}) | AUC {auc:.4f}", flush=True)

    best = max(res, key=lambda k: res[k]["fused_f1"])
    d = res[best]["fused_f1"] - BASE
    verdict = (f"{best} 过通过线 (+{d:.4f}), 建议接 test 侧流程出提交" if d >= 0.001
               else f"最优 {best} 未过通过线 (+{d:.4f}), top-100 门控聚合关闭")
    print("\n" + "=" * 96)
    print(f"  最优: {best} | 判定: {verdict}")
    print("=" * 96)
    with io.open(REPORT, "w", encoding="utf-8") as f:
        json.dump({"results": res, "best": best, "verdict": verdict, "base": BASE}, f,
                  indent=2, ensure_ascii=False)
    print(f"  Report: {REPORT}")
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
