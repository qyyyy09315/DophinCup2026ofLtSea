"""
I089 — 同源聚合算子消融 (CAFA5 官方基线算子 + DeepGOPlus 式 + 去重投票, 零训练)
==============================================================================
CAFA5 官方 BLAST 基线算子: score(q,l) = 命中中携带 l 者的「最大序列同一性」(非 bitscore 加权和)。
DeepGOPlus: CNN 分数与比对分数做 max 组合。
Wei2GO/Argot 系负结果: 近重复命中重复投票是同源分支假阳性的主要来源之一。
全部只用缓存比对文件 (第 5 列即 pident)。
变体 (算子输出 sim1 -> 2-hop -> 0.4*op + 0.6*kmer -> 融合):
  B0 现状: bitscore 加权 top-3
  B1 max-pident: 逐标签取携带该标签命中的最大 pident/100
  B2 max-bitscore: 逐标签取最大 bitscore/1000
  B3 去重投票: top-5 命中按「标签向量相同」分组, 每组取最大 bitscore 代表, 组间 bitscore 加权
  B4 现状+B1并列: 0.5*B0 + 0.5*B1 (两算子并列)
"""
import os, io, json, time
import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
REPORT = os.path.join(REL_DIR, "I089_report.json")
N_LABELS = 500
W_PLM = 0.35
BASE = 0.4475

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe


def parse_aln(tsv, qids, idmap, ev=0.001, mx=5):
    idx = np.full((len(qids), mx), -1, np.int64)
    bits = np.zeros((len(qids), mx), np.float32)
    pid = np.zeros((len(qids), mx), np.float32)
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
            pdt = float(p[4]) if len(p) > 4 else 0.0
            j = 0
            while j < mx and bits[i, j] > bs: j += 1
            if j < mx:
                idx[i, j:] = np.roll(idx[i, j:], 1); bits[i, j:] = np.roll(bits[i, j:], 1)
                pid[i, j:] = np.roll(pid[i, j:], 1)
                idx[i, j] = idmap[t]; bits[i, j] = bs; pid[i, j] = pdt
    return idx, bits, pid


def agg_bitscore(idx, w, ys, tk=3):
    n = idx.shape[0]
    valid = idx[:, :tk] >= 0
    lab = np.zeros((n, tk, ys.shape[1]), np.float32)
    for j in range(tk): lab[:, j] = ys[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :tk], 1e-7); ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)


def agg_max(idx, val, ys, mx, scale):
    """逐标签: 携带该标签的命中中 val 的最大值 / scale"""
    n = idx.shape[0]
    valid = idx >= 0
    out = np.zeros((n, ys.shape[1]), np.float32)
    for s in range(0, n, 2048):
        e = min(s + 2048, n)
        lab = np.zeros((e - s, mx, ys.shape[1]), np.float32)
        for j in range(mx):
            lab[:, j] = ys[idx[s:e, j].clip(min=0)]
        v = np.where(valid[s:e], val[s:e] / scale, 0.0)
        out[s:e] = (lab * v[:, :, None]).max(1)
    return out


def agg_dedup_full(idx, bits, ys, y_src, mx=5):
    """top-mx 命中按标签向量分组 (同标签向量 = 近重复证据), 每组取最大 bitscore 的命中为代表, 组间加权"""
    n = idx.shape[0]
    out = np.zeros((n, ys.shape[1]), np.float32)
    for i in range(n):
        reps = {}   # key=标签向量 -> (命中行向量, bitscore)
        for j in range(mx):
            ti = idx[i, j]
            if ti < 0: break
            key = y_src[ti].tobytes()
            if key not in reps or bits[i, j] > reps[key][1]:
                reps[key] = (ys[ti], float(bits[i, j]))
        if not reps:
            continue
        acc = np.zeros(ys.shape[1], np.float32)
        wsum = 0.0
        for yv, bs in reps.values():
            w = max(bs, 1e-7)
            acc += yv * w
            wsum += w
        out[i] = acc / max(wsum, 1e-7)
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
    print("=" * 100)
    print("  I089: Homology aggregation operator ablation (CAFA5 official operator etc.)")
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

    cnn = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    plm = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn + 0.5 * c005)

    idx_v, bits_v, pid_v = parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                                     val_pids, train_map)
    hit = idx_v[:, 0] >= 0; miss = ~hit
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

    def pipeline(sim1, tag):
        sim2 = transitive_2hop(sim1, hit, miss, v2v, n_va)
        sim = 0.4 * sim2 + 0.6 * kmer
        sf1 = macro_f1(sim, y_va, search_scut(sim, y_va))
        f1, auc, _, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=plm, w=W_PLM)
        print(f"  {tag:20s} 独立 {sf1:.4f} | 融合 {f1:.4f} ({f1 - BASE:+.4f}) | AUC {auc:.4f}", flush=True)
        return {"standalone_f1": sf1, "fused_f1": f1, "auc": float(auc)}

    res = {}
    print("\n  计算各算子 ...", flush=True)
    res["B0_bitscore_top3(现状)"] = pipeline(agg_bitscore(idx_v, bits_v, y_tr, tk=3), "B0 bitscore top3(现状)")
    res["B1_max_pident"] = pipeline(agg_max(idx_v, pid_v, y_tr, 5, 100.0), "B1 max-pident")
    res["B2_max_bitscore"] = pipeline(agg_max(idx_v, bits_v, y_tr, 5, 1000.0), "B2 max-bitscore")
    print("  B3 去重投票 (逐行, 约 1-2 分钟) ...", flush=True)
    res["B3_dedup_vote"] = pipeline(agg_dedup_full(idx_v, bits_v, y_tr, y_tr, mx=5), "B3 去重投票")
    b0 = res["B0_bitscore_top3(现状)"]
    print()

    best = max(res, key=lambda k: res[k]["fused_f1"])
    d = res[best]["fused_f1"] - BASE
    verdict = (f"{best} 超过基线 (+{d:.4f}), 值得接入 test 流程" if d > 0.0005
               else f"最优 {best} 未超基线 (+{d:.4f}), 现行 bitscore 加权 top-3 保持不变")
    print("=" * 100)
    print(f"  最优: {best} | 判定: {verdict}")
    print("=" * 100)
    with io.open(REPORT, "w", encoding="utf-8") as f:
        json.dump({"results": res, "best": best, "verdict": verdict}, f, indent=2, ensure_ascii=False)
    print(f"  Report: {REPORT}")
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
