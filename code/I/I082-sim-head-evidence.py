"""
I082 — 同源支头部漏检的证据分解 → DIAMOND top-100 重跑 go/no-go
================================================================
背景: 待办里有手动项「DIAMOND 重跑保留 top-100」, 但 I078 已证明 top-5 上可学习聚合失败。
      在让用户花时间重跑之前, 必须先量化: 头部漏检到底缺什么证据?
      若证据已在缓存命中里而聚合没用上 → 免费修;
      若证据根本不在 top-5 里 → top-100 才有价值; 若多数漏检与命中无关 → top-100 无用。

分解 (val, sim 支逐标签 SCut 阈值, 对每个真实正例 (蛋白, 标签) 且 sim 分 < 阈值):
  A 孤儿      : 该蛋白无任何 DIAMOND 命中
  B 证据闲置  : top-3 聚合没用到的第 4-5 命中携带该标签 (tk=5 即可救, 免费)
  C 证据已用  : top-3 命中携带该标签但分数仍低于阈值 (校准/聚合问题, top-100 无用)
  D 证据缺失  : top-5 命中都不带该标签 (需 top-100 / 更深检索才可能救)

附: tk=5 直接实测 (CK004b 时代测过但那是在旧管线; I064 现管线从未测过), 接入融合判据。
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
REPORT = os.path.join(REL_DIR, "I082_report.json")
N_LABELS = 500
W_PLM = 0.35
BASE_F1 = 0.4475

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe


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


def build_kmers(y_tr):
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
    print("  I082: Evidence decomposition of sim-branch head misses (top-100 go/no-go)")
    print("=" * 96)

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

    idx_v, bits_v = parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                              val_pids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim1_3 = agg(idx_v, bits_v, y_tr, tk=3)
    sim1_5 = agg(idx_v, bits_v, y_tr, tk=5)
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
    sim2_3 = transitive_2hop(sim1_3, hit, miss, v2v, n_va)
    sim2_5 = transitive_2hop(sim1_5, hit, miss, v2v, n_va)
    kmer = build_kmers(y_tr)
    sim3 = 0.4 * sim2_3 + 0.6 * kmer
    sim5 = 0.4 * sim2_5 + 0.6 * kmer

    # ---- A. tk=5 直接实测 ----
    print("\n" + "=" * 96)
    print("  [A] tk=3 (现状) vs tk=5 (用满缓存) — 独立 + 融合")
    print("=" * 96)
    res = {}
    for name, sim in [("tk=3 现状", sim3), ("tk=5 用满", sim5)]:
        sf1 = macro_f1(sim, y_va, search_scut(sim, y_va))
        f1, auc, mixed, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=plm, w=W_PLM)
        res[name] = {"standalone": sf1, "fused": f1, "auc": auc}
        print(f"  {name:12s} 独立 {sf1:.4f} | 融合 {f1:.4f} (基线 {BASE_F1}) | AUC {auc:.4f}")

    # ---- B. 头部漏检证据分解 ----
    print("\n" + "=" * 96)
    print("  [B] sim 支头部漏检分解 (真实正例且 sim 分 < 逐标签 SCut 阈值)")
    print("=" * 96)
    ths_sim = search_scut(sim3, y_va)
    below = (y_va == 1) & (sim3 < ths_sim[None, :])          # (n, L) 漏检格
    n_fn = int(below.sum())
    print(f"  sim 支漏检格总数: {n_fn:,d}")

    # 命中携带情况: lab5[k, j] = 第 k 命中是否有标签 j
    valid = idx_v >= 0
    lab5 = np.zeros((n_va, 5, N_LABELS), np.float32)
    for k in range(5):
        lab5[:, k] = y_tr[idx_v[:, k].clip(min=0)]
    lab5[~valid] = 0.0
    has_13 = lab5[:, :3].sum(1) > 0      # top-3 有该标签
    has_45 = lab5[:, 3:].sum(1) > 0      # 第 4-5 有该标签
    any_hit = hit[:, None] & np.ones((1, N_LABELS), bool)

    cat = np.zeros((n_va, N_LABELS), np.int8)
    cat[below] = np.where(~any_hit[below], 1,               # A 孤儿
                np.where(~has_13[below] & has_45[below], 2,  # B 证据闲置
                np.where(has_13[below], 3, 4)))              # C 证据已用 / D 证据缺失
    names = {1: "A 孤儿(无命中)", 2: "B 证据闲置(第4-5命中携带)", 3: "C 证据已用(top-3携带仍漏)",
             4: "D 证据缺失(top-5都不带)"}
    print(f"{'类别':32s} {'格数':>10s} {'占比':>8s}")
    print("-" * 60)
    dist = {}
    for c in [1, 2, 3, 4]:
        n_c = int((cat == c).sum())
        dist[names[c]] = {"n": n_c, "pct": n_c / max(n_fn, 1)}
        print(f"{names[c]:32s} {n_c:10,d} {n_c / max(n_fn, 1) * 100:7.2f}%")

    # ---- C. 判定 ----
    pB = dist[names[2]]["pct"]; pD = dist[names[4]]["pct"]; pC = dist[names[3]]["pct"]
    print("\n" + "=" * 96)
    print("  [C] go/no-go 判定")
    print("=" * 96)
    print(f"  免费可救 (B, tk=5 即可): {pB * 100:.1f}% 的漏检格")
    print(f"  top-100 才可能救 (D):    {pD * 100:.1f}% 的漏检格")
    print(f"  top-100 无用 (A+C):      {(1 - pB - pD) * 100:.1f}% 的漏检格")
    d_fused = res["tk=5 用满"]["fused"] - res["tk=3 现状"]["fused"]
    if d_fused > 0.0005:
        top100 = "GO — tk=5 已有增益, top-100 证据更多, 值得手动重跑"
    elif pD > 0.30:
        top100 = "CONDITIONAL — B 类增益有限, 但 D 类占比高, top-100 是唯一杠杆"
    else:
        top100 = "NO-GO — 漏检主因不是检索深度 (A+C 占多数), top-100 预期无用"
    print(f"  tk=5 融合差: {d_fused:+.4f}")
    print(f"  判定: {top100}")

    out = {"tk_scan": res, "fn_categories": dist, "n_fn": n_fn,
           "verdict_top100": top100, "d_fused_tk5": d_fused,
           "base_f1": BASE_F1}
    with io.open(REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
