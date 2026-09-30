"""
I092 — 「独立增益有效但不传导」机理研究（分析性，不写腾讯表格）
================================================================
背景：四点单调法则（I063/I074/I077/I087）+ I073-L1 同形态；152/155 轮确认 PFP 文献无
「分支强度 vs 多样性权衡」系统实验（可发表空白）。

H1 评测协议伪象（探针 a）：历史判定把融合配置（RRF k/β、Sinkhorn τ/α、注入 w）冻结在
   C011a 网格最优点 → 若对候选与基线**同等预算**重扫 (w, β, τ, α) 后传导恢复，「不传导」
   部分是简化评测协议的伪象。
H2 频带错位（探针 b）：独立增益来自尾部/校准，融合瓶颈在头部排序 → 按标签频次四分位
   分解独立/融合传导率。
H3 误差共线吞噬（探针 c）：标签级 ΔF1(独立) vs ΔF1(融合) 相关 + 高增益标签去向。

判据：任一假设给出可复用传导判据或反例 = 有效研究产出；若 H1 重扫后传导恢复 ≥ +0.001，
同时构成换锁候选。输出：rel/I092_report.json（只写实验日志）。
"""
import os
import io
import json
import time

import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
FROZEN = {"w": 0.35, "beta": 0.50, "tau": 0.20, "alpha": 0.60}
W_GRID = [0.25, 0.35, 0.45]
BETA_GRID = [0.25, 0.50, 0.75]
TAU_GRID = [0.1, 0.2, 0.4]
ALPHA_GRID = [0.4, 0.6, 0.8]

i064 = SourceFileLoader(
    "i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")
).load_module()
macro_f1, search_scut = i064.macro_f1, i064.search_scut
ranks, col_norm, sinkhorn = i064.ranks, i064.col_norm, i064.sinkhorn


def pipe_cfg(deep, sim, y_tr, y_va, n_va, plm, w, beta, tau, alpha):
    """I064.full_pipe 的带旋钮副本：w=PLM 注入、β=RRF、τ=Sinkhorn 温度、α=传输项权重。"""
    d = (1.0 - w) * deep + w * plm
    s = d + 0.75 * sim + beta * col_norm(1.0 / (30 + ranks(d) + 1) + 1.0 / (30 + ranks(sim) + 1))
    t_tmp = search_scut(s, y_va)
    row_m = np.maximum((s >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_tr.mean(0) * n_va, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s - s.max(axis=1, keepdims=True)) / tau)
    p_v = sinkhorn(ker, row_m, col_m)
    mixed = (1.0 - alpha) * s + alpha * p_v
    ths = search_scut(mixed, y_va)
    return macro_f1(mixed, y_va, ths), mixed, ths


def per_label_f1(S, y, ths):
    preds = S >= ths
    yb = y.astype(bool)
    tp = (preds & yb).sum(0)
    fp = (preds & ~yb).sum(0)
    fn = (~preds & yb).sum(0)
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)


def build_sim(y_tr, n_va, val_pids, train_map, val_map):
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

    idx_v, bits_v = parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                              val_pids, train_map)
    hit = idx_v[:, 0] >= 0
    miss = ~hit
    sim1 = agg(idx_v, bits_v, y_tr)
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
    sim2 = sim1.copy()
    for qi in range(n_va):
        if miss[qi] and qi in v2v:
            acc = np.zeros(N_LABELS, np.float32)
            aw = 0.0
            for ti, bs in v2v[qi]:
                if hit[ti]:
                    acc += sim1[ti] * bs
                    aw += bs
            if aw > 0:
                sim2[qi] = acc / aw
    kmer = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0:
            continue
        pre = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{pre}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_tr[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        sm = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(sm, 1e-7) ** 2
        kmer += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return 0.4 * sim2 + 0.6 * kmer


def grid_eval(deep, sim, y_tr, y_va, n_va, plm):
    """两阶段重扫：先 (w, β) 再 (τ, α)；返回全格结果与最优。"""
    res = {}
    best1 = None
    for w in W_GRID:
        for beta in BETA_GRID:
            f1, _, _ = pipe_cfg(deep, sim, y_tr, y_va, n_va, plm,
                                w, beta, FROZEN["tau"], FROZEN["alpha"])
            res[f"w{w:.2f}_b{beta:.2f}"] = f1
            if best1 is None or f1 > best1[0]:
                best1 = (f1, w, beta)
    _, w_b, b_b = best1
    best2 = best1
    for tau in TAU_GRID:
        for alpha in ALPHA_GRID:
            f1, _, _ = pipe_cfg(deep, sim, y_tr, y_va, n_va, plm,
                                w_b, b_b, tau, alpha)
            res[f"w{w_b:.2f}_b{b_b:.2f}_t{tau:.2f}_a{alpha:.2f}"] = f1
            if f1 > best2[0]:
                best2 = (f1, w_b, b_b, tau, alpha)
    cfg_best = FROZEN.copy()
    if len(best2) == 5:
        _, cfg_best["w"], cfg_best["beta"], cfg_best["tau"], cfg_best["alpha"] = best2
    else:
        _, cfg_best["w"], cfg_best["beta"] = best2
    return res, best2[0], cfg_best


def main():
    t0 = time.time()
    print("=" * 88)
    print("  I092: standalone-gain non-transfer mechanism study (analytic)")
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
    n_tr_support = y_tr.sum(0).astype(np.float64)

    cnn = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn + 0.5 * c005)
    print("  building sim ...")
    sim = build_sim(y_tr, n_va, val_pids, train_map, val_map)

    arms = {
        "base1664": np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"],
        "I063_2176D": np.load(os.path.join(REL_DIR, "multi_plm_mlp_val_probs.npz"))["val"],
        "I074_prop": np.load(os.path.join(REL_DIR, "plm1664_prop_multiseed_val.npz"))["val"],
        "L0_asl": np.load(os.path.join(REL_DIR, "i073_L0_asl_val.npz"))["val"],
        "L1_asl_prop": np.load(os.path.join(REL_DIR, "i073_L1_asl_prop_val.npz"))["val"],
    }
    pairs = [("I063_2176D", "base1664"), ("I074_prop", "base1664"),
             ("L1_asl_prop", "L0_asl")]

    bands = np.array_split(np.argsort(-n_tr_support), 4)
    band_names = ["tail-Q1", "Q2", "Q3", "head-Q4"]  # argsort 降序切分→[0]=头，倒序命名
    bands = bands[::-1]  # bands[0]=尾
    report = {"experiment": "I092", "analytic": True, "frozen": FROZEN, "arms": {}}

    # ---- 每臂：冻结配置 + 两阶段重扫；独立分（PLM 单独 SCut）----
    for name, plm in tqdm(arms.items(), desc="arms"):
        f_froz, _, _ = pipe_cfg(deep4, sim, y_tr, y_va, n_va, plm, **FROZEN)
        grid, f_best, cfg_best = grid_eval(deep4, sim, y_tr, y_va, n_va, plm)
        ths_s = search_scut(plm, y_va)
        f_solo = macro_f1(plm, y_va, ths_s)
        report["arms"][name] = {"f1_frozen": f_froz, "f1_grid_best": f_best,
                                "cfg_best": cfg_best, "f1_solo": f_solo,
                                "grid": grid}
        print(f"  {name:11s} solo={f_solo:.4f} fused_frozen={f_froz:.4f} "
              f"fused_grid={f_best:.4f}")

    # ---- 探针 a：传导 Δ（冻结 vs 重扫）----
    probe_a = {}
    for cand, base in pairs:
        a, b = report["arms"][cand], report["arms"][base]
        d_frozen = a["f1_frozen"] - b["f1_frozen"]
        d_grid = a["f1_grid_best"] - b["f1_grid_best"]
        probe_a[f"{cand}_vs_{base}"] = {
            "delta_frozen": d_frozen, "delta_retuned": d_grid,
            "recovered": bool(d_grid - d_frozen >= 0.001 or (d_frozen < 0 <= d_grid)),
        }
        print(f"  [H1] {cand:11s} vs {base:9s} Δfrozen={d_frozen:+.4f} "
              f"Δretuned={d_grid:+.4f}")
    report["probe_a_H1"] = probe_a

    # ---- 探针 b/c：频带传导 + 标签级相关（冻结配置）----
    probe_b, probe_c = {}, {}
    for cand, base in pairs:
        ths_sb = search_scut(arms[base], y_va)
        ths_sc = search_scut(arms[cand], y_va)
        fl_solo_b = per_label_f1(arms[base], y_va, ths_sb)
        fl_solo_c = per_label_f1(arms[cand], y_va, ths_sc)
        _, mx_b, ths_fb = pipe_cfg(deep4, sim, y_tr, y_va, n_va, arms[base], **FROZEN)
        _, mx_c, ths_fc = pipe_cfg(deep4, sim, y_tr, y_va, n_va, arms[cand], **FROZEN)
        fl_fus_b = per_label_f1(mx_b, y_va, ths_fb)
        fl_fus_c = per_label_f1(mx_c, y_va, ths_fc)
        d_solo = fl_solo_c - fl_solo_b
        d_fus = fl_fus_c - fl_fus_b
        key = f"{cand}_vs_{base}"
        probe_b[key] = {
            "solo_by_band": {band_names[i]: float(d_solo[bd].mean()) for i, bd in enumerate(bands)},
            "fused_by_band": {band_names[i]: float(d_fus[bd].mean()) for i, bd in enumerate(bands)},
        }
        pr, _ = pearsonr(d_solo, d_fus)
        sr, _ = spearmanr(d_solo, d_fus)
        top = np.argsort(-d_solo)[:100]
        rest = np.argsort(-d_solo)[100:]
        probe_c[key] = {
            "pearson": float(pr), "spearman": float(sr),
            "top100_gain_labels_fused_delta": float(d_fus[top].mean()),
            "rest_labels_fused_delta": float(d_fus[rest].mean()),
        }
        print(f"  [H2/H3] {key}: soloΔ(head..tail)=" +
              "/".join(f"{probe_b[key]['solo_by_band'][bn]:+.4f}" for bn in band_names[::-1]) +
              f"  corr(P)={pr:+.3f}")
    report["probe_b_H2"] = probe_b
    report["probe_c_H3"] = probe_c

    # ---- 结论判定 ----
    h1 = any(v["recovered"] for v in probe_a.values())
    h2 = all(any(v > 0.001 for v in list(pb["solo_by_band"].values())[:2]) and
             all(v <= 0.001 for v in pb["fused_by_band"].values()) for pb in probe_b.values())
    h3 = all(pc["pearson"] < 0.3 for pc in probe_c.values())
    report["conclusion"] = {
        "H1_protocol_artifact": h1,
        "H2_band_mismatch": h2,
        "H3_weak_residual_coupling": h3,
        "note": "任一机制成立即给出传导判据线索；全不成立则「不传导」为信息论层面的真约束",
    }
    with io.open(os.path.join(REL_DIR, "I092_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n  conclusion:", report["conclusion"])
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
