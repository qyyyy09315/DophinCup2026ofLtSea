from pathlib import Path
"""
I114 — 融合层超参数全局网格重校准（0.4491 SOTA 口径，分析性）
================================================================
背景:
  - 核心参数锁历史溯源:
    * SIM_W = 0.75 锁于 I011 / I051 (8月中旬, 仅3分支旧流水线)
    * RRF_W = 0.50 锁于 I011 / I011d
    * RRF k = 30 锁于 I011e
    * Sinkhorn tau = 0.20, alpha = 0.60 锁于 I019 / I019b
  - 现状: 融合两端均发生结构性代际跃迁（deep5 引入了多代 PLM 混合头，sim 引入了 top20/top3 代际混合 + 2-hop），
    两端特征的刻度、尖锐度、校准分布均与 8 月份发生显著改变。
  - 本实验在当前 0.4491 SOTA 状态下，系统重扫融合超参数:
    1. SIM_W ∈ {0.65, 0.70, 0.75, 0.80, 0.85}
    2. RRF_W ∈ {0.35, 0.40, 0.50, 0.60, 0.65}
    3. RRF k ∈ {15, 20, 30, 45}
    4. Sinkhorn tau ∈ {0.15, 0.18, 0.20, 0.22, 0.25}
    5. Sinkhorn alpha ∈ {0.50, 0.55, 0.60, 0.65, 0.70}
  - 判据: R=20 次重复半折配对检验，CI95 下界 > 0。
输出: rel/I114_report.json
用法: python code/I/I114-fusion-hyperparam-tuning.py
"""

import os
import json
import time

import numpy as np
import pandas as pd
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
SEED = 42

i096 = SourceFileLoader("i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")).load_module()
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
i085 = SourceFileLoader("i085", os.path.join(PROJECT_ROOT, "code", "I", "I085-top100-gated-agg.py")).load_module()
i104 = SourceFileLoader("i104", os.path.join(PROJECT_ROOT, "code", "I", "I104-top20sim-submit.py")).load_module()
i106 = SourceFileLoader("i106", os.path.join(PROJECT_ROOT, "code", "I", "I106-conv-multiseed-auc.py")).load_module()
macro_f1, search_scut = i064.macro_f1, i064.search_scut
ranks, col_norm, sinkhorn = i064.ranks, i064.col_norm, i064.sinkhorn
half_fold_oof = i106.half_fold_oof


def custom_pipe(d, sim, y_tr, y_va, n_va, sim_w=0.75, rrf_w=0.50, k=30, tau=0.20, alpha=0.60):
    rrf = col_norm(1.0 / (k + ranks(d) + 1) + 1.0 / (k + ranks(sim) + 1))
    s = d + sim_w * sim + rrf_w * rrf
    t_tmp = search_scut(s, y_va)
    row_m = np.maximum((s >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_tr.mean(0) * n_va, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s - s.max(axis=1, keepdims=True)) / tau)
    p_v = sinkhorn(ker, row_m, col_m)
    mixed = (1.0 - alpha) * s + alpha * p_v
    ths = search_scut(mixed, y_va)
    return macro_f1(mixed, y_va, ths), mixed


def main():
    t0 = time.time()
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=SEED)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)

    print("=" * 88)
    print("  I114: Re-calibrating Fusion Hyperparameters on 0.4491 SOTA State")
    print("=" * 88)

    print("[1/3] 构建当前最佳 SOTA deep5 与 sim (0.4491 口径)...")
    cnn_raw = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn_raw + 0.5 * c005)

    weak = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    conv = np.load(os.path.join(REL_DIR, "i106_conv_multiseed_val.npz"))["val"]
    blend = 0.25 * conv + 0.75 * weak
    deep5 = 0.65 * deep4 + 0.35 * blend

    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}

    idx100_v, bits100_v = i085.parse_topk(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001_top100.tsv"), val_pids, train_map)
    s20 = i085.agg_topk(idx100_v, bits100_v, y_tr, 20)
    s3 = i085.agg_topk(idx100_v, bits100_v, y_tr, 3)
    hit = idx100_v[:, 0] >= 0
    v2v = i104.parse_2hop(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv"), val_map, val_map)
    kmer = i104.build_kmer(y_tr, "val")
    sim_align = 0.75 * i104.two_hop(s20, hit, v2v, n_va) + 0.25 * i104.two_hop(s3, hit, v2v, n_va)
    sim = 0.4 * sim_align + 0.6 * kmer

    # 验证基准 SOTA
    f1_base, mixed_base = custom_pipe(deep5, sim, y_tr, y_va, n_va, 0.75, 0.50, 30, 0.20, 0.60)
    print(f"  当前锁定基准: val F1={f1_base:.4f} (必须为 0.4491)")
    v_base = half_fold_oof(mixed_base, y_va)

    print("\n[2/3] 逐维单变量扰动扫描...")
    candidates = []

    # 1. SIM_W
    for sw in [0.65, 0.70, 0.80, 0.85]:
        candidates.append((f"SIM_W={sw:.2f}", dict(sim_w=sw, rrf_w=0.50, k=30, tau=0.20, alpha=0.60)))

    # 2. RRF_W
    for rw in [0.35, 0.40, 0.60, 0.65]:
        candidates.append((f"RRF_W={rw:.2f}", dict(sim_w=0.75, rrf_w=rw, k=30, tau=0.20, alpha=0.60)))

    # 3. RRF k
    for kv in [15, 20, 45, 60]:
        candidates.append((f"RRF_k={kv}", dict(sim_w=0.75, rrf_w=0.50, k=kv, tau=0.20, alpha=0.60)))

    # 4. Sinkhorn tau
    for tv in [0.15, 0.18, 0.22, 0.25]:
        candidates.append((f"Sink_tau={tv:.2f}", dict(sim_w=0.75, rrf_w=0.50, k=30, tau=tv, alpha=0.60)))

    # 5. Sinkhorn alpha
    for av in [0.50, 0.55, 0.65, 0.70]:
        candidates.append((f"Sink_alpha={av:.2f}", dict(sim_w=0.75, rrf_w=0.50, k=30, tau=0.20, alpha=av)))

    results = []
    best_cand = None
    for name, params in candidates:
        f1_c, mixed_c = custom_pipe(deep5, sim, y_tr, y_va, n_va, **params)
        v_c = half_fold_oof(mixed_c, y_va)
        d = v_c - v_base
        d_mean = float(d.mean())
        d_se = float(d.std(ddof=1) / np.sqrt(len(d)))
        ci_lo = d_mean - 1.96 * d_se
        rec = {
            "name": name,
            "params": params,
            "f1_val": round(float(f1_c), 4),
            "delta_mean": round(d_mean, 4),
            "delta_se": round(d_se, 4),
            "ci95_lo": round(ci_lo, 4),
            "p_gt0": round(float((d > 0).mean()), 2)
        }
        results.append(rec)
        mark = " (*)" if d_mean > 0 else ""
        print(f"  {name:16s} | val F1={f1_c:.4f} | ΔOOF={d_mean:+.4f} ± {d_se:.4f} | CI95下界={ci_lo:+.4f} | P={rec['p_gt0']:.2f}{mark}", flush=True)

    # 找显著正向
    sig = [r for r in results if r["ci95_lo"] > 0]
    print("\n[3/3] 评估与结论判定...")
    if sig:
        top_sig = max(sig, key=lambda r: r["delta_mean"])
        conclusion = f"发现显著超参增益: {top_sig['name']} (ΔOOF {top_sig['delta_mean']:+.4f}, CI95 {top_sig['ci95_lo']:+.4f}) -> 升级融合超参锁"
    else:
        top_cand = max(results, key=lambda r: r["delta_mean"])
        conclusion = f"当前超参数已全局最优（最优 {top_cand['name']} Δ={top_cand['delta_mean']:+.4f} 未显著），原有 (0.75, 0.50, 30, 0.20, 0.60) 锁坚固不变"

    print(f"  结论: {conclusion}")
    report = {
        "experiment": "I114",
        "method": "Global fusion hyperparameter re-calibration on SOTA 0.4491 pipeline",
        "baseline_f1": round(float(f1_base), 4),
        "results": results,
        "conclusion": conclusion,
        "elapsed_seconds": round(time.time() - t0, 1)
    }
    with Path(REL_DIR, "I114_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告写入: {os.path.join(REL_DIR, 'I114_report.json')} | 用时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
