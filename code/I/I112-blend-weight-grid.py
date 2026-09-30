from pathlib import Path
"""
I112 — 混合头融合权重 w 网格 × sim 代际混合（纯后处理收官刀）
===============================================================
背景:
  - I109/I111a: 混合头（0.25 收敛 + 0.75 平滑）microAUC 0.8856 过筛、融合配对显著为正，
    但整个融合矩阵都固定 w_plm=0.35——该权重是 I064 为弱头调的锁定值；
    I098a 已证强头在 0.35 失配，混合头强度介于两者之间，w 可能需要重调（未试）。
  - sim 支同理：I085 top20 完全替换 top3，两代 sim 的混合（beta*top20+(1-beta)*top3）
    未试——I109 的偏差互补逻辑同样适用。
网格:
  A. w_plm ∈ {0.25, 0.30, 0.35, 0.40, 0.45} × 混合头（alpha=0.25）；
  B. sim_beta ∈ {0.5, 0.75, 1.0} × top3/top20 混合（w_plm 回 0.35 或 A 的最优）。
判据: R=20 配对 ΔOOF，CI95 下界 > 0 或 ≥ +0.001 换配置出片。
输出: rel/I112_report.json（纯后处理约 15 min）
用法: python code/I/I112-blend-weight-grid.py
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
SEED = 42

i096 = SourceFileLoader("i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")).load_module()
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
i085 = SourceFileLoader("i085", os.path.join(PROJECT_ROOT, "code", "I", "I085-top100-gated-agg.py")).load_module()
i104 = SourceFileLoader("i104", os.path.join(PROJECT_ROOT, "code", "I", "I104-top20sim-submit.py")).load_module()
i106 = SourceFileLoader("i106", os.path.join(PROJECT_ROOT, "code", "I", "I106-conv-multiseed-auc.py")).load_module()
full_pipe = i064.full_pipe
build_sim = i096.build_sim
half_fold_oof = i106.half_fold_oof


def main():
    t0 = time.time()
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=SEED)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]

    weak = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    conv = np.load(os.path.join(REL_DIR, "i106_conv_multiseed_val.npz"))["val"]
    blend = 0.25 * conv + 0.75 * weak

    print("=" * 88)
    print("  I112: Blend-Head Fusion Weight Grid x sim Generational Mix")
    print("=" * 88)
    print("[1/3] 构建 deep4 与两代 sim...")
    cnn_raw = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn_raw + 0.5 * c005)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}

    idx100_v, bits100_v = i085.parse_topk(
        os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001_top100.tsv"), val_pids, train_map)
    sim_top20 = i085.agg_topk(idx100_v, bits100_v, y_tr, 20)
    sim_top3 = i085.agg_topk(idx100_v, bits100_v, y_tr, 3)
    hit_v = idx100_v[:, 0] >= 0
    v2v = i104.parse_2hop(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv"), val_map, val_map)
    kmer = i104.build_kmer(y_tr, "val")
    sim_align20 = i104.two_hop(sim_top20, hit_v, v2v, len(idx_va))
    sim_align3 = i104.two_hop(sim_top3, hit_v, v2v, len(idx_va))

    sim_variants = {
        "top20(现状)": 0.4 * sim_align20 + 0.6 * kmer,
        "mix_b0.75": 0.4 * (0.75 * sim_align20 + 0.25 * sim_align3) + 0.6 * kmer,
        "mix_b0.50": 0.4 * (0.5 * sim_align20 + 0.5 * sim_align3) + 0.6 * kmer,
    }

    print("[2/3] 基线（弱头 w=0.35 + top20）...")
    _, _, mixed_base, _, _ = full_pipe(0.65 * deep4 + 0.35 * weak,
                                       sim_variants["top20(现状)"], y_tr, y_va, len(idx_va))
    v_base = half_fold_oof(mixed_base, y_va)

    print("[3/3] 网格：A 组 w_plm（blend 头）× B 组 sim 混合...")
    results = []
    combos = []
    for w in (0.25, 0.30, 0.35, 0.40, 0.45):
        combos.append((f"A_w{w:.2f}", (1 - w) * deep4 + w * blend, "top20(现状)"))
    for bname in ("mix_b0.75", "mix_b0.50"):
        for w in (0.35, 0.30):
            combos.append((f"B_{bname}_w{w:.2f}", (1 - w) * deep4 + w * blend, bname))
            combos.append((f"B_{bname}_weak_w{w:.2f}", (1 - w) * deep4 + w * weak, bname))

    best = None
    for name, d5, sname in combos:
        _, _, mixed_new, _, _ = full_pipe(d5, sim_variants[sname], y_tr, y_va, len(idx_va))
        v_new = half_fold_oof(mixed_new, y_va)
        d = v_new - v_base
        d_mean, d_se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        ci_lo = d_mean - 1.96 * d_se
        rec = {"name": name, "sim": sname, "delta_mean": round(d_mean, 4),
               "delta_se": round(d_se, 4), "ci95_lo": round(ci_lo, 4),
               "p_gt0": round(float((d > 0).mean()), 3)}
        results.append(rec)
        mark = ""
        if best is None or rec["delta_mean"] > best["delta_mean"]:
            best = rec
            mark = " (*)"
        print(f"  {name:26s} | ΔOOF={d_mean:+.4f} ± {d_se:.4f} | CI95下界={ci_lo:+.4f}"
              f" | P={rec['p_gt0']:.2f}{mark}", flush=True)

    sig = [r for r in results if r["ci95_lo"] > 0]
    if sig:
        top = max(sig, key=lambda r: r["delta_mean"])
        if top["delta_mean"] > 0.0002:
            verdict = f"最优 {top['name']}（ΔOOF {top['delta_mean']:+.4f}）超越 I110 配置，出新片"
        else:
            verdict = f"最优 {top['name']}（ΔOOF {top['delta_mean']:+.4f}）与 I110 持平量级，维持现片"
    else:
        verdict = f"全网格未显著（最优 {best['name']} ΔOOF {best['delta_mean']:+.4f}），维持 I110 现片"
    report = {"experiment": "I112", "method": "blend-head w grid x sim generational mix",
              "results": results, "best": best, "conclusion": verdict,
              "elapsed_seconds": round(time.time() - t0, 1)}
    with Path(REL_DIR, "I112_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  结论: {verdict}")
    print(f"  报告: {os.path.join(REL_DIR, 'I112_report.json')} | {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
