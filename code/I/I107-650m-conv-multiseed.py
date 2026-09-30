"""
I107 — 650M 嵌入 × 收敛头三种子 × microAUC 筛子（PLM 队列 A 刀）
==================================================================
背景:
  - I084 关闭 650M 嵌入时用的是旧 35ep 弱头协议（最优 0.4474 < 0.4485 未过线）；
    I096 证明收敛协议本身值 +0.0407，「收敛 × 三种子 × 650M 特征」组合从未测过。
  - 判据同 I106（I102 准则）: 头概率均值 microAUC > 0.8783（35ep 三种子现支）才进融合。
配置（顺序跑，各自过筛）:
  A1: 1664D + 650M last (2944D)   —— I084 主配置的收敛协议复检
  A2: 650M last 单独 (1280D)       —— 纯 650M 槽
输出: rel/I107_report.json, rel/i107_{feat}_multiseed_val.npz
用法: python code/I/I107-650m-conv-multiseed.py   (~70 min)
"""

import os
import json
import copy
import time

import numpy as np
import pandas as pd
import torch
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
SEEDS = [11, 22, 33]
MICROAUC_BAR = 0.8783

i096 = SourceFileLoader("i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")).load_module()
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
MLP, ASL = i064.MLP, i064.ASL
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe
load_1664d_features = i096.load_1664d_features
build_sim = i096.build_sim
i106 = SourceFileLoader("i106", os.path.join(PROJECT_ROOT, "code", "I", "I106-conv-multiseed-auc.py")).load_module()
train_one_seed = i106.train_one_seed
half_fold_oof = i106.half_fold_oof


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print("  I107: 650M-Embed x Converged Head x 3-Seed x microAUC Sieve")
    print(f"  筛子: microAUC > {MICROAUC_BAR} | seeds {SEEDS}")
    print("=" * 88)

    print("[1/3] 加载特征...")
    X1664 = load_1664d_features()
    d650 = np.load(os.path.join(REL_DIR, "esm2_650m_embeddings.npz"))
    X_last, X_last3 = d650["train_last"].astype(np.float32), d650["train_last3"].astype(np.float32)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]

    feats = {
        "concat2944": np.concatenate([X1664, X_last], axis=1),
        "last1280": X_last,
    }

    results = {}
    for name, X in feats.items():
        print(f"\n[2/3] 配置 {name} ({X.shape[1]}D) 三种子收敛头...")
        X_tr, X_va = X[idx_tr], X[idx_va]
        seed_probs = []
        for s in SEEDS:
            pr, best_ep, best_f1 = train_one_seed(s, X_tr, y_tr, X_va, y_va, device,
                                               in_dim=X.shape[1])
            seed_probs.append(pr)
            np.savez_compressed(os.path.join(REL_DIR, f"i107_{name}_s{s}.npz"), val=pr)
            print(f"  {name} seed{s}: F1={best_f1:.4f}@ep{best_ep}", flush=True)
        mean_p = np.mean(seed_probs, axis=0)
        np.savez_compressed(os.path.join(REL_DIR, f"i107_{name}_multiseed_val.npz"), val=mean_p)

        ths = search_scut(mean_p, y_va)
        f1 = macro_f1(mean_p, y_va, ths)
        auc_m = float(roc_auc_score(y_va, mean_p, average="macro"))
        auc_u = float(roc_auc_score(y_va, mean_p, average="micro"))
        ok = auc_u > MICROAUC_BAR
        print(f"  [{name}] F1={f1:.4f} | macroAUC={auc_m:.4f} | microAUC={auc_u:.4f} -> "
              f"{'过筛' if ok else '未过筛'}")
        results[name] = {"f1_scut": round(f1, 4), "auc_macro": round(auc_m, 4),
                         "auc_micro": round(auc_u, 4), "sieve_pass": ok}

    # 过筛者进融合配对检验（至多取 microAUC 最高的一个）
    passed = [(n, r) for n, r in results.items() if r["sieve_pass"]]
    if passed:
        name, r = max(passed, key=lambda x: x[1]["auc_micro"])
        print(f"\n[3/3] 过筛者 {name} 进融合配对检验（R=20）...")
        mean_p = np.load(os.path.join(REL_DIR, f"i107_{name}_multiseed_val.npz"))["val"]
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
        sim = build_sim(y_tr, len(idx_va), val_pids, train_map, val_map)
        plm_weak = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
        _, _, mixed_new, _, _ = full_pipe(0.65 * deep4 + 0.35 * mean_p, sim, y_tr, y_va, len(idx_va))
        _, _, mixed_base, _, _ = full_pipe(0.65 * deep4 + 0.35 * plm_weak, sim, y_tr, y_va, len(idx_va))
        v_new, v_base = half_fold_oof(mixed_new, y_va), half_fold_oof(mixed_base, y_va)
        d = v_new - v_base
        d_mean, d_se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        ci_lo = d_mean - 1.96 * d_se
        verdict = ("过线，换 PLM 支进 I100 全量出提交流程" if (ci_lo > 0 or d_mean >= 0.001)
                   else "筛子过但融合未证实，A 刀关闭")
        print(f"  配对 ΔOOF(R=20): {d_mean:+.4f} ± {d_se:.4f} | CI95下界 {ci_lo:+.4f} -> {verdict}")
        results["_fused"] = {"delta_mean": round(d_mean, 4), "delta_se": round(d_se, 4),
                             "ci95_lo": round(ci_lo, 4), "verdict": verdict}
        conclusion = verdict
    else:
        conclusion = "两配置均未过 microAUC 筛，A 刀关闭（B 刀 ESM-C 为下一候选）"
        print(f"\n[3/3] 无配置过筛 -> {conclusion}")

    report = {"experiment": "I107", "method": "650M-embed x converged head x 3-seed x microAUC sieve",
              "results": results, "bar": MICROAUC_BAR, "conclusion": conclusion,
              "elapsed_seconds": round(time.time() - t0, 1)}
    with Path(os.path.join(REL_DIR, "I107_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 88)
    print(f"  I107 完成! -> {conclusion}")
    print(f"  报告: {os.path.join(REL_DIR, 'I107_report.json')} | 总用时 {(time.time()-t0)/60:.1f} min")
    print("=" * 88)


if __name__ == "__main__":
    main()
