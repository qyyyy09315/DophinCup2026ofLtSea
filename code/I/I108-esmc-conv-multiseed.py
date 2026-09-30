"""
I108 — ESM-C-300M 冻结均值池化 × 收敛头 × microAUC 筛子（PLM 队列 B 刀）
==========================================================================
背景:
  - I102 准则: 融合传导 ~ microAUC（筛子 0.8783 = 35ep 三种子现支）；
    D 刀（I106）收敛头三种子均值 microAUC 0.8710 差 0.0073 未过；
    A 刀（I107）650M 嵌入双配置结果见 rel/I107_report.json。
  - B 刀换**更新一代骨干**：ESM-C-300M（EvolutionaryScale 2024-12）冻结均值池化
    960D/残基，替换 1664D 中的 ESM2-150M 640D 槽（1552D=960+1024 ProtBERT），
    同一收敛头协议（80ep+SCut 早停）× 三种子过筛。规则合规：预训练模型明文豁免。
  - 依赖: esm 3.2.1 SDK + transformers 4.48.1（2026-09-29 装）；HF_ENDPOINT=hf-mirror。
阶段:
  1. 嵌入提取（首跑约 1-2h，npz 缓存断点级复用）: train+test 均值池化 960D；
  2. 收敛头三种子 × {ESMC1552, ESMC960 单独} 双配置 → microAUC 筛子；
  3. 过筛者进 R=20 配对融合检验（同 I107）。
输出: rel/i108_esmc_embed.npz, rel/I108_report.json, rel/i108_{feat}_multiseed_val.npz
用法: python code/I/I108-esmc-conv-multiseed.py            # 全流程
      python code/I/I108-esmc-conv-multiseed.py --extract-only
      python code/I/I108-esmc-conv-multiseed.py --head-only  # 嵌入已缓存
"""

import os
import sys
import json
import time
import argparse

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
SEEDS = [11, 22, 33]
MICROAUC_BAR = 0.8783
EMB_PATH = os.path.join(REL_DIR, "i108_esmc_embed.npz")

i096 = SourceFileLoader("i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")).load_module()
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
i106 = SourceFileLoader("i106", os.path.join(PROJECT_ROOT, "code", "I", "I106-conv-multiseed-auc.py")).load_module()
MLP = i064.MLP
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe
load_1664d_features = i096.load_1664d_features
build_sim = i096.build_sim
train_one_seed = i106.train_one_seed
half_fold_oof = i106.half_fold_oof


@torch.no_grad()
def extract_embeddings(sequences, batch=24):
    """ESM-C-300M 均值池化 960D（残基位，去 BOS/EOS 由模型 padding 处理）"""
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein
    model = ESMC.from_pretrained("esmc_300m").cuda().eval()
    outs = []
    for s in tqdm(range(0, len(sequences), batch), desc="ESMC extract", ncols=90):
        chunk = sequences[s:s + batch]
        toks = [model.encode(ESMProtein(sequence=str(sq))).sequence for sq in chunk]
        mx = max(t.shape[0] for t in toks)
        pad = torch.stack([torch.nn.functional.pad(t, (0, mx - t.shape[0])) for t in toks]).cuda()
        r = model(pad)
        emb = r.embeddings  # (B, L, 960)
        for i, t in enumerate(toks):
            L = t.shape[0]
            outs.append(emb[i, :L].mean(0).float().cpu().numpy())
        if s % (batch * 20) == 0:
            torch.cuda.empty_cache()
    return np.vstack(outs).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--head-only", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print("  I108: ESM-C-300M frozen mean-pool x Converged Head x 3-Seed x microAUC Sieve")
    print("=" * 88)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)

    # 1. 嵌入提取（缓存）
    if not args.head_only:
        if os.path.exists(EMB_PATH):
            d = np.load(EMB_PATH)
            if d["train_emb"].shape[0] == len(train_df):
                print(f"[1/3] 嵌入缓存命中: {EMB_PATH}")
            else:
                args.head_only = False
        if not os.path.exists(EMB_PATH):
            print("[1/3] 提取 ESM-C-300M 嵌入（首跑约 1-2h）...")
            tr_emb = extract_embeddings(train_df["sequence"].values)
            te_emb = extract_embeddings(test_df["sequence"].values)
            np.savez_compressed(EMB_PATH, train_emb=tr_emb, test_emb=te_emb)
            print(f"  已存 {EMB_PATH}: train {tr_emb.shape} test {te_emb.shape} "
                  f"({time.time()-t0:.0f}s)")
    if args.extract_only:
        return

    d = np.load(EMB_PATH)
    tr_emb, te_emb = d["train_emb"], d["test_emb"]

    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    X1664 = load_1664d_features()

    feats = {
        "esmc1552": np.concatenate([tr_emb, X1664[:, 640:]], axis=1),  # ESMC960 + ProtBERT1024
        "esmc960": tr_emb,
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
            np.savez_compressed(os.path.join(REL_DIR, f"i108_{name}_s{s}.npz"), val=pr)
            print(f"  {name} seed{s}: F1={best_f1:.4f}@ep{best_ep}", flush=True)
        mean_p = np.mean(seed_probs, axis=0)
        np.savez_compressed(os.path.join(REL_DIR, f"i108_{name}_multiseed_val.npz"), val=mean_p)
        ths = search_scut(mean_p, y_va)
        f1 = macro_f1(mean_p, y_va, ths)
        auc_m = float(roc_auc_score(y_va, mean_p, average="macro"))
        auc_u = float(roc_auc_score(y_va, mean_p, average="micro"))
        ok = auc_u > MICROAUC_BAR
        print(f"  [{name}] F1={f1:.4f} | macroAUC={auc_m:.4f} | microAUC={auc_u:.4f} -> "
              f"{'过筛' if ok else '未过筛'}")
        results[name] = {"f1_scut": round(f1, 4), "auc_macro": round(auc_m, 4),
                         "auc_micro": round(auc_u, 4), "sieve_pass": ok}

    passed = [(n, r) for n, r in results.items() if r["sieve_pass"]]
    if passed:
        name, r = max(passed, key=lambda x: x[1]["auc_micro"])
        print(f"\n[3/3] 过筛者 {name} 进融合配对检验（R=20）...")
        mean_p = np.load(os.path.join(REL_DIR, f"i108_{name}_multiseed_val.npz"))["val"]
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
        dd = v_new - v_base
        d_mean, d_se = float(dd.mean()), float(dd.std(ddof=1) / np.sqrt(len(dd)))
        ci_lo = d_mean - 1.96 * d_se
        verdict = ("过线，换 PLM 支进 I100 全量出提交流程" if (ci_lo > 0 or d_mean >= 0.001)
                   else "筛子过但融合未证实，B 刀关闭")
        print(f"  配对 ΔOOF(R=20): {d_mean:+.4f} ± {d_se:.4f} | CI95下界 {ci_lo:+.4f} -> {verdict}")
        results["_fused"] = {"delta_mean": round(d_mean, 4), "delta_se": round(d_se, 4),
                             "ci95_lo": round(ci_lo, 4), "verdict": verdict}
        conclusion = verdict
    else:
        conclusion = "两配置均未过 microAUC 筛，B 刀关闭（PLM 队列 D/A/B 三刀全灭）"
        print(f"\n[3/3] 无配置过筛 -> {conclusion}")

    report = {"experiment": "I108", "method": "ESM-C-300M frozen mean-pool x converged head x 3-seed",
              "results": results, "bar": MICROAUC_BAR, "conclusion": conclusion,
              "elapsed_seconds": round(time.time() - t0, 1)}
    with Path(os.path.join(REL_DIR, "I108_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 88)
    print(f"  I108 完成! -> {conclusion} | 总用时 {(time.time()-t0)/60:.1f} min")
    print("=" * 88)


if __name__ == "__main__":
    main()
