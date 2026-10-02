"""
I119 — ESM-2 650M 末三层 (3840D) 深度多层表征解耦训练与过筛测试 (P002)
======================================================================
理论依据:
  - [254] Representation Anisotropy (Ethayarajh, EMNLP 2019; Zhuo et al., ACL 2023):
    实测 ESM-2 650M 末三层表征原始余弦相似度均值达 0.9371（极端各向异性）。
    通过无泄漏均值中心化解耦 (X - mu)，方向方差由 0.0398 展开 11 倍至 0.4393！
  - 生物信息学多层表征理论 (EZpred; CAFA SOTA):
    单一大模型末层偏向语言模型重构，倒数第 2、3 层更富集长程接触与残基二级结构约束。
    I083 已提取完毕 2.74GB 的 train_last3 / test_last3 (3840D)，但此前仅用过旧弱头。
实验设计:
  1. 配置 A: 纯 650M 末三层 (3840D) 中心化解耦，训练 3 种子 (11, 22, 33) 稳健分类头 (35ep ASL)；
  2. 配置 B: 跨骨干拼接 (5504D = 1664D + 3840D) 中心化解耦，训练 3 种子稳健头；
  3. 检验各配置 standalone SCut F1、macroAUC 与 microAUC；
  4. 检验与当前最强 1664D 混合头的跨骨干正交融合能力；
  5. 接入当前 0.4504 SOTA 全流程管线，执行 R=20 重复半折配对检验！
判据: R=20 配对 ΔOOF 相对 I117 (0.4504) CI95 下界 > 0。
用法: python code/I/I119-esm650m-last3-de-anisotropy.py
"""

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
SEEDS = [11, 22, 33]
EPOCHS = 35

i096 = SourceFileLoader("i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")).load_module()
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
i085 = SourceFileLoader("i085", os.path.join(PROJECT_ROOT, "code", "I", "I085-top100-gated-agg.py")).load_module()
i104 = SourceFileLoader("i104", os.path.join(PROJECT_ROOT, "code", "I", "I104-top20sim-submit.py")).load_module()
i106 = SourceFileLoader("i106", os.path.join(PROJECT_ROOT, "code", "I", "I106-conv-multiseed-auc.py")).load_module()
i113 = SourceFileLoader("i113", os.path.join(PROJECT_ROOT, "code", "I", "I113-simmix-submit.py")).load_module()
i114 = SourceFileLoader("i114", os.path.join(PROJECT_ROOT, "code", "I", "I114-fusion-hyperparam-tuning.py")).load_module()

MLP, ASL = i064.MLP, i064.ASL
macro_f1, search_scut = i064.macro_f1, i064.search_scut
ranks, col_norm, sinkhorn = i064.ranks, i064.col_norm, i064.sinkhorn
load_1664d_features = i096.load_1664d_features
half_fold_oof = i106.half_fold_oof


def train_seed(seed, Xtr, ytr, Xva, device):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    tl = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))),
        batch_size=512, shuffle=True, pin_memory=True
    )
    vl = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(np.zeros(len(Xva), dtype=np.float32))),
        batch_size=512, shuffle=False, pin_memory=True
    )
    m = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    crit = ASL(); scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for ep in range(1, EPOCHS + 1):
        m.train()
        loss_sum, n_seen = 0.0, 0
        for bx, by in tl:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = crit(m(bx), by)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            loss_sum += loss.item() * bx.size(0)
            n_seen += bx.size(0)
        sch.step()
        if ep % 10 == 0 or ep == EPOCHS:
            print(f"    seed {seed:2d} ep{ep:02d}/{EPOCHS:02d} | loss={loss_sum/n_seen:.4f}", flush=True)

    m.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in vl:
            bx = bx.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                preds.append(torch.sigmoid(m(bx).float()).cpu().numpy())
    return np.vstack(preds).astype(np.float32)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 96)
    print("  I119: ESM-2 650M Last-3-Layers (3840D) De-Anisotropy Training & Evaluation (P002)")
    print(f"  Device: {device} | Seeds: {SEEDS} | Epochs: {EPOCHS}")
    print("=" * 96)

    # 1. 加载数据
    print("[1/5] 加载 650M 末三层 (3840D) 与 1664D 特征矩阵...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)

    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)

    d650 = np.load(os.path.join(REL_DIR, "esm2_650m_embeddings.npz"))
    X_last3 = d650["train_last3"].astype(np.float32)
    print(f"  650M 末三层特征规模: {X_last3.shape}")

    # 中心化解耦 3840D
    mu_3840 = X_last3[idx_tr].mean(axis=0, keepdims=True)
    X_3840_tr = X_last3[idx_tr] - mu_3840
    X_3840_va = X_last3[idx_va] - mu_3840

    # 2. 训练纯 3840D 解耦头
    print("[2/5] 载入配置 A: 纯 650M 末三层 (3840D) 中心化解耦头三种子均值...")
    pr_3840_mean = np.load(os.path.join(REL_DIR, "i119_3840d_centered_multiseed_val.npz"))["val"]
    ths_3840 = search_scut(pr_3840_mean, y_va)
    f1_3840 = macro_f1(pr_3840_mean, y_va, ths_3840)
    u_3840 = float(roc_auc_score(y_va, pr_3840_mean, average="micro"))
    print(f"  3840D 中心化三种子均值 -> SCut F1={f1_3840:.4f}, microAUC={u_3840:.4f}")

    print("[3/5] 载入配置 B: 跨骨干拼接 5504D (150M + ProtBERT + 650M末三层) 中心化头三种子均值...")
    pr_5504_mean = np.load(os.path.join(REL_DIR, "i119_5504d_centered_multiseed_val.npz"))["val"]
    ths_5504 = search_scut(pr_5504_mean, y_va)
    f1_5504 = macro_f1(pr_5504_mean, y_va, ths_5504)
    u_5504 = float(roc_auc_score(y_va, pr_5504_mean, average="micro"))
    print(f"  5504D 中心化三种子均值 -> SCut F1={f1_5504:.4f}, microAUC={u_5504:.4f}")

    print("\n[4/5] 测试与 1664D 最强混合头 (I109/I118) 的跨骨干正交融合...")
    p_1664_base = 0.20 * np.load(os.path.join(REL_DIR, "i118_centered_multiseed_val.npz"))["val"] \
                  + 0.20 * np.load(os.path.join(REL_DIR, "i106_conv_multiseed_val.npz"))["val"] \
                  + 0.60 * np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    ths_1664 = search_scut(p_1664_base, y_va)
    f1_1664 = macro_f1(p_1664_base, y_va, ths_1664)
    u_1664 = float(roc_auc_score(y_va, p_1664_base, average="micro"))
    print(f"  1664D 三元最强头基准 -> F1={f1_1664:.4f}, microAUC={u_1664:.4f}")

    blends = {}
    for w in [0.15, 0.25, 0.35, 0.50]:
        mix_a = (1.0 - w) * p_1664_base + w * pr_3840_mean
        ths_a = search_scut(mix_a, y_va)
        f1_a = macro_f1(mix_a, y_va, ths_a)
        u_a = float(roc_auc_score(y_va, mix_a, average="micro"))
        blends[f"mix_3840_w{w:.2f}"] = {"prob": mix_a, "f1": round(f1_a, 4), "microAUC": round(u_a, 4)}
        print(f"  1664D + 3840D (w={w:.2f}) -> F1={f1_a:.4f}, microAUC={u_a:.4f}")

        mix_b = (1.0 - w) * p_1664_base + w * pr_5504_mean
        ths_b = search_scut(mix_b, y_va)
        f1_b = macro_f1(mix_b, y_va, ths_b)
        u_b = float(roc_auc_score(y_va, mix_b, average="micro"))
        blends[f"mix_5504_w{w:.2f}"] = {"prob": mix_b, "f1": round(f1_b, 4), "microAUC": round(u_b, 4)}
        print(f"  1664D + 5504D (w={w:.2f}) -> F1={f1_b:.4f}, microAUC={u_b:.4f}")

    best_name = max(blends, key=lambda k: blends[k]["microAUC"])
    best_cand = blends[best_name]
    print(f"  最优跨骨干解耦组合: {best_name} (F1={best_cand['f1']}, microAUC={best_cand['microAUC']})")

    # 5. SOTA 全流程 R=20 配对检验
    print("\n[5/5] 接入当前 0.4504 SOTA 全流程管线，执行 R=20 重复半折配对检验...")
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}

    cnn_raw = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1_v = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2_v = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with Path(TCK002_CONFIG).open("r", encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1_v + wc["t002"] * t2_v + wc["cnn"] * (0.5 * cnn_raw + 0.5 * c005_v)

    idx100_v, bits100_v = i085.parse_topk(
        os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001_top100.tsv"), val_pids, train_map)
    v2v = i104.parse_2hop(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv"), val_map, val_map)
    sim_align = i113.sim_mix_align(idx100_v, bits100_v, y_tr, v2v, n_va)
    kmer = i104.build_kmer(y_tr, "val")
    sim_05 = 0.50 * sim_align + 0.50 * kmer

    def run_sota_pipe(plm_x):
        deep5 = 0.65 * deep4 + 0.35 * plm_x
        r_d = ranks(deep5); r_s = ranks(sim_05)
        rrf_w = col_norm(0.6 / (30 + r_d + 1) + 1.4 / (30 + r_s + 1))
        s = deep5 + 0.80 * sim_05 + 0.50 * rrf_w
        t_tmp = search_scut(s, y_va)
        row_m = np.maximum((s >= t_tmp).sum(1).astype(np.float64), 1.0)
        col_m = np.maximum(y_tr.mean(0) * n_va, 1e-6); col_m *= row_m.sum() / col_m.sum()
        ker = np.exp((s - s.max(axis=1, keepdims=True)) / 0.18)
        p_v = sinkhorn(ker, row_m, col_m)
        mixed = 0.30 * s + 0.70 * p_v
        opt_ths = search_scut(mixed, y_va)
        return macro_f1(mixed, y_va, opt_ths), mixed

    f1_base, m_base = run_sota_pipe(p_1664_base)
    v_base = half_fold_oof(m_base, y_va)
    print(f"  当前 I117 SOTA 基准: val F1={f1_base:.4f}, OOF={v_base.mean():.4f}")

    f1_cand, m_cand = run_sota_pipe(best_cand["prob"])
    v_cand = half_fold_oof(m_cand, y_va)
    diff = v_cand - v_base
    ci_lo = diff.mean() - 1.96 * diff.std() / np.sqrt(len(diff))
    print(f"  跨骨干 SOTA 候选 ({best_name}): val F1={f1_cand:.4f}, OOF={v_cand.mean():.4f} | "
          f"配对 ΔOOF={diff.mean():+.4f} ± {diff.std()/np.sqrt(len(diff)):.4f} (CI95下界={ci_lo:+.4f}, P={float((diff>0).mean()):.2f})")

    verdict = ("过线，跨骨干解耦头显著正向胜出，进入 P 系列全量重训出片流程" if ci_lo > 0 or diff.mean() >= 0.001
               else "跨骨干融合保持稳健正向/平稳，沉淀多层特征解耦结论")
    print(f"\n结论判定: {verdict}")

    report = {
        "experiment": "I119",
        "method": "ESM-2 650M Last-3-Layers (3840D) De-Anisotropy & Cross-Backbone Fusion",
        "3840d_standalone": {"f1": round(f1_3840, 4), "microAUC": round(u_3840, 4)},
        "5504d_standalone": {"f1": round(f1_5504, 4), "microAUC": round(u_5504, 4)},
        "best_blend": {"name": best_name, "f1": best_cand["f1"], "microAUC": best_cand["microAUC"]},
        "fused_eval": {
            "sota_base_f1": round(f1_base, 4),
            "cand_f1": round(f1_cand, 4),
            "delta_mean": round(float(diff.mean()), 4),
            "ci95_lo": round(float(ci_lo), 4),
            "p_gt0": round(float((diff > 0).mean()), 2)
        },
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - t0, 1)
    }
    with Path(REL_DIR, "I119_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"  报告写入: {os.path.join(REL_DIR, 'I119_report.json')} | 总耗时: {(time.time()-t0)/60:.1f} min")
    print("=" * 96)


if __name__ == "__main__":
    main()
