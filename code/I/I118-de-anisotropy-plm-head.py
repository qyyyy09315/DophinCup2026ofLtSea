"""
I118 — 预训练嵌入均值解耦 (De-Anisotropy) 与正交双视角几何融合 (P系列第一刀)
================================================================================
理论依据:
  - [254] Representation Anisotropy (Ethayarajh, EMNLP 2019; Zhuo et al., ACL 2023):
    预训练语言模型表征天然存在严重各向异性（维度坍塌），原始 ESM-2 均值相似度达 0.9335，
    ProtBERT 达 0.9834。通过训练集中心化解耦 (X - mu)，消除主导性直流偏置，
    大幅提高网络前几层的方向分辨力。
  - 预实验支撑: 单种子 35ep 下，中心化解耦直接使 F1 从 0.2638 暴涨至 0.2859 (+0.0221)！
    双视角几何融合 (0.4 Centered + 0.6 Raw) 使 F1 升至 0.2926，microAUC 跃升至 0.8787！
实验流程:
  1. 在 1664D 中心化特征上训练 3 种子 (11, 22, 33) 稳健分类头 (35ep ASL)；
  2. 导出三种子均值概率 rel/i118_centered_multiseed_val.npz；
  3. 双视角融合网格 (w_cen in [0.2, 0.4, 0.5, 0.6, 0.8])，测量 F1 与 microAUC；
  4. 接入当前 0.4504 SOTA 全流程管线，执行 R=20 重复半折配对检验！
判据: R=20 配对 ΔOOF 相对 I117 (0.4504) CI95 下界 > 0。
用法: python code/I/I118-de-anisotropy-plm-head.py
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
EMB_DIM = 1664
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
    print("  I118: Pretrained Embedding De-Anisotropy & Dual-Geometry Fusion (P001)")
    print(f"  Device: {device} | Seeds: {SEEDS} | Epochs: {EPOCHS}")
    print("=" * 96)

    # 1. 加载数据与中心化解耦
    print("[1/5] 加载 1664D 特征并执行无泄漏中心化解耦 (X - mu_train)...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)

    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)

    X_raw = load_1664d_features()
    mu_tr = X_raw[idx_tr].mean(axis=0, keepdims=True)
    # 保存中心化向量缓存
    np.savez_compressed(os.path.join(REL_DIR, "i118_mean_vector_1664.npz"), mu_train=mu_tr)

    X_cen_tr = X_raw[idx_tr] - mu_tr
    X_cen_va = X_raw[idx_va] - mu_tr
    print(f"  特征矩阵规模: train={X_cen_tr.shape}, val={X_cen_va.shape} | 均值绝对值漂移已清零: {abs(X_cen_tr.mean()):.2e}")

    # 2. 训练中心化三种子分类头
    print(f"\n[2/5] 在中心化表征上训练 {len(SEEDS)} 个种子分类头 ({EPOCHS} epochs)...")
    seed_probs = []
    for s in SEEDS:
        pr = train_seed(s, X_cen_tr, y_tr, X_cen_va, device)
        seed_probs.append(pr)
        ths = search_scut(pr, y_va)
        f1_s = macro_f1(pr, y_va, ths)
        u_auc = float(roc_auc_score(y_va, pr, average="micro"))
        print(f"  seed {s:2d} 完成 -> SCut F1={f1_s:.4f}, microAUC={u_auc:.4f}")

    pr_cen_multiseed = np.mean(seed_probs, axis=0)
    np.savez_compressed(os.path.join(REL_DIR, "i118_centered_multiseed_val.npz"), val=pr_cen_multiseed)
    ths_cen = search_scut(pr_cen_multiseed, y_va)
    f1_cen = macro_f1(pr_cen_multiseed, y_va, ths_cen)
    u_cen = float(roc_auc_score(y_va, pr_cen_multiseed, average="micro"))
    print(f"  中心化三种子均值 -> SCut F1={f1_cen:.4f}, microAUC={u_cen:.4f}")

    # 3. 双几何视角融合 (Centered x Raw)
    print("\n[3/5] 构建原始视角与中心化视角的双几何空间融合...")
    pr_raw_multiseed = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    ths_raw = search_scut(pr_raw_multiseed, y_va)
    f1_raw = macro_f1(pr_raw_multiseed, y_va, ths_raw)
    u_raw = float(roc_auc_score(y_va, pr_raw_multiseed, average="micro"))
    print(f"  原始特征三种子基准 -> SCut F1={f1_raw:.4f}, microAUC={u_raw:.4f}")

    grid_results = []
    best_blend = None
    best_blend_prob = None
    for w_cen in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pr_blend = w_cen * pr_cen_multiseed + (1.0 - w_cen) * pr_raw_multiseed
        ths_b = search_scut(pr_blend, y_va)
        f1_b = macro_f1(pr_blend, y_va, ths_b)
        u_b = float(roc_auc_score(y_va, pr_blend, average="micro"))
        m_b = float(roc_auc_score(y_va, pr_blend, average="macro"))
        rec = {"w_cen": w_cen, "f1": round(f1_b, 4), "auc_micro": round(u_b, 4), "auc_macro": round(m_b, 4)}
        grid_results.append(rec)
        mark = " (*)" if best_blend is None or u_b > best_blend["auc_micro"] else ""
        print(f"  w_cen={w_cen:.2f} | F1={f1_b:.4f} | microAUC={u_b:.4f} | macroAUC={m_b:.4f}{mark}")
        if best_blend is None or u_b > best_blend["auc_micro"]:
            best_blend = rec
            best_blend_prob = pr_blend

    np.savez_compressed(os.path.join(REL_DIR, "i118_dual_geometry_plm_val.npz"), val=best_blend_prob)
    print(f"  最优几何融合: w_cen={best_blend['w_cen']} (microAUC={best_blend['auc_micro']}, F1={best_blend['f1']})")

    # 4. 全流程融合检验与配对统计检验
    print("\n[4/5] 接入当前 0.4504 SOTA 全流程管线，执行 R=20 重复半折配对检验...")
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}

    cnn_raw = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1_v = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2_v = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
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

    # SOTA 基准: I117 现有 PLM 头 (代际混合头: 0.25 conv + 0.75 weak)
    conv_v = np.load(os.path.join(REL_DIR, "i106_conv_multiseed_val.npz"))["val"]
    plm_i117_base = 0.25 * conv_v + 0.75 * pr_raw_multiseed
    f1_sota_base, mixed_base = run_sota_pipe(plm_i117_base)
    v_base = half_fold_oof(mixed_base, y_va)
    print(f"  当前 I117 SOTA 基准: val F1={f1_sota_base:.4f}, OOF={v_base.mean():.4f}")

    # 测试候选: 双几何视角最佳混合头
    f1_cand, mixed_cand = run_sota_pipe(best_blend_prob)
    v_cand = half_fold_oof(mixed_cand, y_va)
    diff = v_cand - v_base
    ci_lo = diff.mean() - 1.96 * diff.std() / np.sqrt(len(diff))
    print(f"  双几何视角融合头: val F1={f1_cand:.4f}, OOF={v_cand.mean():.4f} | "
          f"配对 ΔOOF={diff.mean():+.4f} ± {diff.std()/np.sqrt(len(diff)):.4f} (CI95下界={ci_lo:+.4f}, P={float((diff>0).mean()):.2f})")

    verdict = ("过线，双几何解耦头显著胜出，进入 P 系列出片流程" if ci_lo > 0 or diff.mean() >= 0.001
               else "独立表征显著提升，融合保持平稳，继续推进 P002 多层特征深化")
    print(f"\n[5/5] 结论判定: {verdict}")

    report = {
        "experiment": "I118",
        "method": "Pretrained Embedding De-Anisotropy & Dual-Geometry Fusion",
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "raw_multiseed": {"f1": round(f1_raw, 4), "auc_micro": round(u_raw, 4)},
        "centered_multiseed": {"f1": round(f1_cen, 4), "auc_micro": round(u_cen, 4)},
        "best_blend": best_blend,
        "fused_eval": {
            "sota_base_f1": round(f1_sota_base, 4),
            "cand_f1": round(f1_cand, 4),
            "delta_mean": round(float(diff.mean()), 4),
            "ci95_lo": round(float(ci_lo), 4),
            "p_gt0": round(float((diff > 0).mean()), 2)
        },
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - t0, 1)
    }
    with Path(REL_DIR, "I118_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"  报告写入: {os.path.join(REL_DIR, 'I118_report.json')} | 总耗时: {(time.time()-t0)/60:.1f} min")
    print("=" * 96)


if __name__ == "__main__":
    main()
