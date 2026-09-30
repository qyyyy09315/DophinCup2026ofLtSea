"""
I106 — 收敛头三种子均值 × microAUC 筛子（PLM 队列 D 刀）
========================================================
背景（I102 准则的直接应用）:
  - I102 实证: 融合传导 ΔOOF 与 microAUC 秩相关 ρ=+0.90（p=0.037），与 F1 负相关 −0.70。
    现 PLM 支（35ep 三种子）microAUC 0.8783 是全队列最高——换支失败的根因。
  - 78ep 收敛头单种子（I096）F1 高但 microAUC 0.8545 下降（操作点特化）；
    集成提升 AUC 幅度大于 F1（Deep Ensembles 经验），本刀测「收敛 × 三种子」能否
    microAUC 过 0.8783——过了才进融合（省管线计算）。
设计:
  1. 平坦 MLP 头按 I096 收敛协议（80ep+SCut 早停 patience=10）训 seeds 11/22/33（80% 训练集）；
  2. 三种子概率平均 → F1 / macroAUC / microAUC 对照 0.8783 筛子；
  3. 过线则 w=0.35 替换 PLM 支走 canonical 全流程 + I105 式 R=20 配对半折（基线=35ep 三种子支）。
输出: rel/I106_report.json, models/i106_conv_head_s{11,22,33}.pt, rel/i106_conv_multiseed_val.npz
用法: python code/I/I106-conv-multiseed-auc.py   (~35 min)
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
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
EMB_DIM = 1664
SEEDS = [11, 22, 33]
MICROAUC_BAR = 0.8783  # I102: 35ep 三种子现支基准

i096 = SourceFileLoader("i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")).load_module()
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
MLP, ASL = i064.MLP, i064.ASL
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe
load_1664d_features = i096.load_1664d_features
build_sim = i096.build_sim


def train_one_seed(seed, X_tr, y_tr, X_va, y_va, device, epochs=80, patience=10,
                   in_dim=EMB_DIM):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    tl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))),
                    batch_size=512, shuffle=True, pin_memory=True)
    vl = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va.astype(np.float32))),
                    batch_size=512, shuffle=False, pin_memory=True)
    m = MLP(in_dim).to(device)
    crit = ASL()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    sc = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_f1, best_ep, stall, best_state = 0.0, 0, 0, None
    for ep in range(1, epochs + 1):
        m.train()
        for bx, by in tl:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = crit(m(bx), by)
            if sc is not None:
                sc.scale(loss).backward(); sc.step(opt); sc.update()
            else:
                loss.backward(); opt.step()
        sch.step()
        m.eval()
        preds = []
        with torch.no_grad():
            for bx, _ in vl:
                with torch.amp.autocast("cuda"):
                    preds.append(torch.sigmoid(m(bx.to(device, non_blocking=True)).float()).cpu().numpy())
        pr = np.vstack(preds)
        f1 = macro_f1(pr, y_va.astype(np.int8), search_scut(pr, y_va.astype(np.int8)))
        if f1 > best_f1:
            best_f1, best_ep, stall = f1, ep, 0
            best_state = copy.deepcopy(m.state_dict())
            mark = "(*)"
        else:
            stall += 1
            mark = "   "
        if ep % 5 == 0 or stall >= patience or ep == epochs:
            print(f"  seed{seed} ep{ep:02d}/{epochs} | val SCut F1={f1:.4f} | best={best_f1:.4f}@ep{best_ep:02d} {mark}",
                  flush=True)
        if stall >= patience:
            print(f"  seed{seed} [早停] ep{ep}")
            break
    m.load_state_dict(best_state)
    m.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in vl:
            with torch.amp.autocast("cuda"):
                preds.append(torch.sigmoid(m(bx.to(device, non_blocking=True)).float()).cpu().numpy())
    return np.vstack(preds).astype(np.float32), best_ep, best_f1


def half_fold_oof(mixed, y_va, reps=20, seed0=1000):
    """I105 式 R 次半折（不同切分）"""
    vals = []
    yb = y_va.astype(bool)
    for r in range(reps):
        a_idx, b_idx = train_test_split(np.arange(len(y_va)), test_size=0.5, random_state=seed0 + r)
        preds = np.zeros_like(y_va, dtype=bool)
        for fi, ei in ((a_idx, b_idx), (b_idx, a_idx)):
            t_sub = search_scut(mixed[fi], y_va[fi])
            preds[ei] = mixed[ei] >= t_sub
        tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
        p_c = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        r_c = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        vals.append(float(np.mean(np.where(p_c + r_c > 0, 2 * p_c * r_c / np.maximum(p_c + r_c, 1e-9), 0.0))))
    return np.array(vals)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print("  I106: Converged Head x 3-Seed Mean x microAUC Sieve")
    print(f"  筛子: microAUC > {MICROAUC_BAR} (I102: 35ep 三种子现支) | seeds {SEEDS}")
    print("=" * 88)

    print("[1/3] 加载 1664D 与切分...")
    X_full = load_1664d_features()
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    X_tr, y_tr = X_full[idx_tr], y_full[idx_tr]
    X_va, y_va = X_full[idx_va], y_full[idx_va]

    print("[2/3] 三种子收敛头训练（I096 协议 80ep+SCut 早停）...")
    probs_seeds = []
    for s in SEEDS:
        pr, best_ep, best_f1 = train_one_seed(s, X_tr, y_tr, X_va, y_va, device)
        probs_seeds.append(pr)
        np.savez_compressed(os.path.join(REL_DIR, f"i106_conv_head_s{s}.npz"), val=pr)
        print(f"  seed {s}: best F1={best_f1:.4f}@ep{best_ep}", flush=True)
    mean_p = np.mean(probs_seeds, axis=0)
    np.savez_compressed(os.path.join(REL_DIR, "i106_conv_multiseed_val.npz"), val=mean_p)

    print("[3/3] microAUC 筛子与（过线）融合配对检验...")
    ths = search_scut(mean_p, y_va)
    f1 = macro_f1(mean_p, y_va, ths)
    auc_m = float(roc_auc_score(y_va, mean_p, average="macro"))
    auc_u = float(roc_auc_score(y_va, mean_p, average="micro"))
    print(f"  三种子均值: F1={f1:.4f} | macroAUC={auc_m:.4f} | microAUC={auc_u:.4f} "
          f"(单种子 I096: 0.3036/0.7779/0.8545; 35ep 三种子: 0.2729/0.7947/0.8783)")
    sieve_pass = auc_u > MICROAUC_BAR
    print(f"  [筛子] microAUC {'>' if sieve_pass else '<='} {MICROAUC_BAR} -> "
          f"{'进融合' if sieve_pass else '不进融合，D 刀关闭'}")

    fused = None
    if sieve_pass:
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
        v_new = half_fold_oof(mixed_new, y_va)
        v_base = half_fold_oof(mixed_base, y_va)
        d = v_new - v_base
        d_mean, d_se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        ci_lo = d_mean - 1.96 * d_se
        print(f"  配对 ΔOOF(R=20): {d_mean:+.4f} ± {d_se:.4f} | CI95下界 {ci_lo:+.4f} | "
              f"P(Δ>0)={float((d > 0).mean()):.2f}")
        fused = {"delta_mean": round(d_mean, 4), "delta_se": round(d_se, 4),
                 "ci95_lo": round(ci_lo, 4), "p_gt0": round(float((d > 0).mean()), 3),
                 "new_oof_mean": round(float(v_new.mean()), 4),
                 "base_oof_mean": round(float(v_base.mean()), 4)}
        if ci_lo > 0 or d_mean >= 0.001:
            fused["verdict"] = "过线，换 PLM 支进 I100 全量出提交流程"
        else:
            fused["verdict"] = "筛子过但融合未证实，D 刀关闭"

    conclusion = ("microAUC 过筛，融合见上" if sieve_pass
                  else f"microAUC {auc_u:.4f} 未过 {MICROAUC_BAR}，D 刀关闭（A 刀 650M 嵌入为下一候选）")
    report = {
        "experiment": "I106",
        "method": "Converged head x 3-seed mean x microAUC sieve",
        "seeds": SEEDS,
        "f1_scut": round(f1, 4), "auc_macro": round(auc_m, 4), "auc_micro": round(auc_u, 4),
        "bar": MICROAUC_BAR,
        "sieve_pass": sieve_pass,
        "fused": fused,
        "conclusion": conclusion,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with Path(os.path.join(REL_DIR, "I106_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 88)
    print(f"  I106 完成! -> {conclusion}")
    print(f"  报告: {os.path.join(REL_DIR, 'I106_report.json')} | 总用时 {(time.time()-t0)/60:.1f} min")
    print("=" * 88)


if __name__ == "__main__":
    main()
