"""
I102 — 排序质量假说：融合传导量由分支排序能力而非操作点 F1 决定（分析性）
============================================================================
背景（方法创新核心，I098a 机理的形式化）:
  - 七次复现「独立增益不传导」（I063/I074/I077/I087/G003/I096/I098），I098a 补全机理:
    SwiGLU 头 F1 0.3141 更高但 AUC 0.7390 更低——操作点特化、全局排序差。
  - 假说 H-R: 融合链（RRF+Sinkhorn）消费全局排序信息，因此分支替换的融合传导量
    ΔOOF 应与分支的**排序质量指标**（macro/micro AUC）相关，而与**操作点指标**
    （SCut F1）无关甚至负相关。若成立，给出可操作设计准则:
    「分支入选融合的先验过滤条件应取 AUC 而非独立 F1」——报告方法学贡献。
设计:
  - 5 个干净头（同训练数据 80%、同评估 val、仅头结构/预算不同）:
      35ep 单种子 MLP（现场补训）/ 35ep 三种子平均 / 78ep 收敛 MLP(I096) /
      SwiGLU(I098) / ML-Decoder(I099)
  - 每头计算: SCut F1、macro-AUC、micro-AUC；再以 w=0.35 替换 PLM 支走 canonical
    全流程，得 ΔOOF（相对三种子基线）
  - Spearman 秩相关: ΔOOF vs {F1, macro-AUC, micro-AUC}（n=5，探索性证据）
输出: rel/I102_report.json（分析性，不写腾讯表格）
用法: python code/I/I102-ranking-quality-hypothesis.py
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
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
EMB_DIM = 1664
SEED = 42

i096 = SourceFileLoader("i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")).load_module()
i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
g004 = SourceFileLoader("g004", os.path.join(PROJECT_ROOT, "code", "G", "G004-latent-flow-matching.py")).load_module()
MLP, ASL = i064.MLP, i064.ASL
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe
load_1664d_features = g004.load_1664d_features


def half_fold_oof(mixed, y_va):
    a_idx, b_idx = train_test_split(np.arange(len(y_va)), test_size=0.5, random_state=SEED)
    preds = np.zeros_like(y_va, dtype=bool)
    for fi, ei in ((a_idx, b_idx), (b_idx, a_idx)):
        t_sub = search_scut(mixed[fi], y_va[fi])
        preds[ei] = mixed[ei] >= t_sub
    yb = y_va.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
    p_c = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r_c = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    return float(np.mean(np.where(p_c + r_c > 0, 2 * p_c * r_c / np.maximum(p_c + r_c, 1e-9), 0.0)))


def train_35ep_single(X_tr, y_tr, device):
    """I064 同款 35ep 单种子平坦头（补齐数据点）"""
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))),
                        batch_size=512, shuffle=True, pin_memory=True)
    m = MLP(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35, eta_min=1e-5)
    crit = ASL(); sc = torch.amp.GradScaler("cuda")
    for ep in range(35):
        m.train()
        for bx, by in loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = crit(m(bx), by)
            sc.scale(loss).backward(); sc.step(opt); sc.update()
        sch.step()
    return m


@torch.no_grad()
def head_probs(model, X_va, device):
    model.eval()
    out = []
    for s in range(0, len(X_va), 512):
        bx = torch.from_numpy(X_va[s:s + 512]).to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out.append(torch.sigmoid(model(bx).float()).cpu().numpy())
    return np.vstack(out)


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print("  I102: Ranking-Quality Hypothesis (fusion transfer ~ AUC, not SCut-F1)")
    print("=" * 88)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_all = np.arange(len(train_df))
    idx_tr, idx_va = train_test_split(idx_all, test_size=0.2, random_state=SEED)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)

    print("[1/4] 构建 deep4/sim 与头概率集合...")
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
    sim = i096.build_sim(y_tr, n_va, val_pids, train_map, val_map)

    X_full = load_1664d_features()
    X_tr, X_va = X_full[idx_tr], X_full[idx_va]

    heads = {}
    heads["35ep_single"] = head_probs(train_35ep_single(X_tr, y_tr, device), X_va, device)
    heads["35ep_multiseed"] = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    heads["78ep_mlp(I096)"] = np.load(os.path.join(REL_DIR, "i096_plm_val_probs.npz"))["val"]
    heads["swiglu(I098)"] = np.load(os.path.join(REL_DIR, "i098_swiglu_val_probs.npz"))["val"]
    heads["mldecoder(I099)"] = np.load(os.path.join(REL_DIR, "i099_mldecoder_val_probs.npz"))["val"]

    print("[2/4] 每头: 操作点与排序指标...")
    rows = []
    for name, pr in heads.items():
        ths = search_scut(pr, y_va)
        f1 = macro_f1(pr, y_va, ths)
        auc_m = float(roc_auc_score(y_va, pr, average="macro"))
        auc_u = float(roc_auc_score(y_va, pr, average="micro"))
        rows.append({"head": name, "f1_scut": round(f1, 4),
                     "auc_macro": round(auc_m, 4), "auc_micro": round(auc_u, 4)})
        print(f"  {name:22s} | F1={f1:.4f} | macroAUC={auc_m:.4f} | microAUC={auc_u:.4f}")

    print("[3/4] 每头 w=0.35 替换 PLM 支的融合传导...")
    _, _, mixed_base, _, _ = full_pipe(
        0.65 * deep4 + 0.35 * heads["35ep_multiseed"], sim, y_tr, y_va, n_va)
    oof_base = half_fold_oof(mixed_base, y_va)
    print(f"  基线（35ep_multiseed 支）: OOF={oof_base:.4f}")
    for r in rows:
        pr = heads[r["head"]]
        _, _, mixed, _, _ = full_pipe(0.65 * deep4 + 0.35 * pr, sim, y_tr, y_va, n_va)
        r["oof"] = round(half_fold_oof(mixed, y_va), 4)
        r["delta_oof"] = round(r["oof"] - oof_base, 4)
        print(f"  {r['head']:22s} | OOF={r['oof']:.4f} | ΔOOF={r['delta_oof']:+.4f}")

    print("[4/4] Spearman 秩相关（n=5，探索性）...")
    d = np.array([[r["f1_scut"], r["auc_macro"], r["auc_micro"], r["delta_oof"]] for r in rows])
    cors = {}
    for j, key in enumerate(["f1_scut", "auc_macro", "auc_micro"]):
        rho, p = spearmanr(d[:, j], d[:, 3])
        cors[key] = {"rho": round(float(rho), 4), "p": round(float(p), 4)}
        print(f"  ΔOOF ~ {key:10s}: rho={rho:+.4f} (p={p:.4f})")

    h_rs = (cors["auc_macro"]["rho"] > cors["f1_scut"]["rho"]) and (cors["auc_micro"]["rho"] > cors["f1_scut"]["rho"])
    conclusion = ("H-R 成立（排序质量假说获探索性支持）：融合传导量与 AUC 的秩相关强于 F1；"
                  "设计准则=分支先验过滤应取 AUC 而非独立 F1" if h_rs else
                  "H-R 未获支持（n=5 样本量下秩相关不占优），准则不成立")
    report = {
        "experiment": "I102",
        "method": "Ranking-Quality Hypothesis (analytical)",
        "heads": rows,
        "baseline_oof": round(oof_base, 4),
        "spearman": cors,
        "conclusion": conclusion,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with Path(os.path.join(REL_DIR, "I102_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 88)
    print(f"  I102 完成! -> {conclusion}")
    print(f"  报告: {os.path.join(REL_DIR, 'I102_report.json')} | 总用时: {(time.time()-t0)/60:.1f} min")
    print("=" * 88)


if __name__ == "__main__":
    main()
