"""
I099 — ML-Decoder 分组查询注意力头与融合评估（I 线大容量头第二刀）
===================================================
背景:
  - [242] ML-Decoder (Ridnik et al., WACV 2023): grouped-query 交叉注意力分类头，
    标签嵌入学习标签共现，大规模多标签上全面优于逐类线性头；
    ATGO (Bioinformatics 2022) 证明 ESM 嵌入 + 注意力头在功能预测有效（生物域先例）。
  - 死穴回应: 平坦 MLP / SwiGLU 头均为"逐标签独立"结构，不建模标签条件共现；
    本刀用标签身份嵌入 + 组级聚合 + 标签间 self-attention 直接建模 500 类共现
    （禁用 GO 层级图，共现全部由数据学习）。
架构（8GB 显存落地版）:
  - 特征 token 化: 1664D = 13×128（ESM2 640=5tok + ProtBERT 1024=8tok，拼接序天然分段）+ 位置编码;
  - grouped query: 25 组（每组 20 类）共享 query 对 13 特征 token 交叉注意力 → 组级强度 g;
  - 标签嵌入 E(500×128) + g 广播残差 → 1 层标签间 self-attention（batch 256 控制 500×500 矩阵显存）
    → FFN → per-class 线性头。
协议差异说明: 与 I096/I098 同预算（1664D / ASL / AdamW 1e-3 / wd 1e-4 / cosine T_max=80 /
  80ep + 逐轮 val SCut 早停 patience=10 / seed=42），两处必要差异——batch 512→256（自注意力显存）、
  加 grad clip 1.0（注意力梯度稳定），已在本 docstring 声明。
判据:
  - Gate 1 独立: SCut F1 > 0.3036（I096 新锚点）
  - Gate 2 融合: 半折 OOF >= 0.4396（基线 0.4386 + 0.001）
用法:
  python code/I/I099-mldecoder-converged-head.py            # 正式训练 (~20 min)
  python code/I/I099-mldecoder-converged-head.py --smoke    # 沙箱冒烟 (~30s)
输出:
  models/i099_mldecoder_head_80ep.pt
  rel/i099_mldecoder_val_probs.npz
  rel/I099_report.json
"""

import os
import time
import json
import copy
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
SEED = 42

i096 = SourceFileLoader(
    "i096", os.path.join(PROJECT_ROOT, "code", "I", "I096-converged-plm-head.py")
).load_module()
macro_f1, search_scut, full_pipe = i096.macro_f1, i096.search_scut, i096.full_pipe
load_1664d_features = i096.load_1664d_features
build_sim = i096.build_sim
ASL = i096.ASL


class MLDecoderHead(nn.Module):
    """[242] 落地版：grouped-query 交叉注意力 + 标签身份嵌入 + 共现 self-attention."""

    def __init__(self, in_dim=EMB_DIM, d=128, n_labels=N_LABELS, n_groups=25,
                 n_heads=4, drop=0.2):
        super().__init__()
        assert in_dim % d == 0 and n_labels % n_groups == 0
        self.d, self.n_tokens, self.n_groups = d, in_dim // d, n_groups
        self.per_group = n_labels // n_groups
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d))
        self.label_emb = nn.Parameter(torch.zeros(n_labels, d))
        self.group_query = nn.Parameter(torch.zeros(n_groups, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.label_emb, std=0.02)
        nn.init.trunc_normal_(self.group_query, std=0.02)
        self.register_buffer("label2group", torch.arange(n_labels) // self.per_group)
        self.cross = nn.MultiheadAttention(d, n_heads, dropout=drop, batch_first=True)
        self.cross_norm = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=drop, batch_first=True)
        self.self_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(drop), nn.Linear(d * 4, d)
        )
        self.ffn_norm = nn.LayerNorm(d)
        self.per_class = nn.Linear(d, 1)

    def forward(self, x):
        b = x.size(0)
        tok = x.view(b, self.n_tokens, self.d) + self.pos
        q = self.group_query.unsqueeze(0).expand(b, -1, -1)
        g, _ = self.cross(q, tok, tok, need_weights=False)
        g = self.cross_norm(q + g)
        h = self.label_emb.unsqueeze(0).expand(b, -1, -1) + g[:, self.label2group]
        sa, _ = self.self_attn(h, h, h, need_weights=False)
        h = self.self_norm(h + sa)
        h = self.ffn_norm(h + self.ffn(h))
        return self.per_class(h).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="沙箱快速冒烟模式 (~30s)")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数 (默认 80)")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心轮数 (默认 10)")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子 (默认 42)")
    parser.add_argument("--batch", type=int, default=256, help="批大小 (默认 256, 自注意力显存)")
    args = parser.parse_args()

    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 84)
    print(f"  I099: ML-Decoder Grouped-Query Converged Head ({'SMOKE' if args.smoke else '正式'})")
    print(f"  Device: {device} | Max Epochs: {args.epochs if not args.smoke else 2} | "
          f"Patience: {args.patience} | Batch: {args.batch}")
    print("=" * 84)

    print("[1/4] 加载 1664D 特征矩阵与标签...")
    X_full = load_1664d_features()
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.float32)

    idx_all = np.arange(len(train_df))
    idx_tr, idx_va = train_test_split(idx_all, test_size=0.2, random_state=SEED)
    X_tr, y_tr = X_full[idx_tr], y_full[idx_tr]
    X_va, y_va = X_full[idx_va], y_full[idx_va]

    if args.smoke:
        X_tr, y_tr = X_tr[:2048], y_tr[:2048]
        X_va, y_va = X_va[:512], y_va[:512]
        epochs, patience = 2, 2
    else:
        epochs, patience = args.epochs, args.patience

    print(f"  训练集: {X_tr.shape[0]} 条, 验证集: {X_va.shape[0]} 条, 特征维度: {X_tr.shape[1]}")

    loader_tr = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch, shuffle=True, pin_memory=True,
    )
    loader_va = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=args.batch, shuffle=False, pin_memory=True,
    )

    print(f"\n[2/4] 训练 ML-Decoder 头 (d=128, 25 组, batch={args.batch})...")
    model = MLDecoderHead(in_dim=EMB_DIM, d=128, n_labels=N_LABELS,
                          n_groups=25, n_heads=4, drop=0.2).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  参数量: {n_params:.2f}M (结构容量: 标签身份 + 组聚合 + 共现 self-attention)")
    crit_asl = ASL()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_f1, best_epoch, patience_cnt, best_state = 0.0, 0, 0, None
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        loss_sum, n_seen = 0.0, 0
        for bx, by in loader_tr:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(bx)
                loss = crit_asl(logits, by)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            loss_sum += loss.item() * bx.size(0)
            n_seen += bx.size(0)
        sch.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for bx, _ in loader_va:
                with torch.amp.autocast("cuda"):
                    pr = torch.sigmoid(model(bx.to(device, non_blocking=True)).float()).cpu().numpy()
                preds.append(pr)
        val_probs = np.vstack(preds)
        val_f1_05 = macro_f1(val_probs, y_va.astype(np.int8), 0.5)
        ths_star = search_scut(val_probs, y_va.astype(np.int8))
        val_f1_scut = macro_f1(val_probs, y_va.astype(np.int8), ths_star)

        if val_f1_scut > best_val_f1:
            best_val_f1, best_epoch, patience_cnt = val_f1_scut, ep, 0
            best_state = copy.deepcopy(model.state_dict())
            mark = "(*)"
        else:
            patience_cnt += 1
            mark = "   "
        print(f"  ep{ep:02d}/{epochs:02d} | loss={loss_sum/n_seen:.4f} | val_f1(0.5)={val_f1_05:.4f} | "
              f"val_f1(SCut)={val_f1_scut:.4f} | best={best_val_f1:.4f}@ep{best_epoch:02d} {mark} "
              f"({time.time()-t_ep:.0f}s)", flush=True)
        if patience_cnt >= patience:
            print(f"  [早停] 连续 {patience} 轮未提升，早停于 epoch {ep}")
            break

    print(f"\n[3/4] 加载最优权重 (ep{best_epoch:02d}) 评估 Gate 1 独立指标...")
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in loader_va:
            with torch.amp.autocast("cuda"):
                pr = torch.sigmoid(model(bx.to(device, non_blocking=True)).float()).cpu().numpy()
            preds.append(pr)
    final_val_probs = np.vstack(preds)
    ths_star = search_scut(final_val_probs, y_va.astype(np.int8))
    standalone_f1 = macro_f1(final_val_probs, y_va.astype(np.int8), ths_star)
    standalone_auc = float(roc_auc_score(y_va.astype(np.int8), final_val_probs, average="macro"))
    yb = y_va.astype(bool)
    bp = final_val_probs >= ths_star
    tp = (bp & yb).sum(0); fp = (bp & ~yb).sum(0); fn = (~bp & yb).sum(0)
    p_arr = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r_arr = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    standalone_prec, standalone_rec = float(p_arr.mean()), float(r_arr.mean())

    gate1_pass = standalone_f1 > 0.3036
    print(f"  [Gate 1 独立门槛] 独立 F1={standalone_f1:.4f}, AUC={standalone_auc:.4f}, "
          f"R={standalone_rec:.4f}, P={standalone_prec:.4f} (I096 平坦收敛头锚点 0.3036) "
          f"-> {'通过' if gate1_pass else '未通过'}")

    print("\n[4/4] 运行 canonical 融合流水线评估 Gate 2...")
    if not args.smoke:
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
        sim = build_sim(y_tr.astype(np.int8), len(idx_va), val_pids, train_map, val_map)

        f1_fused_in, auc_fused_in, mixed_scores, _, _ = full_pipe(
            deep4, sim, y_tr.astype(np.int8), y_va.astype(np.int8), len(idx_va),
            plm=final_val_probs, w=0.35,
        )
        a_idx, b_idx = train_test_split(np.arange(len(idx_va)), test_size=0.5, random_state=SEED)
        preds_oof = np.zeros_like(y_va, dtype=bool)
        for fi, ei in ((a_idx, b_idx), (b_idx, a_idx)):
            t_sub = search_scut(mixed_scores[fi], y_va.astype(np.int8)[fi])
            preds_oof[ei] = mixed_scores[ei] >= t_sub
        yb = y_va.astype(bool)
        tp = (preds_oof & yb).sum(0); fp = (preds_oof & ~yb).sum(0); fn = (~preds_oof & yb).sum(0)
        p_c = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        r_c = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        f1_oof = float(np.mean(np.where(p_c + r_c > 0, 2 * p_c * r_c / np.maximum(p_c + r_c, 1e-9), 0.0)))

        delta_oof = f1_oof - 0.4386
        gate2_pass = delta_oof >= 0.001
        print(f"  [Gate 2 融合门槛] 融合样内 F1={f1_fused_in:.4f}, AUC={auc_fused_in:.4f}, "
              f"半折 OOF F1={f1_oof:.4f} (基线 0.4386, ΔOOF={delta_oof:+.4f}) "
              f"-> {'通过' if gate2_pass else '未通过'}")
    else:
        f1_fused_in, auc_fused_in, f1_oof, delta_oof = 0.0, 0.0, 0.0, 0.0
        gate2_pass = False
        print("  [SMOKE] 跳过完整融合管线")

    if gate1_pass and gate2_pass:
        conclusion = "有效，进入 I100 多种子全量重训换锁流程"
    elif gate1_pass and not gate2_pass:
        conclusion = "独立过线融合不传导（法则第 7 次），转 I101 新支路方案"
    else:
        conclusion = "未过独立门槛，关闭"
    report = {
        "experiment": "I099",
        "method": "ML-Decoder Grouped-Query Converged Head (d=128, 25 groups)",
        "smoke": args.smoke,
        "n_params_m": round(n_params, 2),
        "protocol_diff": "batch 512->256 (self-attn VRAM), grad clip 1.0 (attention stability)",
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "standalone_plm": {
            "f1_scut": round(standalone_f1, 4),
            "precision": round(standalone_prec, 4),
            "recall": round(standalone_rec, 4),
            "auc": round(standalone_auc, 4),
            "gate1_pass": gate1_pass,
            "anchor_i096_mlp": 0.3036,
            "baseline_35ep_mlp": 0.2629,
        },
        "fused_pipeline": {
            "f1_in_sample": round(f1_fused_in, 4),
            "auc_in_sample": round(auc_fused_in, 4),
            "f1_half_fold_oof": round(f1_oof, 4),
            "baseline_oof": 0.4386,
            "delta_oof": round(delta_oof, 4),
            "gate2_pass": gate2_pass,
        },
        "conclusion": conclusion,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with Path(os.path.join(REL_DIR, "I099_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if not args.smoke:
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "i099_mldecoder_head_80ep.pt"))
        np.savez_compressed(os.path.join(REL_DIR, "i099_mldecoder_val_probs.npz"),
                            val=final_val_probs.astype(np.float32))

    print("\n" + "=" * 84)
    print(f"  I099 完成! 独立 F1={standalone_f1:.4f} | 融合 OOF={f1_oof:.4f} (Δ={delta_oof:+.4f}) -> {conclusion}")
    print(f"  报告: {os.path.join(REL_DIR, 'I099_report.json')} | 总用时: {(time.time()-t_start)/60:.1f} min")
    print("=" * 84)


if __name__ == "__main__":
    main()
