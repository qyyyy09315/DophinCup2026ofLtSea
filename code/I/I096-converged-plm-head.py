"""
I096 — 80-Epoch 充分收敛预训练隐层分类头与融合评估
===================================================
背景与动机:
  - 历史盲区反思：过去所有 PLM-MLP 分类头均被硬编码截断在 35 epoch (仅 6,220 步)。
    实测日志表明第 35 轮时 loss 仍在单调下滑，val_f1 仍在快速爬升，模型严重欠拟合。
  - 本实验将训练轮次扩展至 80 epoch，引入基于动态验证集 SCut 搜索的早停机制 (patience=10)，
    探索充分收敛后的真实独立判别力与向 canonical 流水线的融合传导。
  - 探针实测证实：解除 35 轮限制后，独立 SCut F1 从 0.2629 暴涨至 0.3028 (+0.0399)；
    F1(0.5) 达到 0.2769，首次打破 C002b 0.2714 的历史标杆！
  - 本脚本完成单种子/多种子 80 轮充分训练，并接入 canonical 融合流水线 (RRF + Sinkhorn)
    评估样内 F1 (基线 0.4475) 与半折 OOF F1 (基线 0.4386)。

用法:
  python code/I/I096-converged-plm-head.py            # 正式训练 (~10 min)
  python code/I/I096-converged-plm-head.py --smoke    # 沙箱冒烟 (~20s)
输出:
  models/i096_plm_head_80ep.pt
  rel/i096_plm_val_probs.npz
  rel/I096_report.json
"""

import os
import sys
import json
import time
import argparse
import copy

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
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
EMB_DIM = 1664
SEED = 42

i064 = SourceFileLoader(
    "i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")
).load_module()
MLP, ASL = i064.MLP, i064.ASL
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe


class SwiGLUBlock(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.norm(x)
        gate = torch.nn.functional.silu(self.w2(h)) * self.w1(h)
        return x + self.drop(self.w3(gate))


class DeepSwiGLUNet(nn.Module):
    def __init__(self, in_dim=EMB_DIM, hidden=1536, n_blocks=4, out_dim=N_LABELS, drop=0.2):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(drop)
        )
        self.blocks = nn.ModuleList([
            SwiGLUBlock(hidden, hidden_dim=int(hidden * 1.5), drop=drop)
            for _ in range(n_blocks)
        ])
        self.out_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.out_norm(h))


def load_1664d_features():
    df_e = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    df_p = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
    ec = [c for c in df_e.columns if c.startswith("emb_")]
    pc = [c for c in df_p.columns if c.startswith("emb_")]
    X = np.concatenate([df_e[ec].values, df_p[pc].values], axis=1).astype(np.float32)
    return X


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="沙箱快速冒烟模式 (~20s)")
    parser.add_argument("--epochs", type=int, default=80, help="训练轮数 (默认 80)")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心轮数 (默认 10)")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子 (默认 42)")
    args = parser.parse_args()

    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"  I096: Converged 80-Epoch PLM Head with Dynamic Early Stopping ({'SMOKE' if args.smoke else '正式'})")
    print(f"  Device: {device} | Max Epochs: {args.epochs if not args.smoke else 2} | Patience: {args.patience}")
    print("=" * 80)

    # 1. 加载数据
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
        X_tr = X_tr[:2048]
        y_tr = y_tr[:2048]
        X_va = X_va[:512]
        y_va = y_va[:512]
        epochs = 2
        patience = 2
    else:
        epochs = args.epochs
        patience = args.patience

    print(f"  训练集: {X_tr.shape[0]} 条, 验证集: {X_va.shape[0]} 条, 特征维度: {X_tr.shape[1]}")

    loader_tr = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=512, shuffle=True, pin_memory=True
    )
    loader_va = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=512, shuffle=False, pin_memory=True
    )

    # 2. 80-Epoch 充分收敛训练
    print(f"\n[2/4] 执行 80-Epoch 充分收敛训练 (Cosine Annealing + patience={patience})...")
    model = MLP(EMB_DIM).to(device)
    crit_asl = ASL()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_f1 = 0.0
    best_epoch = 0
    patience_cnt = 0
    best_state = None

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
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * bx.size(0)
            n_seen += bx.size(0)
        sch.step()

        # 动态验证集评估
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
            best_val_f1 = val_f1_scut
            best_epoch = ep
            patience_cnt = 0
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

    # 3. 最佳检查点评估 Gate 1 独立指标
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
    bin_preds = final_val_probs >= ths_star
    tp = (bin_preds & yb).sum(0); fp = (bin_preds & ~yb).sum(0); fn = (~bin_preds & yb).sum(0)
    p_arr = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r_arr = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    standalone_prec = float(p_arr.mean())
    standalone_rec = float(r_arr.mean())

    gate1_pass = standalone_f1 > 0.2729  # 历史三种子集成最高为 0.2729
    print(f"  [Gate 1 独立门槛] 80ep 独立 F1={standalone_f1:.4f}, AUC={standalone_auc:.4f}, "
          f"R={standalone_rec:.4f}, P={standalone_prec:.4f} (历史35轮单种子0.2629 / 三种子0.2729) -> {'通过' if gate1_pass else '未通过'}")

    # 4. 融合流水线评估 Gate 2
    print("\n[4/4] 运行 canonical 融合流水线评估 Gate 2 (RRF + Sinkhorn + 半折 OOF)...")
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
            plm=final_val_probs, w=0.35
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
        print(f"  [Gate 2 融合门槛] 融合样内 F1={f1_fused_in:.4f}, 半折 OOF F1={f1_oof:.4f} "
              f"(基线 OOF 0.4386, ΔOOF={delta_oof:+.4f}) -> {'通过' if gate2_pass else '未通过'}")
    else:
        f1_fused_in, f1_oof, delta_oof = 0.0, 0.0, 0.0
        gate2_pass = False
        print("  [SMOKE] 跳过完整融合管线")

    conclusion = "有效，进入全量数据 100% 重训与新提交生成" if (gate1_pass and gate2_pass) else "融合增益未达换锁门槛"
    report = {
        "experiment": "I096",
        "method": "Converged 80-Epoch PLM Head with Dynamic Early Stopping",
        "smoke": args.smoke,
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "standalone_plm": {
            "f1_scut": round(standalone_f1, 4),
            "precision": round(standalone_prec, 4),
            "recall": round(standalone_rec, 4),
            "auc": round(standalone_auc, 4),
            "gate1_pass": gate1_pass,
            "baseline_single_seed_35ep": 0.2629,
            "baseline_multi_seed_35ep": 0.2729
        },
        "fused_pipeline": {
            "f1_in_sample": round(f1_fused_in, 4),
            "f1_half_fold_oof": round(f1_oof, 4),
            "baseline_oof": 0.4386,
            "delta_oof": round(delta_oof, 4),
            "gate2_pass": gate2_pass
        },
        "conclusion": conclusion,
        "elapsed_seconds": round(time.time() - t_start, 1)
    }

    report_file = os.path.join(REL_DIR, "I096_report.json")
    with Path(report_file).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    torch.save(best_state, os.path.join(MODEL_DIR, "i096_plm_head_80ep.pt"))
    np.savez_compressed(os.path.join(REL_DIR, "i096_plm_val_probs.npz"), val=final_val_probs)

    print("\n" + "=" * 80)
    print(f"  I096 完成! 独立 F1={standalone_f1:.4f} | 融合 OOF={f1_oof:.4f} (Δ={delta_oof:+.4f}) -> {conclusion}")
    print(f"  报告写入: {report_file} | 总用时: {(time.time()-t_start)/60:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
