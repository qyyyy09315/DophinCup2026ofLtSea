"""
G003 — 跨模态特征对抗互译与双向循环一致性扩增 (Cross-Modal Cycle-GAN on ESM2↔ProtBERT)
=====================================================================================
背景与动机:
  - 1664D 特征本质上由两个天然异质的预训练视角拼接而成:
      ESM2-150M (640D, 进化保守/氨基酸接触先验) 与 ProtBERT (1024D, 语法语义/双向掩码语言先验)。
  - 单个模型在高难孤儿序列或低基数标签上往往存在某一视角的特征退化或模态盲区。
  - G003 引入配对循环对抗映射 (Cycle-Consistent Adversarial Translation):
      训练 G_E2P: R^640 -> R^1024 与 G_P2E: R^1024 -> R^640，
      约束双向对抗判别损失 + 双向循环一致性损失 L_cyc + 成对配对重构损失 L_recon。
  - 扩增机制:
      对低基数与孤儿样本，通过跨模态对抗生成其互译伪视角特征对 [x_E, G_E2P(x_E)] 与 [G_P2E(x_P), x_P]，
      平滑两模态在特征交界面的互补性，构建抗退化的鲁棒多模态分类头。

用法:
  python code/G/G003-crossmodal-cyclegan.py            # 正式运行 (~12 min)
  python code/G/G003-crossmodal-cyclegan.py --smoke    # 沙箱冒烟 (~20s)
输出:
  models/g003_cyclegan_*.pt, models/g003_plm_head.pt
  rel/G003_report.json
  rel/g003_cyclegan_val_probs.npz
"""

import os
import sys
import json
import time
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
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

N_LABELS = 500
DIM_E = 640
DIM_P = 1024
DIM_FULL = DIM_E + DIM_P  # 1664
SEED = 42

i064 = SourceFileLoader(
    "i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")
).load_module()
MLP, ASL = i064.MLP, i064.ASL
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe


# ==============================================================================
# 网络结构定义
# ==============================================================================
class TranslationGenerator(nn.Module):
    """跨视角映射生成器：in_dim -> out_dim"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 800),
            nn.LayerNorm(800),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(800, 800),
            nn.LayerNorm(800),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(800, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class ViewDiscriminator(nn.Module):
    """视角判别器：dim -> 1"""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


def compute_gp(D, real_x, fake_x, device):
    b = real_x.size(0)
    eps = torch.rand(b, 1, device=device).expand_as(real_x)
    inter = (eps * real_x + (1.0 - eps) * fake_x).requires_grad_(True)
    d_out = D(inter)
    grads = torch.autograd.grad(
        outputs=d_out, inputs=inter, grad_outputs=torch.ones_like(d_out),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0].view(b, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


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
    parser.add_argument("--epochs-cycle", type=int, default=15, help="Cycle-GAN 训练轮数 (默认 15)")
    parser.add_argument("--epochs-mlp", type=int, default=35, help="下游分类头训练轮数 (默认 35)")
    parser.add_argument("--n-synth", type=int, default=3000, help="合成样本量 (默认 3000)")
    args = parser.parse_args()

    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"  G003: Cross-Modal Cycle-GAN (ESM2 <-> ProtBERT) on 1664D ({'SMOKE' if args.smoke else '正式'})")
    print(f"  Device: {device} | Cycle Epochs: {args.epochs_cycle if not args.smoke else 2} | MLP Epochs: {args.epochs_mlp if not args.smoke else 2}")
    print("=" * 80)

    # 1. 加载双模态特征
    print("[1/5] 加载 ESM2(640D) 与 ProtBERT(1024D) 双视角特征...")
    df_e = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    df_p = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
    ec = [c for c in df_e.columns if c.startswith("emb_")]
    pc = [c for c in df_p.columns if c.startswith("emb_")]
    X_E_all = df_e[ec].values.astype(np.float32)
    X_P_all = df_p[pc].values.astype(np.float32)
    X_full = np.concatenate([X_E_all, X_P_all], axis=1)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.float32)

    idx_all = np.arange(len(train_df))
    idx_tr, idx_va = train_test_split(idx_all, test_size=0.2, random_state=SEED)
    XE_tr, XP_tr, y_tr = X_E_all[idx_tr], X_P_all[idx_tr], y_full[idx_tr]
    XE_va, XP_va, y_va = X_E_all[idx_va], X_P_all[idx_va], y_full[idx_va]
    X_va = X_full[idx_va]

    if args.smoke:
        XE_tr = XE_tr[:2048]
        XP_tr = XP_tr[:2048]
        y_tr = y_tr[:2048]
        X_va = X_va[:512]
        y_va = y_va[:512]
        cycle_epochs = 2
        mlp_epochs = 2
        n_synth = 200
    else:
        cycle_epochs = args.epochs_cycle
        mlp_epochs = args.epochs_mlp
        n_synth = args.n_synth

    # 2. 训练双向 Cycle-GAN
    print(f"\n[2/5] 训练 ESM2 <-> ProtBERT 双向循环对抗映射 ({cycle_epochs} epochs)...")
    loader_cycle = DataLoader(
        TensorDataset(torch.from_numpy(XE_tr), torch.from_numpy(XP_tr)),
        batch_size=512, shuffle=True, pin_memory=True
    )

    G_E2P = TranslationGenerator(DIM_E, DIM_P).to(device)
    G_P2E = TranslationGenerator(DIM_P, DIM_E).to(device)
    D_P = ViewDiscriminator(DIM_P).to(device)
    D_E = ViewDiscriminator(DIM_E).to(device)

    opt_G = torch.optim.Adam(list(G_E2P.parameters()) + list(G_P2E.parameters()), lr=2e-4, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(list(D_P.parameters()) + list(D_E.parameters()), lr=2e-4, betas=(0.5, 0.9))
    l1_crit = nn.L1Loss()

    for ep in range(1, cycle_epochs + 1):
        t_ep = time.time()
        d_loss_sum, g_loss_sum, n_batches = 0.0, 0.0, 0
        for xe, xp in loader_cycle:
            xe = xe.to(device, non_blocking=True)
            xp = xp.to(device, non_blocking=True)

            # 判别器更新
            opt_D.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_xp = G_E2P(xe)
                fake_xe = G_P2E(xp)

            dp_real = D_P(xp).mean()
            dp_fake = D_P(fake_xp).mean()
            gp_p = compute_gp(D_P, xp, fake_xp, device)
            loss_dp = dp_fake - dp_real + 10.0 * gp_p

            de_real = D_E(xe).mean()
            de_fake = D_E(fake_xe).mean()
            gp_e = compute_gp(D_E, xe, fake_xe, device)
            loss_de = de_fake - de_real + 10.0 * gp_e

            loss_d = loss_dp + loss_de
            loss_d.backward()
            opt_D.step()

            # 生成器更新
            opt_G.zero_grad(set_to_none=True)
            gen_xp = G_E2P(xe)
            gen_xe = G_P2E(xp)

            # 对抗损失
            adv_p = -D_P(gen_xp).mean()
            adv_e = -D_E(gen_xe).mean()

            # 双向循环一致性与配对重构损失
            rec_xe = G_P2E(gen_xp)
            rec_xp = G_E2P(gen_xe)
            cyc_loss = l1_crit(rec_xe, xe) + l1_crit(rec_xp, xp)
            recon_loss = l1_crit(gen_xp, xp) + l1_crit(gen_xe, xe)

            loss_g = adv_p + adv_e + 10.0 * cyc_loss + 5.0 * recon_loss
            loss_g.backward()
            opt_G.step()

            d_loss_sum += loss_d.item()
            g_loss_sum += loss_g.item()
            n_batches += 1

        print(f"  cycle ep{ep:02d}/{cycle_epochs:02d} | d_loss={d_loss_sum/n_batches:.4f} | "
              f"g_loss={g_loss_sum/n_batches:.4f} | cyc_recon={recon_loss.item():.4f} ({time.time()-t_ep:.0f}s)", flush=True)

    # 3. 跨模态互译扩增特征构建
    print(f"\n[3/5] 构建跨视角互译强化特征 (为低基数与孤儿样本合成补充特征)...")
    cardinality = y_tr.sum(axis=1)
    target_indices = np.where(cardinality <= 10)[0]
    if len(target_indices) < n_synth:
        target_indices = np.arange(len(y_tr))

    np.random.seed(SEED)
    chosen_idx = np.random.choice(target_indices, size=n_synth, replace=True)
    sub_xe = torch.from_numpy(XE_tr[chosen_idx]).to(device)
    sub_xp = torch.from_numpy(XP_tr[chosen_idx]).to(device)
    sub_y = y_tr[chosen_idx]

    G_E2P.eval()
    G_P2E.eval()
    with torch.no_grad():
        fake_xp_gen = G_E2P(sub_xe).cpu().numpy()
        fake_xe_gen = G_P2E(sub_xp).cpu().numpy()

    # 合成 1: [真实 ESM2, 生成 ProtBERT]
    synth_pair1 = np.concatenate([XE_tr[chosen_idx], fake_xp_gen], axis=1)
    # 合成 2: [生成 ESM2, 真实 ProtBERT]
    synth_pair2 = np.concatenate([fake_xe_gen, XP_tr[chosen_idx]], axis=1)

    X_train_clean = np.concatenate([XE_tr, XP_tr], axis=1)
    # 注入合成特征
    X_train_final = np.vstack([X_train_clean, synth_pair1[:n_synth//2], synth_pair2[:n_synth//2]])
    y_train_final = np.vstack([y_tr, sub_y[:n_synth//2], sub_y[:n_synth//2]])
    print(f"  训练集扩展: {len(X_train_clean)} -> {len(X_train_final)} (+{n_synth})")

    # 4. 重训 PLM-MLP 分类头与评估 Gate 1
    print(f"\n[4/5] 在跨视角增强特征上重训 PLM-MLP 头 ({mlp_epochs} epochs)...")
    loader_mlp_tr = DataLoader(
        TensorDataset(torch.from_numpy(X_train_final), torch.from_numpy(y_train_final)),
        batch_size=512, shuffle=True, pin_memory=True
    )
    loader_mlp_va = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=512, shuffle=False, pin_memory=True
    )

    mlp_head = MLP(DIM_FULL).to(device)
    crit_asl = ASL()
    opt_mlp = torch.optim.AdamW(mlp_head.parameters(), lr=1e-3, weight_decay=1e-4)
    sch_mlp = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mlp, T_max=mlp_epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_f1 = 0.0
    best_state = None
    for ep in range(1, mlp_epochs + 1):
        t_ep = time.time()
        mlp_head.train()
        loss_sum, n_seen = 0.0, 0
        for bx, by in loader_mlp_tr:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            opt_mlp.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = mlp_head(bx)
                loss = crit_asl(logits, by)
            scaler.scale(loss).backward()
            scaler.step(opt_mlp)
            scaler.update()
            loss_sum += loss.item() * bx.size(0)
            n_seen += bx.size(0)
        sch_mlp.step()

        # 验证集评估
        mlp_head.eval()
        preds = []
        with torch.no_grad():
            for bx, _ in loader_mlp_va:
                with torch.amp.autocast("cuda"):
                    pr = torch.sigmoid(mlp_head(bx.to(device, non_blocking=True)).float()).cpu().numpy()
                preds.append(pr)
        val_probs = np.vstack(preds)
        val_f1_05 = macro_f1(val_probs, y_va.astype(np.int8), 0.5)
        ths_star = search_scut(val_probs, y_va.astype(np.int8))
        val_f1_scut = macro_f1(val_probs, y_va.astype(np.int8), ths_star)

        if val_f1_scut > best_val_f1:
            best_val_f1 = val_f1_scut
            best_state = {k: v.cpu().clone() for k, v in mlp_head.state_dict().items()}

        print(f"  mlp ep{ep:02d}/{mlp_epochs:02d} | loss={loss_sum/n_seen:.4f} | "
              f"val_f1(0.5)={val_f1_05:.4f} | val_f1(SCut)={val_f1_scut:.4f} | best={best_val_f1:.4f} "
              f"({time.time()-t_ep:.0f}s)", flush=True)

    mlp_head.load_state_dict(best_state)
    mlp_head.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in loader_mlp_va:
            with torch.amp.autocast("cuda"):
                pr = torch.sigmoid(mlp_head(bx.to(device, non_blocking=True)).float()).cpu().numpy()
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

    gate1_pass = standalone_f1 > 0.2629
    print(f"\n  [Gate 1 独立门槛] G003 Cycle-GAN 独立 F1={standalone_f1:.4f}, AUC={standalone_auc:.4f}, "
          f"R={standalone_rec:.4f}, P={standalone_prec:.4f} (基线单种子 0.2629 / 三种子 0.2729) -> {'通过' if gate1_pass else '未通过'}")

    # 5. 融合流水线评估 Gate 2
    print("\n[5/5] 运行 canonical 融合流水线评估 Gate 2 (RRF + Sinkhorn + 半折 OOF)...")
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

    conclusion = "有效，进入提交与模型替换" if (gate1_pass and gate2_pass) else "未达到融合换锁线，关闭此尝试"
    report = {
        "experiment": "G003",
        "method": "Cross-Modal Cycle-GAN (ESM2 <-> ProtBERT) Augmentation",
        "smoke": args.smoke,
        "n_synth": n_synth,
        "standalone_plm": {
            "f1_scut": round(standalone_f1, 4),
            "precision": round(standalone_prec, 4),
            "recall": round(standalone_rec, 4),
            "auc": round(standalone_auc, 4),
            "gate1_pass": gate1_pass,
            "baseline_single_seed": 0.2629,
            "baseline_multi_seed": 0.2729
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

    report_file = os.path.join(REL_DIR, "G003_report.json")
    with Path(report_file).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    torch.save(G_E2P.state_dict(), os.path.join(MODEL_DIR, "g003_cyclegan_e2p.pt"))
    torch.save(G_P2E.state_dict(), os.path.join(MODEL_DIR, "g003_cyclegan_p2e.pt"))
    torch.save(mlp_head.state_dict(), os.path.join(MODEL_DIR, "g003_plm_head.pt"))
    np.savez_compressed(os.path.join(REL_DIR, "g003_cyclegan_val_probs.npz"), val=final_val_probs)

    print("\n" + "=" * 80)
    print(f"  G003 完成! 独立 F1={standalone_f1:.4f} | 融合 OOF={f1_oof:.4f} (Δ={delta_oof:+.4f}) -> {conclusion}")
    print(f"  报告写入: {report_file} | 总用时: {(time.time()-t_start)/60:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
