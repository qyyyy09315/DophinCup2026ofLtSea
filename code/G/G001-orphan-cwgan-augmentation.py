"""
G001 — 孤儿/低基数切片条件流形生成扩增 (Orphan-Targeted cWGAN-GP on 1664D)
=======================================================================
背景与设计:
  - 依据第 158 轮文献调研与 UCL David Jones 团队 FFPred-GAN (Nature Machine Intelligence 2020):
    针对蛋白质功能多标签长尾与孤儿切片（Miss 子集 F1 仅 0.0767；card<=10 仅 0.2017），
    在预训练连续隐空间 (1664D = ESM2 640D + ProtBERT 1024D) 构建条件 WGAN-GP。
  - 生成器 G(z, y) 学习输入多标签条件 y 与噪声 z 合成拟真连续嵌入；
  - 判别器 D(x, y) 施加 Wasserstein 距离与梯度惩罚 (lambda_gp=10.0)；
  - 关键护栏：Classifier Two-Sample Test (C2ST) 双样本检验，过滤离群虚构样本；
  - 下游在增强数据集上重训 PLM-MLP 分类头，做两级门槛评估：
    门槛 1 (独立)：单种子独立 Macro F1 > 0.2629 (基线单种子) / 0.2729 (三种子)；
    门槛 2 (融合)：canonical 融合流水线半折 OOF Macro F1 较基线 0.4386 >= +0.001。

用法:
  python code/G/G001-orphan-cwgan-augmentation.py            # 正式训练 (~15 min)
  python code/G/G001-orphan-cwgan-augmentation.py --smoke    # 沙箱冒烟 (~30s)
输出:
  models/g001_cwgan_gen.pt, models/g001_cwgan_disc.pt, models/g001_plm_head.pt
  rel/G001_report.json
  rel/g001_augmented_val_probs.npz
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
from sklearn.linear_model import LogisticRegression
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
NOISE_DIM = 128
COND_DIM = 128
LAMBDA_GP = 10.0
N_CRITIC = 5
SEED = 42

i064 = SourceFileLoader(
    "i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")
).load_module()
MLP, ASL = i064.MLP, i064.ASL
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe


# ==============================================================================
# 网络结构定义
# ==============================================================================
class ConditionalGenerator(nn.Module):
    """条件生成器：G([z, proj(y)]) -> x_fake in R^1664"""

    def __init__(self, emb_dim=EMB_DIM, noise_dim=NOISE_DIM, cond_dim=COND_DIM, n_labels=N_LABELS):
        super().__init__()
        self.label_proj = nn.Sequential(
            nn.Linear(n_labels, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        in_dim = noise_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, emb_dim)
        )

    def forward(self, z, y):
        y_c = self.label_proj(y)
        h = torch.cat([z, y_c], dim=1)
        return self.net(h)


class ConditionalDiscriminator(nn.Module):
    """条件判别器：D([x, proj(y)]) -> R^1 (WGAN-GP 评分)"""

    def __init__(self, emb_dim=EMB_DIM, cond_dim=COND_DIM, n_labels=N_LABELS):
        super().__init__()
        self.label_proj = nn.Sequential(
            nn.Linear(n_labels, cond_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        in_dim = emb_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, x, y):
        y_c = self.label_proj(y)
        h = torch.cat([x, y_c], dim=1)
        return self.net(h)


def compute_gradient_penalty(D, real_x, fake_x, y, device):
    """WGAN-GP 梯度惩罚项"""
    b = real_x.size(0)
    eps = torch.rand(b, 1, device=device).expand_as(real_x)
    interpolated = (eps * real_x + (1.0 - eps) * fake_x).requires_grad_(True)
    d_inter = D(interpolated, y)
    grad_outputs = torch.ones_like(d_inter)
    grads = torch.autograd.grad(
        outputs=d_inter,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grads = grads.view(b, -1)
    norm = grads.norm(2, dim=1)
    return ((norm - 1.0) ** 2).mean()


# ==============================================================================
# 数据加载与相似性构建
# ==============================================================================
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


# ==============================================================================
# 主执行流程
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="沙箱快速冒烟模式 (~30s)")
    parser.add_argument("--epochs-wgan", type=int, default=20, help="cWGAN-GP 训练轮数")
    parser.add_argument("--epochs-mlp", type=int, default=35, help="下游分类头训练轮数")
    parser.add_argument("--n-synth", type=int, default=3000, help="合成候选样本量")
    args = parser.parse_args()

    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"  G001: Orphan-Targeted cWGAN-GP on 1664D Continuous Embedding ({'SMOKE' if args.smoke else '正式'})")
    print(f"  Device: {device} | WGAN Epochs: {args.epochs_wgan if not args.smoke else 2} | MLP Epochs: {args.epochs_mlp if not args.smoke else 2}")
    print("=" * 80)

    # 1. 数据准备
    print("[1/6] 加载 1664D 特征矩阵与标签...")
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
        wgan_epochs = 2
        mlp_epochs = 2
        n_synth = 200
    else:
        wgan_epochs = args.epochs_wgan
        mlp_epochs = args.epochs_mlp
        n_synth = args.n_synth

    print(f"  训练集: {X_tr.shape[0]} 条, 验证集: {X_va.shape[0]} 条, 特征维度: {X_tr.shape[1]}")

    # 2. 训练条件 WGAN-GP
    print(f"\n[2/6] 训练条件 WGAN-GP ({wgan_epochs} epochs, batch=512)...")
    loader_wgan = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=512, shuffle=True, pin_memory=True
    )

    G = ConditionalGenerator().to(device)
    D = ConditionalDiscriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))

    for ep in range(1, wgan_epochs + 1):
        t_ep = time.time()
        d_loss_sum, g_loss_sum = 0.0, 0.0
        n_batches = 0
        for b_idx, (real_x, real_y) in enumerate(loader_wgan):
            real_x = real_x.to(device, non_blocking=True)
            real_y = real_y.to(device, non_blocking=True)
            b = real_x.size(0)

            # 判别器更新
            opt_D.zero_grad(set_to_none=True)
            z = torch.randn(b, NOISE_DIM, device=device)
            with torch.no_grad():
                fake_x = G(z, real_y)

            d_real = D(real_x, real_y).mean()
            d_fake = D(fake_x, real_y).mean()
            gp = compute_gradient_penalty(D, real_x, fake_x, real_y, device)
            d_loss = d_fake - d_real + LAMBDA_GP * gp
            d_loss.backward()
            opt_D.step()
            d_loss_sum += d_loss.item()

            # 生成器更新 (每 N_CRITIC 步更新一次)
            if (b_idx + 1) % N_CRITIC == 0 or (b_idx + 1) == len(loader_wgan):
                opt_G.zero_grad(set_to_none=True)
                z = torch.randn(b, NOISE_DIM, device=device)
                gen_x = G(z, real_y)
                g_loss = -D(gen_x, real_y).mean()
                g_loss.backward()
                opt_G.step()
                g_loss_sum += g_loss.item()

            n_batches += 1

        print(f"  wgan ep{ep:02d}/{wgan_epochs:02d} | d_loss={d_loss_sum/n_batches:.4f} | "
              f"g_loss={g_loss_sum/max(1, n_batches//N_CRITIC):.4f} ({time.time()-t_ep:.0f}s)", flush=True)

    # 3. 孤儿与低基数切片针对性合成
    print(f"\n[3/6] 针对低基数与孤儿样本标签切片合成 {n_synth} 条潜变量...")
    cardinality = y_tr.sum(axis=1)
    target_indices = np.where(cardinality <= 10)[0]
    if len(target_indices) < n_synth:
        target_indices = np.arange(len(y_tr))

    np.random.seed(SEED)
    chosen_idx = np.random.choice(target_indices, size=n_synth, replace=True)
    y_target = y_tr[chosen_idx]

    G.eval()
    with torch.no_grad():
        z = torch.randn(n_synth, NOISE_DIM, device=device)
        fake_X = G(z, torch.from_numpy(y_target).to(device)).cpu().numpy()

    # 4. Classifier Two-Sample Test (C2ST) 保真度过滤门控
    print("[4/6] 执行 C2ST (Classifier Two-Sample Test) 保真度检验门控...")
    n_test_eval = min(n_synth, len(X_tr))
    real_sample_idx = np.random.choice(len(X_tr), size=n_test_eval, replace=False)
    X_c2st = np.vstack([X_tr[real_sample_idx], fake_X])
    y_c2st = np.hstack([np.ones(n_test_eval), np.zeros(n_synth)])

    clf = LogisticRegression(max_iter=200, C=0.1, random_state=SEED)
    clf.fit(X_c2st, y_c2st)
    prob_real = clf.predict_proba(fake_X)[:, 1]

    # 保留预测属于合理分布区间的合成样本 (0.35 <= P(real) <= 0.65 判定为与真实流形高度契合)
    valid_mask = (prob_real >= 0.35) & (prob_real <= 0.65)
    n_kept = int(valid_mask.sum())
    pass_rate = n_kept / n_synth
    print(f"  C2ST 整体辨别器平均分: {prob_real.mean():.4f}, 检验放行率: {pass_rate:.1%} ({n_kept}/{n_synth})")

    if n_kept >= 100:
        X_aug_extra = fake_X[valid_mask]
        y_aug_extra = y_target[valid_mask]
        X_train_final = np.vstack([X_tr, X_aug_extra])
        y_train_final = np.vstack([y_tr, y_aug_extra])
    else:
        print("  [警告] C2ST 放行样本不足，使用全部合成样本前 50% 均值保底")
        top_half = np.argsort(np.abs(prob_real - 0.5))[:max(n_synth//2, 100)]
        X_train_final = np.vstack([X_tr, fake_X[top_half]])
        y_train_final = np.vstack([y_tr, y_target[top_half]])
        n_kept = len(top_half)

    print(f"  最终扩增训练集规模: {len(X_tr)} -> {len(X_train_final)} (+{n_kept})")

    # 5. 重训 PLM-MLP 头并评估独立指标 (Gate 1)
    print(f"\n[5/6] 在扩增数据集上重训 PLM-MLP 分类头 ({mlp_epochs} epochs)...")
    loader_mlp_tr = DataLoader(
        TensorDataset(torch.from_numpy(X_train_final), torch.from_numpy(y_train_final)),
        batch_size=512, shuffle=True, pin_memory=True
    )
    loader_mlp_va = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=512, shuffle=False, pin_memory=True
    )

    mlp_head = MLP(EMB_DIM).to(device)
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

    # 独立门槛判断 (基线单种子 0.2629 / 三种子 0.2729)
    gate1_pass = standalone_f1 > 0.2629
    print(f"\n  [Gate 1 独立门槛] G001 PLM-MLP 独立 F1={standalone_f1:.4f}, AUC={standalone_auc:.4f} "
          f"(基线单种子 0.2629 / 三种子 0.2729) -> {'通过' if gate1_pass else '未通过'}")

    # 6. 融合流水线评估 (Gate 2)
    print("\n[6/6] 运行 canonical 融合流水线 (RRF + Sinkhorn + 半折 OOF)...")
    if not args.smoke:
        # 加载 deep4 与 sim
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

        # 融合评估
        f1_fused_in, auc_fused_in, mixed_scores, _, _ = full_pipe(
            deep4, sim, y_tr.astype(np.int8), y_va.astype(np.int8), len(idx_va),
            plm=final_val_probs, w=0.35
        )

        # 半折 OOF 评估
        a_idx, b_idx = train_test_split(np.arange(len(idx_va)), test_size=0.5, random_state=SEED)
        preds_oof = np.zeros_like(y_va, dtype=bool)
        for fi, ei in ((a_idx, b_idx), (b_idx, a_idx)):
            t_sub = search_scut(mixed_scores[fi], y_va.astype(np.int8)[fi])
            preds_oof[ei] = mixed_scores[ei] >= t_sub
        yb = y_va.astype(bool)
        tp = (preds_oof & yb).sum(0)
        fp = (preds_oof & ~yb).sum(0)
        fn = (~preds_oof & yb).sum(0)
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

    # 7. 保存产物与报告
    conclusion = "有效，进入提交与模型替换" if (gate1_pass and gate2_pass) else "未达到融合换锁线，关闭此尝试"
    report = {
        "experiment": "G001",
        "method": "Orphan-Targeted cWGAN-GP on 1664D + C2ST Filter",
        "smoke": args.smoke,
        "n_synth_generated": n_synth,
        "n_synth_retained": n_kept,
        "c2st_pass_rate": round(pass_rate, 4),
        "standalone_plm": {
            "f1_scut": round(standalone_f1, 4),
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

    report_file = os.path.join(REL_DIR, "G001_report.json")
    with Path(report_file).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    torch.save(G.state_dict(), os.path.join(MODEL_DIR, "g001_cwgan_gen.pt"))
    torch.save(D.state_dict(), os.path.join(MODEL_DIR, "g001_cwgan_disc.pt"))
    torch.save(mlp_head.state_dict(), os.path.join(MODEL_DIR, "g001_plm_head.pt"))
    np.savez_compressed(os.path.join(REL_DIR, "g001_augmented_val_probs.npz"), val=final_val_probs)

    print("\n" + "=" * 80)
    print(f"  G001 完成! 独立 F1={standalone_f1:.4f} | 融合 OOF={f1_oof:.4f} (Δ={delta_oof:+.4f}) -> {conclusion}")
    print(f"  报告写入: {report_file} | 总用时: {(time.time()-t_start)/60:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
