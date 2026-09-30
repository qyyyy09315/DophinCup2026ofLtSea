"""
G008 — 物理/生物描述符三通道注入的条件生成扩增（G 线核心创新刀）
================================================================
思想来源:
  - [244] Chroma (Ingraham et al., Nature 2023): "蛋白质设计 = 外部约束下的贝叶斯推断"——
    对称性/子结构/形状等约束以条件形式注入生成模型;
  - [245] RNA-EFM: 能量引导生成的约束正则思路。
关键技术约束（计划 3.5 已声明）: ESM2 为 encoder-only、无 decoder，"生成特征→解码序列→
  物理校验"链路不存在。物理约束只能以**辅助信息通道**形式注入潜空间，本刀三通道:
  1. 条件通道: G(z, y, c)，c 为 46 维可计算生物描述符（AA20 + 2-mer 组频率16 + 聚合量10，
     全部来自序列本身，不违反禁外部注释库），目标 c 从靶向切片真实描述符分布采样;
  2. 辅助回归判别器: D(x,y) 除 Wasserstein 主头外加物理回归头 φ(x)->c，
     损失含 λ_phys·(‖φ(real)-c_real‖² + ‖φ(fake)-c_cond‖²)，生成器侧同步加一致性项;
  3. 分布匹配正则: MMD(φ(fake) 批, c_real 批)（高斯核，中位数带宽）。
双门控: C2ST 严格区间放行 ∩ 描述符逐维 KS 检验（scipy.stats.ks_2samp，
  φ(fake)[:,j] vs 靶向切片真实 c[:,j]，p>0.05 维度占比 ≥ 0.60）。
下游: I096 收敛协议（平坦 MLP / 80ep / 逐轮 SCut 早停 patience=10）。
判据: Gate 1 独立 SCut F1 > 0.3036；Gate 2 融合半折 OOF >= 0.4396。
用法:
  python code/G/G008-physics-constrained-cwgan.py            # 正式 (~25 min)
  python code/G/G008-physics-constrained-cwgan.py --smoke    # 沙箱冒烟 (~40s)
输出:
  models/g008_cwgan_phys_G.pt / g008_cwgan_phys_D.pt / g008_plm_head_80ep.pt
  rel/g008_descriptors.npz, rel/g008_flow_val_probs.npz, rel/G008_report.json
"""

import os
import json
import copy
import time
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from importlib.machinery import SourceFileLoader
from scipy import stats as sps
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
NOISE_DIM = 128
COND_DIM = 128
DESC_DIM = 46
LAMBDA_GP = 10.0
LAMBDA_PHYS = 0.5
LAMBDA_MMD = 0.1
SEED = 42

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {a: i for i, a in enumerate(AA20)}
GRP4 = {a: (0 if a in "AILMFWV" else 1 if a in "NQST" else 2 if a in "KRHDE" else 3) for a in AA20}
KD_HYDRO = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
            "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
            "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}

i064 = SourceFileLoader(
    "i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")
).load_module()
g004 = SourceFileLoader(
    "g004", os.path.join(PROJECT_ROOT, "code", "G", "G004-latent-flow-matching.py")
).load_module()
load_1664d_features = g004.load_1664d_features
build_sim = g004.build_sim
MLP, ASL = i064.MLP, i064.ASL
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe


def compute_descriptors(seqs, desc="计算序列物理/生物描述符"):
    """46 维: AA 组成 20 + 2-mer 组频率 16 + 聚合量 10（纯序列可计算，零外部依赖）"""
    n = len(seqs)
    out = np.zeros((n, DESC_DIM), np.float32)
    for i, s in enumerate(tqdm(seqs, desc=desc, ncols=90)):
        L = max(len(s), 1)
        cnt = Counter(s)
        for a, j in AA_IDX.items():
            out[i, j] = cnt.get(a, 0) / L
        base = 20
        for g1 in range(4):
            for g2 in range(4):
                c12 = 0
                for k in range(len(s) - 1):
                    if GRP4.get(s[k], 3) == g1 and GRP4.get(s[k + 1], 3) == g2:
                        c12 += 1
                out[i, base + g1 * 4 + g2] = c12 / max(len(s) - 1, 1)
        base = 36
        n_charge = sum(cnt.get(a, 0) for a in "KRH") - sum(cnt.get(a, 0) for a in "DE")
        out[i, base + 0] = n_charge / L
        out[i, base + 1] = sum(cnt.get(a, 0) * KD_HYDRO[a] for a in AA20) / L
        out[i, base + 2] = sum(cnt.get(a, 0) for a in "FWY") / L
        out[i, base + 3] = sum(cnt.get(a, 0) for a in "CM") / L
        out[i, base + 4] = cnt.get("P", 0) / L
        out[i, base + 5] = cnt.get("G", 0) / L
        out[i, base + 6] = min(len(s), 1024) / 1024.0
        ent = -sum((c / L) * np.log2(c / L) for c in cnt.values() if c > 0)
        out[i, base + 7] = ent / np.log2(20)
        out[i, base + 8] = (sum(cnt.get(a, 0) for a in "KRHDE")) / L
        out[i, base + 9] = len(set(s)) / 20.0
    return out


def get_descriptors(seqs, cache_path):
    if os.path.isfile(cache_path):
        d = np.load(cache_path)
        if d["desc"].shape[0] == len(seqs):
            print(f"  描述符缓存命中: {cache_path}")
            return d["desc"].astype(np.float32)
    desc = compute_descriptors(seqs)
    np.savez_compressed(cache_path, desc=desc)
    return desc


def mmd_rbf(a, b):
    """高斯核 MMD（无偏估计，带宽=两批合并距离中位数）"""
    z = torch.cat([a, b], 0)
    d2 = torch.cdist(z, z) ** 2
    med = d2.detach().median().clamp(min=1e-6)
    gamma = 1.0 / med
    k = torch.exp(-gamma * d2)
    n1, n2 = a.size(0), b.size(0)
    return k[:n1, :n1].mean() + k[n1:, n1:].mean() - 2 * k[:n1, n1:].mean()


class PhysConditionalGenerator(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, noise_dim=NOISE_DIM, cond_dim=COND_DIM,
                 desc_dim=DESC_DIM, n_labels=N_LABELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + n_labels + desc_dim, cond_dim + 256),
            nn.LayerNorm(cond_dim + 256), nn.SiLU(),
            nn.Linear(cond_dim + 256, 1500), nn.LayerNorm(1500), nn.SiLU(),
            nn.Linear(1500, emb_dim),
        )

    def forward(self, z, y, c):
        return self.net(torch.cat([z, y, c], dim=1))


class PhysDiscriminator(nn.Module):
    """WGAN-GP 主头 + 物理回归头 φ(x)->c"""

    def __init__(self, emb_dim=EMB_DIM, cond_dim=COND_DIM, desc_dim=DESC_DIM, n_labels=N_LABELS):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(emb_dim + n_labels, 1500), nn.LayerNorm(1500), nn.SiLU(),
            nn.Linear(1500, 1024), nn.LayerNorm(1024), nn.SiLU(),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.SiLU(),
        )
        self.adv_head = nn.Linear(512, 1)
        self.phys_head = nn.Linear(512, desc_dim)

    def forward(self, x, y):
        h = self.backbone(torch.cat([x, y], dim=1))
        return self.adv_head(h), self.phys_head(h)


def compute_gradient_penalty(D, real_x, fake_x, y, device):
    alpha = torch.rand(real_x.size(0), 1, device=device)
    interp = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)
    d_interp, _ = D(interp, y)
    grad = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="沙箱快速冒烟 (~40s)")
    parser.add_argument("--epochs-wgan", type=int, default=40, help="cWGAN-GP 训练轮数 (默认 40)")
    parser.add_argument("--epochs-head", type=int, default=80, help="下游头训练轮数 (默认 80)")
    parser.add_argument("--head-patience", type=int, default=10, help="下游头 SCut 早停耐心 (默认 10)")
    parser.add_argument("--n-synth", type=int, default=3000, help="合成样本量 (默认 3000)")
    args = parser.parse_args()

    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 84)
    print(f"  G008: Physics-Constrained cWGAN-GP Latent Augmentation ({'SMOKE' if args.smoke else '正式'})")
    print(f"  Device: {device} | WGAN Epochs: {args.epochs_wgan if not args.smoke else 4} | "
          f"λ_phys={LAMBDA_PHYS} λ_mmd={LAMBDA_MMD} | Head: {args.epochs_head}ep")
    print("=" * 84)

    print("[1/7] 加载特征、标签与序列描述符...")
    X_full = load_1664d_features()
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.float32)
    desc_cache = os.path.join(REL_DIR, "g008_descriptors.npz")
    D_full = get_descriptors(train_df["sequence"].tolist(), desc_cache)
    mu, sd = D_full.mean(0), D_full.std(0) + 1e-6
    D_norm = (D_full - mu) / sd

    idx_all = np.arange(len(train_df))
    idx_tr, idx_va = train_test_split(idx_all, test_size=0.2, random_state=SEED)
    X_tr, y_tr, Dn_tr = X_full[idx_tr], y_full[idx_tr], D_norm[idx_tr]
    X_va, y_va = X_full[idx_va], y_full[idx_va]

    if args.smoke:
        X_tr, y_tr, Dn_tr = X_tr[:2048], y_tr[:2048], Dn_tr[:2048]
        X_va, y_va = X_va[:512], y_va[:512]
        wgan_epochs, head_epochs, head_patience = 4, 2, 2
        n_synth = 200
    else:
        wgan_epochs = args.epochs_wgan
        head_epochs, head_patience = args.epochs_head, args.head_patience
        n_synth = args.n_synth
    print(f"  训练集: {X_tr.shape[0]} 条, 描述符维度: {Dn_tr.shape[1]}, 特征维度: {X_tr.shape[1]}")

    # 2. cWGAN-GP + 物理三通道训练
    print(f"\n[2/7] 训练物理约束 cWGAN-GP ({wgan_epochs} epochs)...")
    loader_gan = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr), torch.from_numpy(Dn_tr)),
        batch_size=512, shuffle=True, pin_memory=True,
    )
    G = PhysConditionalGenerator().to(device)
    D = PhysDiscriminator().to(device)
    opt_g = torch.optim.AdamW(G.parameters(), lr=2e-4, weight_decay=1e-4)
    opt_d = torch.optim.AdamW(D.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for ep in range(1, wgan_epochs + 1):
        t_ep = time.time()
        d_loss_sum = g_loss_sum = phys_sum = mmd_sum = n_batches = 0.0
        for real_x, real_y, real_c in loader_gan:
            real_x, real_y, real_c = (
                real_x.to(device, non_blocking=True),
                real_y.to(device, non_blocking=True),
                real_c.to(device, non_blocking=True),
            )
            b = real_x.size(0)

            # ---- D step: Wasserstein + GP + 物理回归 ----
            z = torch.randn(b, NOISE_DIM, device=device)
            with torch.amp.autocast("cuda"):
                fake_x = G(z, real_y, real_c)
                d_fake, phys_fake = D(fake_x.detach(), real_y)
                d_real, phys_real = D(real_x, real_y)
                gp = compute_gradient_penalty(D, real_x, fake_x.detach(), real_y, device)
                phys_loss_d = ((phys_real - real_c) ** 2).mean() + ((phys_fake - real_c) ** 2).mean()
                d_loss = d_fake.mean() - d_real.mean() + LAMBDA_GP * gp + LAMBDA_PHYS * phys_loss_d
            opt_d.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(d_loss).backward()
                scaler.step(opt_d)
                scaler.update()
            else:
                d_loss.backward()
                opt_d.step()

            # ---- G step: 对抗 + 物理一致性 + MMD（D 冻结作探针）----
            z = torch.randn(b, NOISE_DIM, device=device)
            with torch.amp.autocast("cuda"):
                fake_x2 = G(z, real_y, real_c)
                for p in D.parameters():
                    p.requires_grad_(False)
                d_fake2, phys_fake2 = D(fake_x2, real_y)
                phys_consistency = ((phys_fake2 - real_c) ** 2).mean()
                mmd = mmd_rbf(phys_fake2.float(), real_c.float())
                g_loss = -d_fake2.mean() + LAMBDA_PHYS * phys_consistency + LAMBDA_MMD * mmd
            opt_g.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(g_loss).backward()
                scaler.step(opt_g)
                scaler.update()
            else:
                g_loss.backward()
                opt_g.step()
            for p in D.parameters():
                p.requires_grad_(True)

            d_loss_sum += d_loss.item(); g_loss_sum += g_loss.item()
            phys_sum += phys_consistency.item(); mmd_sum += mmd.item(); n_batches += 1
        print(f"  wgan ep{ep:02d}/{wgan_epochs:02d} | D={d_loss_sum/n_batches:.3f} | G={g_loss_sum/n_batches:.3f} | "
              f"phys_mse={phys_sum/n_batches:.4f} | mmd={mmd_sum/n_batches:.4f} "
              f"({time.time()-t_ep:.0f}s)", flush=True)

    # 3. 合成: 靶向切片（孤儿 + 低基数）条件采样
    print(f"\n[3/7] 靶向孤儿/低基数切片合成 {n_synth} 条...")
    cardinality = y_tr.sum(axis=1)
    hit_proxy = cardinality > 0
    target_mask = (cardinality <= 10) | (~hit_proxy.astype(bool))
    target_idx = np.where(target_mask)[0]
    if len(target_idx) < 100:
        target_idx = np.arange(len(y_tr))
    print(f"  靶向切片规模: {len(target_idx)} (孤儿+低基数)")
    np.random.seed(SEED)
    chosen = np.random.choice(target_idx, size=n_synth, replace=True)
    y_target = y_tr[chosen]
    c_target = Dn_tr[chosen]
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_synth, NOISE_DIM, device=device)
        fake_X = G(z, torch.from_numpy(y_target).to(device),
                   torch.from_numpy(c_target).to(device)).cpu().numpy()

    # 4. 双门控: C2ST ∩ 描述符 KS
    print("[4/7] 双门控（C2ST ∩ 描述符 KS）...")
    n_eval = min(n_synth, len(X_tr))
    rng = np.random.RandomState(SEED)
    real_idx = rng.choice(len(X_tr), size=n_eval, replace=False)
    X_c2st = np.vstack([X_tr[real_idx], fake_X])
    y_c2st = np.hstack([np.ones(n_eval), np.zeros(n_synth)])
    clf = LogisticRegression(max_iter=200, C=0.1, random_state=SEED)
    clf.fit(X_c2st, y_c2st)
    prob_real = clf.predict_proba(fake_X)[:, 1]
    mask_c2st = (prob_real >= 0.35) & (prob_real <= 0.65)

    D.eval()
    with torch.no_grad():
        phys_fake = []
        for s in range(0, n_synth, 512):
            xb = torch.from_numpy(fake_X[s:s + 512]).to(device)
            yb_ = torch.from_numpy(y_target[s:s + 512]).to(device)
            _, ph = D(xb, yb_)
            phys_fake.append(ph.cpu().numpy())
        phys_fake = np.vstack(phys_fake)
    c_target_real = Dn_tr[chosen]
    ks_pass = []
    for j in range(DESC_DIM):
        _, pval = sps.ks_2samp(phys_fake[:, j], c_target_real[:, j])
        ks_pass.append(pval > 0.05)
    ks_ratio = float(np.mean(ks_pass))
    mask_ks = ks_ratio >= 0.60
    mask_final = mask_c2st & mask_ks
    n_kept = int(mask_final.sum())
    print(f"  C2ST 放行率: {mask_c2st.mean():.1%} | KS 通过维度占比: {ks_ratio:.1%} (门 60%) | "
          f"双门放行: {n_kept}/{n_synth} ({mask_final.mean():.1%})")
    if n_kept >= 100:
        X_final = np.vstack([X_tr, fake_X[mask_final]])
        y_final = np.vstack([y_tr, y_target[mask_final]])
    else:
        print("  [说明] 双门放行不足，退回 C2ST 单门；仍不足则 50% 拟真保底（G 系列同款）")
        mask_final = mask_c2st if mask_c2st.sum() >= 100 else None
        if mask_final is None:
            top_half = np.argsort(np.abs(prob_real - 0.5))[: max(n_synth // 2, 100)]
            X_final = np.vstack([X_tr, fake_X[top_half]])
            y_final = np.vstack([y_tr, y_target[top_half]])
            n_kept = len(top_half)
        else:
            n_kept = int(mask_final.sum())
            X_final = np.vstack([X_tr, fake_X[mask_final]])
            y_final = np.vstack([y_tr, y_target[mask_final]])
    print(f"  最终扩增: {len(X_tr)} -> {len(X_final)} (+{n_kept})")

    # 5. 下游 I096 收敛头
    print(f"\n[5/7] 扩增数据上训练收敛头 ({head_epochs}ep + SCut 早停 {head_patience})...")
    loader_hd_tr = DataLoader(
        TensorDataset(torch.from_numpy(X_final), torch.from_numpy(y_final)),
        batch_size=512, shuffle=True, pin_memory=True,
    )
    loader_hd_va = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=512, shuffle=False, pin_memory=True,
    )
    head = MLP(EMB_DIM).to(device)
    crit = ASL()
    opt_h = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sch_h = torch.optim.lr_scheduler.CosineAnnealingLR(opt_h, T_max=head_epochs, eta_min=1e-5)
    sc2 = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_f1, best_ep, stall, best_state = 0.0, 0, 0, None
    for ep in range(1, head_epochs + 1):
        t_ep = time.time()
        head.train()
        loss_sum, n_seen = 0.0, 0
        for bx, by in loader_hd_tr:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            opt_h.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = crit(head(bx), by)
            if sc2 is not None:
                sc2.scale(loss).backward()
                sc2.step(opt_h)
                sc2.update()
            else:
                loss.backward()
                opt_h.step()
            loss_sum += loss.item() * bx.size(0)
            n_seen += bx.size(0)
        sch_h.step()
        head.eval()
        preds = []
        with torch.no_grad():
            for bx, _ in loader_hd_va:
                with torch.amp.autocast("cuda"):
                    pr = torch.sigmoid(head(bx.to(device, non_blocking=True)).float()).cpu().numpy()
                preds.append(pr)
        val_probs = np.vstack(preds)
        f1_05 = macro_f1(val_probs, y_va.astype(np.int8), 0.5)
        ths_e = search_scut(val_probs, y_va.astype(np.int8))
        f1_sc = macro_f1(val_probs, y_va.astype(np.int8), ths_e)
        if f1_sc > best_val_f1:
            best_val_f1, best_ep, stall = f1_sc, ep, 0
            best_state = copy.deepcopy(head.state_dict())
            mark = "(*)"
        else:
            stall += 1
            mark = "   "
        print(f"  head ep{ep:02d}/{head_epochs:02d} | loss={loss_sum/n_seen:.4f} | "
              f"val_f1(0.5)={f1_05:.4f} | val_f1(SCut)={f1_sc:.4f} | best={best_val_f1:.4f}@ep{best_ep:02d} {mark} "
              f"({time.time()-t_ep:.0f}s)", flush=True)
        if stall >= head_patience:
            print(f"  [早停] 连续 {head_patience} 轮未提升，早停于 epoch {ep}")
            break

    head.load_state_dict(best_state)
    head.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in loader_hd_va:
            with torch.amp.autocast("cuda"):
                pr = torch.sigmoid(head(bx.to(device, non_blocking=True)).float()).cpu().numpy()
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
          f"R={standalone_rec:.4f}, P={standalone_prec:.4f} (I096 锚点 0.3036) "
          f"-> {'通过' if gate1_pass else '未通过'}")

    # 6. Gate 2
    print("\n[6/7] canonical 融合流水线评估 Gate 2...")
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
        print(f"  [Gate 2 融合门槛] 样内 F1={f1_fused_in:.4f}, AUC={auc_fused_in:.4f}, "
              f"半折 OOF F1={f1_oof:.4f} (基线 0.4386, ΔOOF={delta_oof:+.4f}) "
              f"-> {'通过' if gate2_pass else '未通过'}")
    else:
        f1_fused_in, auc_fused_in, f1_oof, delta_oof = 0.0, 0.0, 0.0, 0.0
        gate2_pass = False
        print("  [SMOKE] 跳过完整融合管线")

    conclusion = ("有效，物理约束生成传导成功，进入 I100 全量换锁流程"
                  if (gate1_pass and gate2_pass) else "未过双门槛，G 线物理约束刀关闭")
    report = {
        "experiment": "G008",
        "method": "Physics-Constrained cWGAN-GP (desc-condition + aux-regression D + MMD) + dual gate",
        "smoke": args.smoke,
        "desc_dim": DESC_DIM,
        "c2st_pass_rate": round(float(mask_c2st.mean()), 4),
        "ks_pass_dim_ratio": round(ks_ratio, 4),
        "dual_gate_kept": n_kept,
        "n_synth": n_synth,
        "standalone_plm": {
            "f1_scut": round(standalone_f1, 4),
            "precision": round(standalone_prec, 4),
            "recall": round(standalone_rec, 4),
            "auc": round(standalone_auc, 4),
            "gate1_pass": gate1_pass,
            "anchor_i096_no_aug": 0.3036,
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
    with Path(os.path.join(REL_DIR, "G008_report.json")).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if not args.smoke:
        torch.save(G.state_dict(), os.path.join(MODEL_DIR, "g008_cwgan_phys_G.pt"))
        torch.save(D.state_dict(), os.path.join(MODEL_DIR, "g008_cwgan_phys_D.pt"))
        torch.save(head.state_dict(), os.path.join(MODEL_DIR, "g008_plm_head_80ep.pt"))
        np.savez_compressed(os.path.join(REL_DIR, "g008_flow_val_probs.npz"),
                            val=final_val_probs.astype(np.float32))

    print("\n" + "=" * 84)
    print(f"  G008 完成! C2ST {mask_c2st.mean():.1%} | KS {ks_ratio:.1%} | 独立 F1={standalone_f1:.4f} | "
          f"融合 OOF={f1_oof:.4f} (Δ={delta_oof:+.4f}) -> {conclusion}")
    print(f"  报告: {os.path.join(REL_DIR, 'G008_report.json')} | 总用时: {(time.time()-t_start)/60:.1f} min")
    print("=" * 84)


if __name__ == "__main__":
    from tqdm import tqdm
    main()
