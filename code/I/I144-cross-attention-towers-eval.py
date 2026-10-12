"""
I144 — 跨模态多头交叉注意力融合塔 (Cross-Modal Cross-Attention Towers, CMAT) 评估
========================================================================================
理论背景与设计依据:
  1. 跨模态自适应结构-序列查询假说 (Cross-Modal Cross-Attention):
     在 DMT 架构中，各模态独立降维后直接拼接属于静态早期融合；
     引入多头交叉注意力机制 (Multi-Head Cross-Attention)：
     以 ESM-2 650M 多层结构特征 (t3) 作为 Query，以 ESM-2 150M (t1) 与 ProtBERT (t2) 作为 Key/Value，
     让高阶空间折叠拓扑主动引导对底层序列基序的动态查询加权。
  2. 实测结果:
     - CMAT 单模型三种子 SWA: microAUC = 0.8828 (相比 DMT-v1 的 0.8815 进一步提升 +0.0013！)；
     - 六元解耦融合 (Hexa-Blend: Quad 0.70 + DMT 0.15 + CMAT 0.15):
       microAUC = 0.8952，但全流程 F1 0.45118 低于 Penta 的 0.45125；
     - 原报告 R=20 平均差 -0.00002、胜率 0.40；该胜率不是独立 OOF 显著性。
     - 2026-10-12 两级阈值交叉拟合复核：Hexa 较 Penta -0.00023273，当前配方不换锁。
"""

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
REPORT_PATH = os.path.join(REL_DIR, "I144_report.json")

N_LABELS = 500
SEED = 42
W_PLM = 0.35
ALIGN_W = 0.50
SIM_W = 0.80
SINK_TAU = 0.20
SINK_ALPHA = 0.55
WD_RRF = 0.6
WS_RRF = 1.4
GAMMA_ENT = 0.10

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
i085 = SourceFileLoader("i085", os.path.join(PROJECT_ROOT, "code", "I", "I085-top100-gated-agg.py")).load_module()
i104 = SourceFileLoader("i104", os.path.join(PROJECT_ROOT, "code", "I", "I104-top20sim-submit.py")).load_module()
i113 = SourceFileLoader("i113", os.path.join(PROJECT_ROOT, "code", "I", "I113-simmix-submit.py")).load_module()
i128 = SourceFileLoader("i128", os.path.join(PROJECT_ROOT, "code", "I", "I128-twilight-markov-diffusion-eval.py")).load_module()

macro_f1 = i064.macro_f1
search_scut = i064.search_scut
ranks = i064.ranks
col_norm = i064.col_norm
sinkhorn = i064.sinkhorn
conditional_diffuse = i128.conditional_diffuse


class CMAT_MLP(nn.Module):
    def __init__(self, d1=640, d2=1024, d3=3840, out_dim=N_LABELS, dp=0.2):
        super().__init__()
        self.t1 = nn.Sequential(nn.Linear(d1, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dp), nn.Linear(256, 128), nn.LayerNorm(128))
        self.t2 = nn.Sequential(nn.Linear(d2, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dp), nn.Linear(256, 128), nn.LayerNorm(128))
        self.t3 = nn.Sequential(nn.Linear(d3, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dp), nn.Linear(512, 256), nn.LayerNorm(256))
        
        self.q_proj = nn.Linear(256, 128)
        self.mha = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=dp, batch_first=True)
        
        fdim = 256 + 128 + 128  # 512D
        self.fnorm = nn.LayerNorm(fdim)
        self.b1 = nn.Sequential(nn.Linear(fdim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dp), nn.Linear(512, 512), nn.LayerNorm(512))
        self.b2 = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dp), nn.Linear(512, 512), nn.LayerNorm(512))
        self.act = nn.GELU()
        self.head = nn.Linear(512, out_dim)

    def forward(self, x1, x2, x3):
        t1, t2, t3 = self.t1(x1), self.t2(x2), self.t3(x3)
        q = self.q_proj(t3).unsqueeze(1)
        kv = torch.stack([t1, t2], dim=1)
        attn_out, _ = self.mha(q, kv, kv)
        attn_flat = attn_out.squeeze(1)
        
        fused = torch.cat([t3, attn_flat, t1], dim=-1)
        h0 = self.fnorm(fused)
        h1 = self.act(h0 + self.b1(h0))
        h2 = self.act(h1 + self.b2(h1))
        return self.head(h2)


class ASL(nn.Module):
    def forward(self, lg, tg):
        p = torch.sigmoid(lg.float()).clamp(1e-6, 1 - 1e-6)
        pos = -tg * torch.log(p)
        pm = torch.clamp(p - 0.05, min=0.0)
        neg = -(1 - tg) * (pm ** 4.0) * torch.log(1 - pm)
        return (pos + neg).mean()


def main():
    t_start = time.time()
    print("=" * 96)
    print("  I144: 跨模态多头交叉注意力融合塔 (Cross-Modal Cross-Attention Towers, CMAT) 评估")
    print("=" * 96)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    Y_all = train_df[[f"label_{i}" for i in range(N_LABELS)]].values.astype(np.float32)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.20, random_state=SEED)
    y_va = Y_all[idx_va]; y_tr = Y_all[idx_tr]; n_va = len(idx_va)

    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}

    # 1. 提取并中心化多模态特征
    df_e = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    df_p = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
    emb_650 = np.load(os.path.join(REL_DIR, "esm2_650m_embeddings.npz"))

    ec = [c for c in df_e.columns if c.startswith("emb_")]
    pc = [c for c in df_p.columns if c.startswith("emb_")]
    X1_all = df_e[ec].values.astype(np.float32)
    X2_all = df_p[pc].values.astype(np.float32)
    X3_all = emb_650["train_last3"].astype(np.float32)

    X1_tr, X1_va = X1_all[idx_tr], X1_all[idx_va]
    X2_tr, X2_va = X2_all[idx_tr], X2_all[idx_va]
    X3_tr, X3_va = X3_all[idx_tr], X3_all[idx_va]

    mu1, mu2, mu3 = X1_tr.mean(axis=0, keepdims=True), X2_tr.mean(axis=0, keepdims=True), X3_tr.mean(axis=0, keepdims=True)
    X1_tr_c, X1_va_c = X1_tr - mu1, X1_va - mu1
    X2_tr_c, X2_va_c = X2_tr - mu2, X2_va - mu2
    X3_tr_c, X3_va_c = X3_tr - mu3, X3_va - mu3

    # 2. 训练 CMAT-MLP 三种子 SWA 模型
    print("  训练 CMAT-MLP 跨模态注意力网络 (3 种子, 35ep, SWA)...")
    cmat_preds = []
    for s in [11, 22, 33]:
        torch.manual_seed(s); np.random.seed(s); torch.cuda.manual_seed_all(s)
        tl = DataLoader(TensorDataset(torch.from_numpy(X1_tr_c), torch.from_numpy(X2_tr_c), torch.from_numpy(X3_tr_c), torch.from_numpy(y_tr.astype(np.float32))), batch_size=512, shuffle=True, pin_memory=True)
        m = CMAT_MLP().cuda()
        swa_m = AveragedModel(m)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35, eta_min=1e-5)
        crit = ASL(); sc = torch.amp.GradScaler("cuda")
        for ep in range(1, 36):
            m.train()
            for b1, b2, b3, by in tl:
                b1, b2, b3, by = b1.cuda(), b2.cuda(), b3.cuda(), by.cuda()
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    out = m(b1, b2, b3)
                    loss = crit(out, by)
                sc.scale(loss).backward(); sc.step(opt); sc.update()
            sch.step()
            if ep >= 20:
                swa_m.update_parameters(m)
        swa_m.eval()
        preds = []
        with torch.no_grad():
            for st in range(0, len(X1_va_c), 512):
                b1, b2, b3 = torch.from_numpy(X1_va_c[st:st+512]).cuda(), torch.from_numpy(X2_va_c[st:st+512]).cuda(), torch.from_numpy(X3_va_c[st:st+512]).cuda()
                with torch.amp.autocast("cuda"):
                    pr = torch.sigmoid(swa_m(b1, b2, b3).float()).cpu().numpy()
                preds.append(pr)
        cmat_preds.append(np.vstack(preds))
    p_cmat_swa = np.mean(cmat_preds, axis=0)
    np.savez_compressed(os.path.join(REL_DIR, "i144_cmat_swa_val.npz"), val=p_cmat_swa)

    u_cmat = float(roc_auc_score(y_va, p_cmat_swa, average="micro"))
    f1_cmat = float(macro_f1(p_cmat_swa, y_va, search_scut(p_cmat_swa, y_va)))
    print(f"  CMAT 单模型 SWA -> microAUC = {u_cmat:.4f}, F1 = {f1_cmat:.4f}")

    # 3. 加载已有各路表示构建多模型融合
    p_smooth = np.load(os.path.join(REL_DIR, "multi_plm_mlp_val_probs.npz"))["val"]
    p_conv = np.load(os.path.join(REL_DIR, "i106_conv_multiseed_val.npz"))["val"]
    p_cen1664_swa = np.load(os.path.join(REL_DIR, "i122_1664d_swa_val.npz"))["val"]
    p_cen3840_swa = np.load(os.path.join(REL_DIR, "i122_3840d_swa_val.npz"))["val"]
    p_dmt_swa = np.load(os.path.join(REL_DIR, "i123_dmt_swa_val.npz"))["val"]
    p_quad_swa = (0.40 * p_smooth + 0.20 * p_conv + 0.20 * p_cen1664_swa + 0.20 * p_cen3840_swa)

    # Penta-DMT 基准
    p_penta = 0.70 * p_quad_swa + 0.30 * p_dmt_swa
    # Hexa-CMAT 增强版
    p_hexa = 0.70 * p_quad_swa + 0.15 * p_dmt_swa + 0.15 * p_cmat_swa

    u_penta = float(roc_auc_score(y_va, p_penta, average="micro"))
    u_hexa = float(roc_auc_score(y_va, p_hexa, average="micro"))
    print(f"  Penta-DMT 基准 microAUC = {u_penta:.4f}")
    print(f"  Hexa-CMAT 融合 microAUC = {u_hexa:.4f} (ΔAUC = {u_hexa - u_penta:+.4f})")

    # 4. 全流程融合与 R=20 配对检验
    cnn_raw = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1_v = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2_v = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    deep4_rebal = 0.12 * t1_v + 0.18 * t2_v + 0.42 * cnn_raw + 0.28 * c005_v

    idx100_v, bits100_v = i085.parse_topk(
        os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001_top100.tsv"), val_pids, train_map)
    v2v = i104.parse_2hop(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv"), val_map, val_map)
    s20_raw = i085.agg_topk(idx100_v, bits100_v, y_tr, 20)
    s3_raw = i085.agg_topk(idx100_v, bits100_v, y_tr, 3)
    hit = idx100_v[:, 0] >= 0
    top_bits = bits100_v[:, 0]
    kmer = i104.build_kmer(y_tr, "val")
    s20_d = conditional_diffuse(s20_raw, hit, top_bits, v2v, n_va, threshold_b=70, beta_diff=0.20)
    s3_d = conditional_diffuse(s3_raw, hit, top_bits, v2v, n_va, threshold_b=70, beta_diff=0.20)
    sim_diff = 0.50 * (0.75 * s20_d + 0.25 * s3_d) + 0.50 * kmer

    p_sim_clip = np.clip(sim_diff, 1e-6, 1.0 - 1e-6)
    ent_sim = -(p_sim_clip * np.log2(p_sim_clip) + (1.0 - p_sim_clip) * np.log2(1.0 - p_sim_clip)).sum(axis=1)
    ent_norm = (ent_sim - ent_sim.min()) / (ent_sim.max() - ent_sim.min() + 1e-7)
    w_sim_gated = 0.80 * ((1.0 - ent_norm[:, None]) ** GAMMA_ENT)

    def run_full_pipe(plm_prob):
        deep5 = (1.0 - W_PLM) * deep4_rebal + W_PLM * plm_prob
        r_d = ranks(deep5); r_s = ranks(sim_diff)
        rrf = col_norm(WD_RRF / (30 + r_d + 1) + WS_RRF / (30 + r_s + 1))
        s = deep5 + w_sim_gated * sim_diff + 0.50 * rrf
        t_tmp = search_scut(s, y_va)
        row_m = np.maximum((s >= t_tmp).sum(1).astype(np.float64), 1.0)
        col_m = np.maximum(y_tr.mean(0) * len(s), 1e-6); col_m *= row_m.sum() / col_m.sum()
        ker = np.exp((s - s.max(axis=1, keepdims=True)) / SINK_TAU)
        p_trans = sinkhorn(ker, row_m, col_m)
        final = (1.0 - SINK_ALPHA) * s + SINK_ALPHA * p_trans
        return final

    final_penta = run_full_pipe(p_penta)
    final_hexa = run_full_pipe(p_hexa)

    f1_penta = float(macro_f1(final_penta, y_va, search_scut(final_penta, y_va)))
    f1_hexa = float(macro_f1(final_hexa, y_va, search_scut(final_hexa, y_va)))

    print(f"\n  Penta-DMT 全流程 val F1 = {f1_penta:.5f}")
    print(f"  Hexa-CMAT 全流程 val F1 = {f1_hexa:.5f} (ΔF1 = {f1_hexa - f1_penta:+.5f})")

    # R=20 配对检验
    np.random.seed(42)
    R = 20
    sub_size = n_va // 2
    deltas = []
    f1_pentas, f1_hexas = [], []

    for r in tqdm(range(R), desc="  Pairing Hexa vs Penta", ncols=80):
        sub_idx = np.random.choice(n_va, size=sub_size, replace=False)
        y_sub = y_va[sub_idx]
        th_p = search_scut(final_penta[sub_idx], y_sub)
        fp = macro_f1(final_penta[sub_idx], y_sub, th_p)
        th_h = search_scut(final_hexa[sub_idx], y_sub)
        fh = macro_f1(final_hexa[sub_idx], y_sub, th_h)
        deltas.append(fh - fp)
        f1_pentas.append(fp)
        f1_hexas.append(fh)

    deltas = np.array(deltas)
    mean_delta = float(deltas.mean())
    std_delta = float(deltas.std())
    ci95_lo = float(mean_delta - 1.96 * std_delta / np.sqrt(R))
    ci95_hi = float(mean_delta + 1.96 * std_delta / np.sqrt(R))
    p_gt0 = float((deltas > 0).mean())

    print(f"\n  R=20 配对检验结果: ΔOOF = {mean_delta:+.5f} ± {std_delta:.5f}, "
          f"CI95 = [{ci95_lo:+.5f}, {ci95_hi:+.5f}], P(Δ>0) = {p_gt0:.2f}")

    elapsed = round(time.time() - t_start, 1)
    report = {
        "experiment": "I144",
        "method": "Cross-Modal Cross-Attention Towers (CMAT-MLP, N001)",
        "standalone_cmat": {
            "microAUC": round(u_cmat, 4),
            "f1": round(f1_cmat, 4),
        },
        "ensemble_microAUC": {
            "penta_base": round(u_penta, 4),
            "hexa_cmat": round(u_hexa, 4),
            "delta": round(u_hexa - u_penta, 4),
        },
        "full_pipeline_f1": {
            "penta_f1": round(f1_penta, 5),
            "hexa_f1": round(f1_hexa, 5),
            "delta_single": round(f1_hexa - f1_penta, 5),
        },
        "r20_paired_analysis": {
            "mean_delta": round(mean_delta, 5),
            "std_delta": round(std_delta, 5),
            "ci95_lo": round(ci95_lo, 5),
            "ci95_hi": round(ci95_hi, 5),
            "p_gt0": round(p_gt0, 2),
            "mean_hexa_f1": round(float(np.mean(f1_hexas)), 5),
            "mean_penta_f1": round(float(np.mean(f1_pentas)), 5),
        },
        "elapsed_seconds": elapsed,
        "verdict": (f"CMAT 单模型 microAUC={u_cmat:.4f}，六元融合 microAUC={u_hexa:.4f}；"
                    f"全流程 F1 差={f1_hexa - f1_penta:+.5f}；"
                    f"R=20 重复半样本平均差={mean_delta:+.5f}、胜率={p_gt0:.2f}；"
                    "胜率不等同独立 OOF 显著性，方法准入以完整融合 Macro F1 为准"),
    }

    with Path(REPORT_PATH).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n完整报告已导出至: {REPORT_PATH}")
    print("=" * 96)


if __name__ == "__main__":
    main()
