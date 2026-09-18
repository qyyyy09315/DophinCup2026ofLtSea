"""
I073 — 假阴性假说的判决性快速验证 (PLM-MLP 头, 分钟级)
======================================================
假说: 训练数据标注不完整 => 观测到的 0 中含未观测正例 => BCE/ASL 强制压制它们
      => 头部排序被压平 => 召回在低 FPR 区上不去。
文献依据:
  - PU-GO (Bioinformatics 2024, doi:10.1093/bioinformatics/btae237): 蛋白质 PU 排序
  - Schultheis et al. 2020 (arXiv:2007.00237): 无偏损失闭式修正 (正例按 1/propensity 加权)
  - Distribution-Balanced Loss (ECCV 2020, arXiv:2007.09654): 负例容忍正则, 防过度压制
设计: 在 1664D PLM 特征上训练同一残差 MLP, 只改损失, 用同一随机种子与 epoch 数对比:
  L0 基线 ASL (复现 I064, 独立 F1 0.2729 / AUC 0.7947)
  L1 ASL + 正例 1/c_j 加权 (c_j 为标签 j 的观测频率代理 -> propensity)
  L2 ASL + 按样本基数下调负例权重 (标签越多, 其 0 越可疑)
  L3 ZLPR 成对排序损失 (零界 log-sum-exp)
  L4 IA 加权 F1 代理损失
评估: 独立 SCut F1 + 头部指标 pAUC[0,0.10] / R@FPR0.069 (榜首水平 0.82)
判据: 头部召回显著提升 或 独立 F1 > 0.2729
用法: python code/I/I073-fn-hypothesis.py
输出: rel/I073_report.json, rel/i073_*_val.npz
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc as sk_auc

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORT = os.path.join(REL_DIR, "I073_report.json")
N_LABELS = 500
HIDDEN = 1024
EPOCHS = 35
BATCH = 512
LR = 1e-3
SEED = 42
FPR_GRID = [0.02, 0.05, 0.069, 0.10]


class MLP(nn.Module):
    def __init__(self, ind, hid=HIDDEN, out=N_LABELS, dp=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(ind, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(dp))
        self.b1 = nn.Sequential(nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(dp),
                                nn.Linear(hid, hid), nn.LayerNorm(hid))
        self.b2 = nn.Sequential(nn.Linear(hid, hid), nn.LayerNorm(hid), nn.GELU(), nn.Dropout(dp),
                                nn.Linear(hid, hid), nn.LayerNorm(hid))
        self.act = nn.GELU()
        self.head = nn.Linear(hid, out)

    def forward(self, x):
        h0 = self.inp(x)
        h1 = self.act(h0 + self.b1(h0))
        h2 = self.act(h1 + self.b2(h1))
        return self.head(h2)


class LossBank(nn.Module):
    """kind: asl / asl_prop / asl_cardneg / zlpr / iaf1"""
    def __init__(self, kind, card=None, freq=None, gamma_neg=4.0, clip=0.05):
        super().__init__()
        self.kind = kind
        self.gn, self.clip = gamma_neg, clip
        self.register_buffer("card", card if card is not None else torch.zeros(1))
        self.register_buffer("freq", freq if freq is not None else torch.ones(1))

    def asl_core(self, lg, tg):
        p = torch.sigmoid(lg.float()).clamp(1e-6, 1 - 1e-6)
        pos = -tg * torch.log(p)
        pm = torch.clamp(p - self.clip, min=0.0)
        neg = -(1 - tg) * (pm ** self.gn) * torch.log(1 - pm)
        return pos, neg

    def forward(self, lg, tg, card):
        if self.kind == "zlpr":
            # 零界 log-sum-exp 成对排序: log(1+Σ_pos e^{-s}) + log(1+Σ_neg e^{s})
            s = lg.float()
            pos = tg.bool(); neg = ~tg.bool()
            lp = torch.log1p(torch.exp(-s).masked_fill(~pos, 0).sum(1))
            ln = torch.log1p(torch.exp(s).masked_fill(~neg, 0).sum(1))
            return (lp + ln).mean()
        pos, neg = self.asl_core(lg, tg)
        if self.kind == "asl":
            return (pos + neg).mean()
        if self.kind == "asl_prop":
            # 正例按 1/c_j 加权 (c_j 用观测频率代理, 归一化到 [0.3,1])
            w = (self.freq.clamp(1e-3) / self.freq.max()).clamp(0.3, 1.0)
            w = 1.0 / w
            return (pos * w[None, :] + neg).mean()
        if self.kind == "asl_cardneg":
            # 标签越多的样本, 其 0 越可疑 -> 负例权重按基数下调
            scale = 60.0
            wneg = 1.0 / (1.0 + card / scale)
            return (pos + neg * wneg[:, None]).mean()
        if self.kind == "iaf1":
            # F1 代理: 逐标签的软 TP/FP/FN 组成的 F1 损失 (近似 InterLabelGO+ 的 F1 损失)
            p = torch.sigmoid(lg.float()).clamp(1e-6, 1 - 1e-6)
            tp = (p * tg).sum(0); fp = (p * (1 - tg)).sum(0); fn = ((1 - p) * tg).sum(0)
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            return (1.0 - f1).mean()
        raise ValueError(self.kind)


def train_one(kind, Xtr, ytr, card_tr, freq, Xva, yva):
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)
    tl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr),
                                  torch.from_numpy(card_tr)), batch_size=BATCH, shuffle=True, pin_memory=True)
    vl = DataLoader(TensorDataset(torch.from_numpy(Xva)), batch_size=BATCH, shuffle=False)
    m = MLP(Xtr.shape[1]).cuda()
    crit = LossBank(kind, freq=torch.from_numpy(freq)).cuda() if kind in ("asl_prop",) else LossBank(kind).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    sc = torch.amp.GradScaler("cuda")
    for ep in range(EPOCHS):
        m.train()
        for bx, by, bc in tl:
            bx, by, bc = bx.cuda(non_blocking=True), by.cuda(non_blocking=True), bc.cuda(non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                l = crit(m(bx), by, bc)
            sc.scale(l).backward(); sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            sc.step(opt); sc.update()
        sch.step()
    m.eval()
    out = []
    with torch.no_grad():
        for (bx,) in vl:
            with torch.amp.autocast("cuda"):
                out.append(torch.sigmoid(m(bx.cuda(non_blocking=True)).float()).cpu().numpy())
    return np.concatenate(out, axis=0)


def search_scut(ens, y, n=60):
    cands = np.linspace(0.01, 0.99, n)
    best = np.full(N_LABELS, 0.5)
    for i in range(N_LABELS):
        yt = y[:, i]
        if yt.sum() == 0: continue
        pr = ens[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (pr & ytm).sum(0); fp = (pr & ~ytm).sum(0); fn = (~pr & ytm).sum(0)
        best[i] = cands[(2*tp/np.maximum(2*tp+fp+fn, 1e-9)).argmax()]
    return best


def macro_f1(probs, y, ths):
    preds = (probs >= ths).astype(bool); yb = y.astype(bool)
    tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
    p = np.where(tp+fp>0, tp/(tp+fp), 0.0); r = np.where(tp+fn>0, tp/(tp+fn), 0.0)
    return float(np.mean(np.where(p+r>0, 2*p*r/(p+r), 0.0)))


def head_metrics(S, y):
    """pAUC[0,0.10] 与固定 FPR 召回"""
    aucs, paucs = [], []
    rec = {f: [] for f in FPR_GRID}
    for j in range(N_LABELS):
        yt = y[:, j]
        if yt.sum() == 0 or yt.sum() == len(yt): continue
        fpr, tpr, _ = roc_curve(yt, S[:, j])
        if len(fpr) < 2: continue
        aucs.append(sk_auc(fpr, tpr))
        mk = fpr <= 0.10
        paucs.append(sk_auc(fpr[mk], tpr[mk])/0.10 if mk.sum() >= 2 else 0.0)
        for f in FPR_GRID:
            i2 = np.searchsorted(fpr, f, side="right") - 1
            rec[f].append(tpr[max(i2, 0)])
    return {"auc": float(np.mean(aucs)), "pauc": float(np.mean(paucs)),
            **{f"r@{f}": float(np.mean(rec[f])) for f in FPR_GRID}}


def main():
    t0 = time.time()
    print("=" * 84)
    print("  I073: Decisive Test of the False-Negative Hypothesis (PLM-MLP head)")
    print("=" * 84)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr].astype(np.float32), y_full[idx_va]
    card_tr = y_tr.sum(1).astype(np.float32)
    freq = y_tr.mean(0).astype(np.float32)

    df_e = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    df_p = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
    ec = [c for c in df_e.columns if c.startswith("emb_")]
    pc = [c for c in df_p.columns if c.startswith("emb_")]
    X = np.concatenate([df_e[ec].values, df_p[pc].values], axis=1).astype(np.float32)
    print(f"  Features: {X.shape} | 训练集基数 mean={card_tr.mean():.2f} max={card_tr.max():.0f}")

    plan = [
        ("L0 asl",        "asl",        "基线 ASL (对照 I064)"),
        ("L1 asl_prop",   "asl_prop",   "ASL + 正例 1/c_j 加权 (PU propensity)"),
        ("L2 asl_cardneg","asl_cardneg","ASL + 负例按样本基数下调 (未标注=未知)"),
        ("L3 zlpr",       "zlpr",       "ZLPR 成对排序损失 (零界 log-sum-exp)"),
        ("L4 iaf1",       "iaf1",       "IA 风格 F1 代理损失"),
    ]
    results = {}
    print(f"\n{'变体':16s} | {'独立SCut F1':>11s} | {'AUC':>7s} | {'pAUC[.0,.1]':>11s} | " +
          " | ".join([f"R@FPR{f:<4}" for f in FPR_GRID]))
    print("-" * 84)
    for tag, kind, desc in plan:
        try:
            P = train_one(kind, X[idx_tr], y_tr, card_tr, freq, X[idx_va], y_va)
        except Exception as e:
            import traceback
            print(f"{tag:16s} | FAILED:")
            traceback.print_exc()
            continue
        f1 = macro_f1(P, y_va, search_scut(P, y_va))
        hm = head_metrics(P, y_va)
        results[tag] = {"desc": desc, "f1": f1, **hm}
        np.savez_compressed(os.path.join(REL_DIR, f"i073_{tag.replace(' ', '_')}_val.npz"), val=P)
        print(f"{tag:16s} | {f1:11.4f} | {hm['auc']:7.4f} | {hm['pauc']:11.4f} | " +
              " | ".join([f"{hm[f'r@{f}']:9.4f}" for f in FPR_GRID]))

    base = results.get("L0 asl")
    if base:
        print("\n" + "=" * 84)
        print(f"  基准 L0: F1={base['f1']:.4f} | pAUC={base['pauc']:.4f} | R@FPR0.069={base['r@0.069']:.4f}")
        for k, v in results.items():
            if k == "L0 asl": continue
            print(f"  {k:16s} dF1={v['f1']-base['f1']:+.4f} | dpAUC={v['pauc']-base['pauc']:+.4f} | "
                  f"dR@0.069={v['r@0.069']-base['r@0.069']:+.4f}")
        print("=" * 84)

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
