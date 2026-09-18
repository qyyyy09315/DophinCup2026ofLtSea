"""
I074 — PU 倾向加权正例能否传导到融合分
=======================================
I073 结论 (PLM-MLP 独立头, 5 变体, 8.3 min, seed 42):
  L1 asl_prop  (正例按 1/freq 加权)  +0.0088 F1 | +0.0095 pAUC | +0.0106 R@FPR0.069  <- 唯一正向
  L2 cardneg   -0.0054 | L3 zlpr -0.0212 | L4 iaf1 -0.0439   <- 换排序目标有害

结论解读: 数据存在真实漏标 (未标注 != 负例), 但正确处方是"重新加权正例",
          而不是"更换排序损失目标"。ZLPR/IA-F1 把全部 0 当真负例, 放大漏标噪声。

本实验回答唯一有意义的问题: 该增益是否传导到最终融合分?
独立头变强 != 融合变强 (I063 教训: 独立 0.2590->0.2867 而融合 0.4471->0.4462)。
I073 的 L1 独立提升幅度 (+0.0088) 与 I063 的"独立强但融合弱"情形量级相近, 必须实测。

A 段 (单种子 seed=11): 重训对照 + 4 组 (alpha, Wmax), 判据 = 融合分 val F1
B 段 (3 种子 11/22/33): A 段最优配置 vs 缓存基线 plm1664_multiseed_val.npz

融合链路严格复用 I064 的 full_pipe (w=0.35, 1e-3 AdamW, 35 ep, 按独立 F1 选轮次),
保证与 0.4475 基线逐行可比。
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from importlib.machinery import SourceFileLoader
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc as sk_auc

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(MODEL_DIR, "tck002-ensemble.json")
REPORT = os.path.join(REL_DIR, "I074_report.json")

N_LABELS = 500
FPR_GRID = [0.02, 0.05, 0.069, 0.10]
W_PLM = 0.35          # I064 最优线性权重
BASE_F1 = 0.4475      # I064/I065a 缓存基线, 用于传导判定

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
MLP, macro_f1, search_scut = i064.MLP, i064.macro_f1, i064.search_scut
ranks, col_norm, sinkhorn, full_pipe = i064.ranks, i064.col_norm, i064.sinkhorn, i064.full_pipe


class ASLProp(nn.Module):
    """ASL + 正例按 f_j^{-alpha} 加权 (f_j = 训练集标签频率, 权重截断到 [1, Wmax])。

    alpha=0 时 w 恒为 1, 严格退化为 I064 的 ASL, 用作"重训对照"。
    倾向加权动机 (PU 学习): 稀有 GO 术语漏标率高 -> 标注倾向低 -> 1/倾向 上调其正例项。
    """
    def __init__(self, freq, alpha=1.0, wmax=3.33, gamma_neg=4.0, clip=0.05):
        super().__init__()
        f = np.asarray(freq, np.float64)
        f = f / max(f.max(), 1e-12)
        w = np.power(np.maximum(f, 1e-9), -float(alpha))
        w = np.clip(w, 1.0, float(wmax))
        self.register_buffer("w", torch.from_numpy(w.astype(np.float32)))
        self.gn, self.cl, self.alpha = gamma_neg, clip, alpha

    def forward(self, lg, tg):
        p = torch.sigmoid(lg.float()).clamp(1e-6, 1 - 1e-6)
        pos = -tg * torch.log(p)
        pm = torch.clamp(p - self.cl, min=0.0)
        neg = -(1 - tg) * (pm ** self.gn) * torch.log(1 - pm)
        return (pos * self.w[None, :].to(pos.dtype) + neg).mean()


def branch_metrics(S, y, name):
    """逐标签 ROC -> 汇总 AUC / 归一化 pAUC[0,0.1] / 固定 FPR 召回 (同 I072 口径)"""
    aucs, paucs = [], []
    rec_at = {f: [] for f in FPR_GRID}
    for j in range(y.shape[1]):
        yt = y[:, j]
        if yt.min() == yt.max():
            continue
        fpr, tpr, _ = roc_curve(yt, S[:, j])
        if len(fpr) < 2:
            continue
        aucs.append(sk_auc(fpr, tpr))
        m = fpr <= 0.10
        if m.sum() >= 2:
            paucs.append(sk_auc(fpr[m], tpr[m]) / 0.10)
        else:
            paucs.append(0.0)
        for f in FPR_GRID:
            k = np.searchsorted(fpr, f, side="right") - 1
            rec_at[f].append(float(tpr[max(k, 0)]))
    out = {"name": name, "auc": float(np.mean(aucs)), "pauc": float(np.mean(paucs))}
    for f in FPR_GRID:
        out[f"r@{f}"] = float(np.mean(rec_at[f]))
    return out


def train_seed(seed, crit, Xtr, ytr, Xva, yva, tag):
    """严格对齐 I064.train_seed: 1e-3 AdamW / CosineAnnealing(35) / bs512 / 按独立 val F1 选轮"""
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    tl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=512, shuffle=True, pin_memory=True)
    vl = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva.astype(np.float32))),
                    batch_size=512, shuffle=False, pin_memory=True)
    m = MLP(Xtr.shape[1]).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35, eta_min=1e-5)
    sc = torch.amp.GradScaler("cuda")
    best_f1, best_p = -1.0, None
    for ep in range(35):
        m.train()
        for bx, by in tl:
            bx, by = bx.cuda(non_blocking=True), by.cuda(non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                l = crit(m(bx), by)
            sc.scale(l).backward(); sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            sc.step(opt); sc.update()
        sch.step()
        if (ep + 1) % 15 == 0 or ep == 34:
            m.eval(); out = []
            with torch.no_grad():
                for bx, _ in vl:
                    with torch.amp.autocast("cuda"):
                        out.append(torch.sigmoid(m(bx.cuda(non_blocking=True)).float()).cpu().numpy())
            pr = np.concatenate(out, axis=0)
            f1 = macro_f1(pr, yva, search_scut(pr, yva))
            if f1 > best_f1:
                best_f1, best_p = f1, pr
            print(f"    {tag} seed{seed} ep{ep+1} 独立F1={f1:.4f} (best {best_f1:.4f})", flush=True)
    return best_f1, best_p


def load_branches(train_df, idx_tr, idx_va, y_tr, y_va, n_va, val_pids, train_map, val_map):
    """深度 4 路 + 同源(含 2-hop) 分路, 逐行对齐 I064.main 的加载逻辑"""
    cnn_v = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn_v + 0.5 * c005_v)

    def parse_aln(tsv, qids, idmap, ev=0.001, mx=5):
        idx = np.full((len(qids), mx), -1, np.int64); bits = np.zeros((len(qids), mx), np.float32)
        qp = {q: i for i, q in enumerate(qids)}
        with open(tsv) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 5: continue
                q, t = p[0], p[1]
                if q not in qp or t not in idmap: continue
                i = qp[q]
                try: bs, e = float(p[2]), float(p[3])
                except ValueError: continue
                if e > ev: continue
                j = 0
                while j < mx and bits[i, j] > bs: j += 1
                if j < mx:
                    idx[i, j:] = np.roll(idx[i, j:], 1); bits[i, j:] = np.roll(bits[i, j:], 1)
                    idx[i, j] = idmap[t]; bits[i, j] = bs
        return idx, bits

    def agg(idx, w, ys, tk=3):
        n = idx.shape[0]; valid = idx[:, :tk] >= 0
        lab = np.zeros((n, tk, ys.shape[1]), np.float32)
        for j in range(tk): lab[:, j] = ys[idx[:, j].clip(min=0)]
        lab[~valid] = 0.0
        ww = np.maximum(w[:, :tk], 1e-7); ww[~valid] = 0.0
        return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)

    idx_v, bits_v = parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                              val_pids, train_map)
    hit, miss = idx_v[:, 0] >= 0, ~(idx_v[:, 0] >= 0)
    sim1 = agg(idx_v, bits_v, y_tr)
    v2v = {}
    with open(os.path.join(CK005A_TMP, "align_val_to_val_e0.001.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            q, t = p[0], p[1]
            if q == t or q not in val_map or t not in val_map: continue
            qi, ti = val_map[q], val_map[t]
            v2v.setdefault(qi, [])
            if len(v2v[qi]) < 5: v2v[qi].append((ti, float(p[2])))
    sim2 = sim1.copy()
    for qi in range(n_va):
        if miss[qi] and qi in v2v:
            acc = np.zeros(N_LABELS, np.float32); aw = 0.0
            for ti, bs in v2v[qi]:
                if hit[ti]: acc += sim1[ti] * bs; aw += bs
            if aw > 0: sim2[qi] = acc / aw
    kmer = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0: continue
        pre = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{pre}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_tr[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        sm = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(sm, 1e-7) ** 2
        kmer += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    sim = 0.4 * sim2 + 0.6 * kmer
    return deep4, sim


def main():
    t0 = time.time()
    print("=" * 96)
    print("  I074: Does the PU-propensity Gain Transfer to the FUSED Score?")
    print("=" * 96)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    freq = y_tr.mean(0)

    df_e = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    df_p = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
    ec = [c for c in df_e.columns if c.startswith("emb_")]
    pc = [c for c in df_p.columns if c.startswith("emb_")]
    X = np.concatenate([df_e[ec].values, df_p[pc].values], axis=1).astype(np.float32)
    print(f"  1664D features: {X.shape} | 频率范围 [{freq.min():.2e}, {freq.max():.4f}] "
          f"| 中位 {np.median(freq):.2e}")

    Xtr, Xva = X[idx_tr], X[idx_va]
    ytr32, yva32 = y_tr.astype(np.float32), y_va.astype(np.float32)

    print("\n  加载深度4路 + 同源分路 ...", flush=True)
    deep4, sim = load_branches(train_df, idx_tr, idx_va, y_tr, y_va, n_va, val_pids, train_map, val_map)

    base_multi = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    assert base_multi.shape == (n_va, N_LABELS), base_multi.shape

    def fused_eval(plm, name):
        f1, auc, mixed, ths, s = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=plm, w=W_PLM)
        hm = branch_metrics(mixed, y_va, name)
        hm.update({"f1": f1, "auc_full": auc})
        return hm

    print("\n" + "=" * 96)
    print("  [A 段] 单种子 (seed=11) 倾向权重扫描 — 判据 = 融合分 val F1")
    print("=" * 96)
    print(f"{'配置':38s} | {'独立F1':>7s} | {'融合F1':>7s} | {'dF1基':>7s} | {'pAUC':>7s} | " +
          " | ".join([f"R@F{f:<5}" for f in FPR_GRID]))
    print("-" * 96)

    cached = fused_eval(base_multi, "缓存基线 ASL 3seed")
    print(f"{'0. 缓存基线 (I064 ASL 3seed)':38s} | {'-':>7s} | {cached['f1']:7.4f} | "
          f"{cached['f1']-BASE_F1:+7.4f} | {cached['pauc']:7.4f} | " +
          " | ".join([f"{cached[f'r@{f}']:8.4f}" for f in FPR_GRID]))

    sweep = [("1.1 alpha=0.0 (ASL重训对照)", 0.0, 3.33),
             ("1.2 alpha=0.5 Wmax=3.33", 0.5, 3.33),
             ("1.3 alpha=1.0 Wmax=3.33", 1.0, 3.33),
             ("1.4 alpha=1.0 Wmax=6.0", 1.0, 6.0)]
    A = {}
    prog = {}
    for name, alpha, wmax in sweep:
        crit = ASLProp(freq, alpha=alpha, wmax=wmax).cuda()
        sf1, p = train_seed(11, crit, Xtr, ytr32, Xva, yva32, f"a{alpha}w{wmax}")
        hm = fused_eval(p, name)
        A[name] = {"alpha": alpha, "wmax": wmax, "standalone_f1": sf1, **hm}
        prog[name] = p
        print(f"{name:38s} | {sf1:7.4f} | {hm['f1']:7.4f} | {hm['f1']-BASE_F1:+7.4f} | "
              f"{hm['pauc']:7.4f} | " + " | ".join([f"{hm[f'r@{f}']:8.4f}" for f in FPR_GRID]), flush=True)

    best_name = max(A, key=lambda k: A[k]["f1"])
    best = A[best_name]
    print(f"\n  A 段最优: {best_name} -> 融合 F1 {best['f1']:.4f} "
          f"(独立 {best['standalone_f1']:.4f}, alpha={best['alpha']}, Wmax={best['wmax']})")

    print("\n" + "=" * 96)
    print(f"  [B 段] 3 种子确认 (11/22/33) | alpha={best['alpha']} Wmax={best['wmax']}")
    print("=" * 96)
    seed_probs = [prog[best_name]]
    for s in (22, 33):
        crit = ASLProp(freq, alpha=best["alpha"], wmax=best["wmax"]).cuda()
        sf1, p = train_seed(s, crit, Xtr, ytr32, Xva, yva32, f"b{best['alpha']}w{best['wmax']}")
        seed_probs.append(p)
        print(f"    seed{s} 独立F1={sf1:.4f}", flush=True)
    multi_prop = np.mean(seed_probs, axis=0)
    np.savez_compressed(os.path.join(REL_DIR, "plm1664_prop_multiseed_val.npz"), val=multi_prop)

    ms = macro_f1(multi_prop, y_va, search_scut(multi_prop, y_va))
    print(f"\n  3seed 独立融合前 F1={ms:.4f} (基线 3seed 独立 F1 见 I064)")
    new = fused_eval(multi_prop, "倾向加权 3seed")
    print(f"  倾向加权 3seed  融合F1={new['f1']:.4f} | pAUC={new['pauc']:.4f} | R@0.069={new['r@0.069']:.4f}")
    print(f"  缓存 ASL 3seed  融合F1={cached['f1']:.4f} | pAUC={cached['pauc']:.4f} | R@0.069={cached['r@0.069']:.4f}")
    print(f"  ---------------------------------------------------------------")
    print(f"  d融合F1={new['f1']-cached['f1']:+.4f} | dpAUC={new['pauc']-cached['pauc']:+.4f} | "
          f"dR@0.069={new['r@0.069']-cached['r@0.069']:+.4f}")

    verdict = ("传导成功" if new["f1"] > cached["f1"] + 0.0005 else
               "未传导 (独立增益被融合吸收)" if A[best_name]["standalone_f1"] > 0.27 else "无增益")
    print(f"\n  判定: {verdict}")

    out = {"A_sweep": A, "best_A": best_name, "cached_baseline": cached,
           "prop_3seed": new, "prop_3seed_standalone_f1": ms,
           "d_fused_f1": new["f1"] - cached["f1"],
           "d_pauc": new["pauc"] - cached["pauc"],
           "d_r069": new["r@0.069"] - cached["r@0.069"],
           "verdict": verdict, "w_plm": W_PLM, "fpr_grid": FPR_GRID}
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
