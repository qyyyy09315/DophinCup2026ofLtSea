"""
I078 — 同源支可学习条件化聚合（现有 top-5 概念验证）
====================================================
动机: 现有 sim = 0.4*sim2 + 0.6*kmer 是人为固定的聚合器。
ProtEx / ProtIR 把「查询 × 标签 × 命中」的聚合换成学习型打分器。
本实验用已有 top-5 DIAMOND 命中做概念验证（top-100 需用户手动重跑，不在本脚本）。

设计: 对每个 (样本, 标签) 构造同源证据特征, 共享小 MLP 打分,
5 折 OOF 避免用验证标签训练再在同一验证集上评价。
判据 = 接入 I064 full_pipe 后的融合 Macro F1 (w_plm=0.35)。

变体:
  V0 现状 0.4*sim2+0.6*kmer
  V1 线性打分器 (共享 Linear)
  V2 两层 MLP
  V3 0.7*现状 + 0.3*V2  (保守混合, 防止替换毁掉已验证的 sim)
"""
import os, io, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
REPORT = os.path.join(REL_DIR, "I078_report.json")
N_LABELS = 500
W_PLM = 0.35
BASE_F1 = 0.4475
N_FOLDS = 5
EPOCHS = 20
BATCH = 512
SEED = 42

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe


class CellScorer(nn.Module):
    def __init__(self, n_feat, hidden=0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(n_feat, hidden), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Linear(n_feat, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ASL(nn.Module):
    def forward(self, lg, tg):
        p = torch.sigmoid(lg.float()).clamp(1e-6, 1 - 1e-6)
        pos = -tg * torch.log(p)
        pm = torch.clamp(p - 0.05, min=0.0)
        neg = -(1 - tg) * (pm ** 4.0) * torch.log(1 - pm)
        return (pos + neg).mean()


def parse_aln(tsv, qids, idmap, ev=0.001, mx=5):
    idx = np.full((len(qids), mx), -1, np.int64)
    bits = np.zeros((len(qids), mx), np.float32)
    qp = {q: i for i, q in enumerate(qids)}
    with open(tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
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


def hit_features(idx, bits, y_src, freq):
    """(n, L, 4): pos_frac, wpos, maxbits_pos, npos_norm"""
    n, mx = idx.shape
    L = y_src.shape[1]
    valid = idx >= 0
    lab = np.zeros((n, mx, L), np.float32)
    for k in range(mx):
        lab[:, k] = y_src[idx[:, k].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(bits, 0.0)
    ww[~valid] = 0.0
    n_valid = valid.sum(1, keepdims=True).astype(np.float32)
    pos_frac = lab.sum(1) / np.maximum(n_valid, 1.0)
    wpos = (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)
    maxbits = (lab * ww[:, :, None]).max(1) / 100.0
    npos = lab.sum(1) / float(mx)
    return np.stack([pos_frac, wpos, maxbits, npos], axis=2).astype(np.float32)


def train_oof(X, y, hidden):
    """X: (n, L, F)  y: (n, L)  -> OOF sigmoid scores (n, L)"""
    n, L, F = X.shape
    oof = np.zeros((n, L), np.float32)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    crit = ASL()
    for fi, (tr, te) in enumerate(kf.split(np.arange(n))):
        torch.manual_seed(SEED + fi)
        m = CellScorer(F, hidden=hidden).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
        Xt = torch.from_numpy(X[tr])
        yt = torch.from_numpy(y[tr].astype(np.float32))
        ds = torch.utils.data.TensorDataset(Xt, yt)
        dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, pin_memory=True)
        for ep in range(EPOCHS):
            m.train()
            for bx, by in dl:
                bx, by = bx.cuda(non_blocking=True), by.cuda(non_blocking=True)
                opt.zero_grad()
                loss = crit(m(bx), by)
                loss.backward()
                opt.step()
            sch.step()
        m.eval()
        with torch.no_grad():
            chunks = []
            Xe = torch.from_numpy(X[te])
            for s in range(0, len(te), BATCH):
                bx = Xe[s:s + BATCH].cuda()
                chunks.append(torch.sigmoid(m(bx).float()).cpu().numpy())
            oof[te] = np.concatenate(chunks, axis=0)
        print(f"    fold{fi + 1}/{N_FOLDS} hidden={hidden} done", flush=True)
    return oof


def main():
    t0 = time.time()
    print("=" * 88)
    print("  I078: Learnable homology aggregator (top-5 concept check)")
    print("=" * 88)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    freq = y_tr.mean(0).astype(np.float32)

    cnn = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    plm = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn + 0.5 * c005)

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
    sim_old = 0.4 * sim2 + 0.6 * kmer
    print(f"  hit rate={hit.mean():.3f} | sim_old shape={sim_old.shape}")

    hf = hit_features(idx_v, bits_v, y_tr, freq)
    hit_flag = hit.astype(np.float32)[:, None, None].repeat(N_LABELS, axis=1)
    freq_b = np.broadcast_to(freq[None, :, None], (n_va, N_LABELS, 1)).copy()
    X = np.concatenate([
        sim2[:, :, None], kmer[:, :, None], hf, freq_b, hit_flag,
    ], axis=2).astype(np.float32)
    print(f"  features X {X.shape} (sim2, kmer, pos_frac, wpos, maxbits, npos, freq, hit)")

    print("\n  5-fold OOF linear ...", flush=True)
    s_lin = train_oof(X, y_va, hidden=0)
    print("  5-fold OOF MLP-64 ...", flush=True)
    s_mlp = train_oof(X, y_va, hidden=64)
    np.savez_compressed(os.path.join(REL_DIR, "i078_oof_lin.npz"), val=s_lin)
    np.savez_compressed(os.path.join(REL_DIR, "i078_oof_mlp.npz"), val=s_mlp)

    mix = 0.7 * sim_old + 0.3 * s_mlp
    variants = [
        ("V0 现状固定加权", sim_old),
        ("V1 OOF线性打分器", s_lin),
        ("V2 OOF两层MLP", s_mlp),
        ("V3 保守混合 0.7现状+0.3MLP", mix),
    ]
    print("\n" + "=" * 88)
    print(f"{'配置':32s} | {'独立F1':>8s} | {'融合F1':>8s} | {'AUC':>8s} | {'d融合':>8s}")
    print("-" * 88)
    res = {}
    for name, sim in variants:
        sf1 = macro_f1(sim, y_va, search_scut(sim, y_va))
        f1, auc, _, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=plm, w=W_PLM)
        res[name] = {"standalone_f1": sf1, "fused_f1": f1, "auc": auc, "d": f1 - BASE_F1}
        print(f"{name:32s} | {sf1:8.4f} | {f1:8.4f} | {auc:8.4f} | {f1 - BASE_F1:+8.4f}", flush=True)

    best = max(res, key=lambda k: res[k]["fused_f1"])
    print(f"\n  最优: {best} -> 融合 {res[best]['fused_f1']:.4f}")
    verdict = ("传导成功" if res[best]["fused_f1"] > BASE_F1 + 0.0005
               else "未超过现状, 固定加权已够用 (top-5 信息量不足或聚合器已饱和)")
    print(f"  判定: {verdict}")

    out = {"results": res, "best": best, "verdict": verdict, "w_plm": W_PLM,
           "n_feat": int(X.shape[-1]), "epochs": EPOCHS, "folds": N_FOLDS}
    with io.open(REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
