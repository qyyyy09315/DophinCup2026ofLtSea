"""
I080 — 只在融合分数上训低 FPR 重排头 (路线 C)
==========================================
不重训 A/B 两支. 输入 = 每格 (deep5, sim, plm, 现状融合分) 四通道,
共享小 MLP 输出新分数, 5 折 OOF.

损失: 普通 ASL 对照 + 难负例 ASL (每标签只保留分数最高的一批负例, 近似 pAUC 的 FPR 上界).
难负例比例 beta=0.07 对齐诊断点 FPR=0.069.
绝不同时对两支施加排序损失.

判据 = 接入阈值搜索后的融合 Macro F1, 辅看 R@FPR0.069.
"""
import os, io, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc as sk_auc

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
REPORT = os.path.join(REL_DIR, "I080_report.json")
N_LABELS = 500
W_PLM = 0.35
BASE_F1 = 0.4475
BETA = 0.07
N_FOLDS = 5
EPOCHS = 15
BATCH = 256
SEED = 42
FPR_GRID = [0.02, 0.05, 0.069, 0.10]

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
macro_f1, search_scut = i064.macro_f1, i064.search_scut
ranks, col_norm, sinkhorn = i064.ranks, i064.col_norm, i064.sinkhorn


class Head(nn.Module):
    def __init__(self, n_feat, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def asl(lg, tg):
    p = torch.sigmoid(lg.float()).clamp(1e-6, 1 - 1e-6)
    pos = -tg * torch.log(p)
    pm = torch.clamp(p - 0.05, min=0.0)
    neg = -(1 - tg) * (pm ** 4.0) * torch.log(1 - pm)
    return pos, neg


def loss_asl(lg, tg):
    pos, neg = asl(lg, tg)
    return (pos + neg).mean()


def loss_hardneg(lg, tg, beta=BETA):
    """每标签: 全部正例 + 分数最高的 beta 比例负例. 近似 one-way pAUC."""
    pos, neg = asl(lg, tg)
    n = tg.shape[0]
    k = max(int(n * beta), 1)
    score = lg.detach().masked_fill(tg > 0.5, -1e9)
    k = min(k, n)
    thresh = torch.topk(score, k, dim=0).values[-1]
    hard = (tg < 0.5) & (lg.detach() >= thresh)
    keep = (tg > 0.5) | hard
    return (pos + neg)[keep].mean()


def head_metrics(S, y):
    aucs, paucs = [], []
    rec = {f: [] for f in FPR_GRID}
    for j in range(y.shape[1]):
        yt = y[:, j]
        if yt.min() == yt.max():
            continue
        fpr, tpr, _ = roc_curve(yt, S[:, j])
        if len(fpr) < 2:
            continue
        aucs.append(sk_auc(fpr, tpr))
        m = fpr <= 0.10
        paucs.append(sk_auc(fpr[m], tpr[m]) / 0.10 if m.sum() >= 2 else 0.0)
        for f in FPR_GRID:
            k = np.searchsorted(fpr, f, side="right") - 1
            rec[f].append(float(tpr[max(k, 0)]))
    return {"auc": float(np.mean(aucs)), "pauc": float(np.mean(paucs)),
            **{f"r@{f}": float(np.mean(rec[f])) for f in FPR_GRID}}


def fuse(deep, sim, y_tr, y_va, n_va, plm):
    d = (1 - W_PLM) * deep + W_PLM * plm
    s = d + 0.75 * sim + 0.50 * col_norm(1.0 / (30 + ranks(d) + 1) + 1.0 / (30 + ranks(sim) + 1))
    t_tmp = search_scut(s, y_va)
    row_m = np.maximum((s >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_tr.mean(0) * n_va, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s - s.max(axis=1, keepdims=True)) / 0.20)
    p_v = sinkhorn(ker, row_m, col_m)
    mixed = 0.40 * s + 0.60 * p_v
    return mixed, d, s


def train_oof(X, y, kind):
    n, L, F = X.shape
    oof = np.zeros((n, L), np.float32)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    crit = loss_hardneg if kind == "hardneg" else loss_asl
    for fi, (tr, te) in enumerate(kf.split(np.arange(n))):
        torch.manual_seed(SEED + fi)
        m = Head(F).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X[tr]),
                                            torch.from_numpy(y[tr].astype(np.float32)))
        dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, pin_memory=True)
        for ep in range(EPOCHS):
            m.train()
            for bx, by in dl:
                bx, by = bx.cuda(non_blocking=True), by.cuda(non_blocking=True)
                opt.zero_grad()
                crit(m(bx), by).backward()
                opt.step()
            sch.step()
        m.eval()
        with torch.no_grad():
            chunks = []
            Xe = torch.from_numpy(X[te])
            for s in range(0, len(te), BATCH):
                chunks.append(torch.sigmoid(m(Xe[s:s + BATCH].cuda()).float()).cpu().numpy())
            oof[te] = np.concatenate(chunks, axis=0)
        print(f"    {kind} fold{fi + 1}/{N_FOLDS} done", flush=True)
    return oof


def load_sim(y_tr, n_va, val_pids, train_map, val_map):
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
    t0 = time.time()
    print("=" * 96)
    print("  I080: Low-FPR rerank head on frozen fusion scores")
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

    cnn = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    plm = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn + 0.5 * c005)
    sim = load_sim(y_tr, n_va, val_pids, train_map, val_map)
    mixed, deep5, s_rrf = fuse(deep4, sim, y_tr, y_va, n_va, plm)
    print(f"  baseline fused F1={macro_f1(mixed, y_va, search_scut(mixed, y_va)):.4f}")

    X = np.stack([deep5, sim, plm, mixed], axis=2).astype(np.float32)
    print(f"  X {X.shape} channels=deep5,sim,plm,fused")

    print("  OOF ASL rerank ...", flush=True)
    p_asl = train_oof(X, y_va, "asl")
    print("  OOF hard-neg (beta=0.07) rerank ...", flush=True)
    p_hn = train_oof(X, y_va, "hardneg")
    mix_asl = 0.7 * mixed + 0.3 * p_asl
    mix_hn = 0.7 * mixed + 0.3 * p_hn

    variants = [("V0 现状融合", mixed),
                ("V1 OOF-ASL 替换", p_asl),
                ("V2 OOF-难负例 替换", p_hn),
                ("V3 0.7现状+0.3 ASL", mix_asl),
                ("V4 0.7现状+0.3 难负例", mix_hn)]
    print("\n" + "=" * 96)
    print(f"{'配置':24s} | {'F1':>7s} | {'AUC':>7s} | {'pAUC':>7s} | {'R@0.069':>8s} | {'dF1':>8s}")
    print("-" * 96)
    res = {}
    for name, S in variants:
        f1 = macro_f1(S, y_va, search_scut(S, y_va))
        hm = head_metrics(S, y_va)
        res[name] = {"f1": f1, **hm, "d": f1 - BASE_F1}
        print(f"{name:24s} | {f1:7.4f} | {hm['auc']:7.4f} | {hm['pauc']:7.4f} | "
              f"{hm['r@0.069']:8.4f} | {f1 - BASE_F1:+8.4f}", flush=True)

    best = max(res, key=lambda k: res[k]["f1"])
    print(f"\n  最优: {best} F1={res[best]['f1']:.4f}")
    verdict = ("重排头有效" if res[best]["f1"] > BASE_F1 + 0.0005
               else "重排头未超过现状, 关闭路线 C")
    print(f"  判定: {verdict}")
    np.savez_compressed(os.path.join(REL_DIR, "i080_oof_hardneg.npz"), val=p_hn)
    out = {"results": res, "best": best, "verdict": verdict, "beta": BETA}
    with io.open(REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
