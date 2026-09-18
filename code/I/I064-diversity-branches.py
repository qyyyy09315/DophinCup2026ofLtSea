"""
I064 — 多样性优先的分支设计: 1664D 多种子 + 线性 vs RRF 融合
=============================================================
I063 教训: 加入同任务训练的 R001 特征使独立分升高 (0.2590->0.2867) 但融合分下降
          (0.4471->0.4462)，即"独立更强 != 融合更强"，多样性才是关键。
本实验对照:
  1. 1664D 单种子 (I060 现状) + 线性混合
  2. 1664D 3 种子平均 + 线性混合  (纯降方差，不引入相关特征)
  3. 1664D 单种子 + RRF 秩融合 (而非线性)
  4. 1664D 3 种子 + RRF 秩融合
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
SEEDS = [11, 22, 33]


class MLP(nn.Module):
    def __init__(self, ind, hid=1024, out=N_LABELS, dp=0.2):
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


class ASL(nn.Module):
    def forward(self, lg, tg):
        p = torch.sigmoid(lg.float()).clamp(1e-6, 1 - 1e-6)
        pos = -tg * torch.log(p)
        pm = torch.clamp(p - 0.05, min=0.0)
        neg = -(1 - tg) * (pm ** 4.0) * torch.log(1 - pm)
        return (pos + neg).mean()


def search_scut(ens, y, n=100):
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


def ranks(s): return np.argsort(np.argsort(-s, axis=1, kind="mergesort"), axis=1)
def col_norm(x):
    lo, hi = x.min(axis=0), x.max(axis=0)
    return (x-lo)/np.maximum(hi-lo, 1e-9)


def sinkhorn(K, r, c, it=100, tol=1e-5):
    u = np.ones_like(r); v = np.ones_like(c)
    for _ in range(it):
        un = r/np.maximum(K@v, 1e-12); vn = c/np.maximum(K.T@un, 1e-12)
        if np.max(np.abs(un-u)) < tol and np.max(np.abs(vn-v)) < tol: break
        u, v = un, vn
    P = u[:,None]*K*v[None,:]
    lo, hi = P.min(axis=0), P.max(axis=0)
    return (P-lo)/np.maximum(hi-lo, 1e-9)


def full_pipe(deep, sim, y_tr, y_va, n_va, use_rrf_plm=False, plm=None, w=0.25):
    """use_rrf_plm: 用 RRF 秩融合注入 PLM 分支；否则线性混合。"""
    if use_rrf_plm and plm is not None:
        d = deep + w * col_norm(1.0/(30 + ranks(plm) + 1))
    elif plm is not None:
        d = (1-w)*deep + w*plm
    else:
        d = deep
    s = d + 0.75*sim + 0.50*col_norm(1.0/(30+ranks(d)+1) + 1.0/(30+ranks(sim)+1))
    t_tmp = search_scut(s, y_va)
    row_m = np.maximum((s >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_tr.mean(0)*n_va, 1e-6); col_m *= row_m.sum()/col_m.sum()
    ker = np.exp((s - s.max(axis=1, keepdims=True))/0.20)
    p_v = sinkhorn(ker, row_m, col_m)
    mixed = 0.40*s + 0.60*p_v
    ths = search_scut(mixed, y_va)
    return macro_f1(mixed, y_va, ths), roc_auc_score(y_va, mixed, average="macro"), mixed, ths, s


def train_seed(seed, Xtr, ytr, Xva, yva, ind, tag):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    tl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=512, shuffle=True, pin_memory=True)
    vl = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                    batch_size=512, shuffle=False, pin_memory=True)
    m = MLP(ind).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35, eta_min=1e-5)
    crit = ASL(); sc = torch.amp.GradScaler("cuda")
    best_f1, best_p = -1, None
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
        if (ep+1) % 15 == 0 or ep == 34:
            m.eval(); out = []
            with torch.no_grad():
                for bx, _ in vl:
                    with torch.amp.autocast("cuda"):
                        out.append(torch.sigmoid(m(bx.cuda(non_blocking=True)).float()).cpu().numpy())
            pr = np.concatenate(out, axis=0)
            f1 = macro_f1(pr, yva, search_scut(pr, yva))
            if f1 > best_f1:
                best_f1, best_p = f1, pr
                torch.save(m.state_dict(), os.path.join(MODEL_DIR, f"{tag}_s{seed}.pt"))
            print(f"    {tag} seed{seed} ep{ep+1} F1={f1:.4f} (best {best_f1:.4f})", flush=True)
    return best_f1, best_p


def main():
    t0 = time.time()
    print("=" * 72)
    print("  I064: Diversity-First Branch Design (1664D multi-seed; linear vs RRF)")
    print("=" * 72)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}

    # 1664D 特征
    df_e = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    df_p = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
    ec = [c for c in df_e.columns if c.startswith("emb_")]
    pc = [c for c in df_p.columns if c.startswith("emb_")]
    X = np.concatenate([df_e[ec].values, df_p[pc].values], axis=1).astype(np.float32)
    print(f"  1664D features: {X.shape}")

    # 训练 3 个种子
    seed_probs = []
    for s in SEEDS:
        f1, p = train_seed(s, X[idx_tr], y_tr.astype(np.float32), X[idx_va], y_va.astype(np.float32),
                           X.shape[1], "i064_plm1664")
        seed_probs.append(p)
        print(f"  seed{s}: F1={f1:.4f}")
    multi = np.mean(seed_probs, axis=0)
    mf1 = macro_f1(multi, y_va, search_scut(multi, y_va))
    mauc = roc_auc_score(y_va, multi, average="macro")
    print(f"\n  Multi-seed 1664D standalone: F1={mf1:.4f} AUC={mauc:.4f}")
    np.savez_compressed(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"), val=multi)

    single = np.load(os.path.join(REL_DIR, "fast_plm_mlp_val_probs.npz"))["val"]

    # 深度 4 路
    cnn_v = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"]*t1 + wc["t002"]*t2 + wc["cnn"]*(0.5*cnn_v+0.5*c005_v)

    # 同源 (含 2-hop)
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
        return (lab*ww[:,:,None]).sum(1)/(ww.sum(1)[:,None]+1e-7)

    idx_v, bits_v = parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                              val_pids, train_map)
    hit = idx_v[:, 0] >= 0; miss = ~hit
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
                if hit[ti]: acc += sim1[ti]*bs; aw += bs
            if aw > 0: sim2[qi] = acc/aw
    kmer = 0.0
    for k, wt in zip([3,4,5,6,7,8], [0,0.1,0.2,0.4,0.2,0.1]):
        if wt <= 0: continue
        pre = "CK003" if k >= 7 else "CK002"
        d = np.load(os.path.join(REL_DIR, f"{pre}_neigh_k{k}.npz"))
        n = d["idx_val"].shape[0]
        lab = y_tr[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        sm = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(sm, 1e-7)**2
        kmer += wt*(lab*ww[:,:,None]).sum(1)/ww.sum(1)[:,None]
    sim = 0.4*sim2 + 0.6*kmer

    print("\n" + "=" * 72)
    print(f"{'Configuration':52s} | {'Val F1':>8s} | {'AUC':>8s} | {'vs C011a':>9s}")
    print("=" * 72)

    res = {}
    # 1. I060 现状: 单种子 + 线性 w=0.25
    for w in [0.25]:
        f1, auc, _, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=single, w=w)
        res[f"1. I060: single-seed 1664D, linear w={w:.2f}"] = f1
        print(f"{'1. I060: single-seed 1664D, linear w=%.2f'%w:52s} | {f1:8.4f} | {auc:8.4f} | {f1-0.4453:+9.4f}")

    # 2. 多种子 + 线性 (w 扫描)
    for w in [0.15, 0.20, 0.25, 0.30, 0.35]:
        f1, auc, _, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=multi, w=w)
        res[f"2. multi-seed 1664D, linear w={w:.2f}"] = f1
        print(f"{'2. multi-seed 1664D, linear w=%.2f'%w:52s} | {f1:8.4f} | {auc:8.4f} | {f1-0.4453:+9.4f}")

    # 3. 单种子 + RRF
    for w in [0.10, 0.15, 0.20, 0.30]:
        f1, auc, _, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, use_rrf_plm=True, plm=single, w=w)
        res[f"3. single-seed 1664D, RRF w={w:.2f}"] = f1
        print(f"{'3. single-seed 1664D, RRF w=%.2f'%w:52s} | {f1:8.4f} | {auc:8.4f} | {f1-0.4453:+9.4f}")

    # 4. 多种子 + RRF
    for w in [0.10, 0.15, 0.20, 0.30]:
        f1, auc, _, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, use_rrf_plm=True, plm=multi, w=w)
        res[f"4. multi-seed 1664D, RRF w={w:.2f}"] = f1
        print(f"{'4. multi-seed 1664D, RRF w=%.2f'%w:52s} | {f1:8.4f} | {auc:8.4f} | {f1-0.4453:+9.4f}")

    best_cfg = max(res, key=res.get)
    print("\n" + "=" * 72)
    print(f"  BEST: {best_cfg} -> {res[best_cfg]:.4f}")
    print("=" * 72)

    with open(os.path.join(REL_DIR, "I064_report.json"), "w", encoding="utf-8") as f:
        json.dump({"results": res, "best_config": best_cfg, "best_f1": res[best_cfg]}, f, indent=2)
    print(f"\nCompleted in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
