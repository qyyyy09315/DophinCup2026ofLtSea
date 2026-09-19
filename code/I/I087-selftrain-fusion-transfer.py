"""
I087 — 自训练 PLM 头的融合传导判定 (I074 式检验)
==================================================
I086 结论: 单轮护栏自训练 (factor=0.5 硬标签 λ=1.0) 使 PLM 头独立 F1
           0.2626 -> 0.2681 (+0.0054), 真值上限 +0.0987。
本项目铁律: 独立变强常伴随融合变差 (I063/I074/I077)。必须测传导。
流程:
  1. teacher = train(80%) 模型 (seed 42)
  2. 给 val (模拟测试池) 打伪标签: 每标签取 top (0.5 x 流行率) 分位, 硬标签, 仅正例
  3. 重训于 train + 伪标签 val
  4. 融合: full_pipe(deep4, sim, plm=自训练头) -> 融合 F1 vs 基线 0.4475
  5. 过线 (+0.001) 则 3 种子确认
"""
import os, io, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
REPORT = os.path.join(REL_DIR, "I087_report.json")
N_LABELS = 500
W_PLM = 0.35
BASE = 0.4475
PASS = 0.4485
EPOCHS = 35
FACTOR = 0.5
LAM = 1.0
SEEDS = [42]
SEEDS_CONF = [11, 22, 33]

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
MLP, macro_f1, search_scut, full_pipe = i064.MLP, i064.macro_f1, i064.search_scut, i064.full_pipe


class ASL(nn.Module):
    def forward(self, lg, tg):
        p = torch.sigmoid(lg.float()).clamp(1e-6, 1 - 1e-6)
        pos = -tg * torch.log(p)
        pm = torch.clamp(p - 0.05, min=0.0)
        neg = -(1 - tg) * (pm ** 4.0) * torch.log(1 - pm)
        return pos + neg


def train_predict(seed, Xall, T, M, w, n_eval, tag):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    tl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xall), torch.from_numpy(T),
                                       torch.from_numpy(M), torch.from_numpy(w)),
        batch_size=512, shuffle=True, pin_memory=True)
    m = MLP(Xall.shape[1]).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    crit = ASL(); sc = torch.amp.GradScaler("cuda")
    for ep in tqdm(range(EPOCHS), desc=f"  {tag} s{seed}", ncols=90):
        m.train()
        for bx, bt, bm, bw in tl:
            bx, bt, bm, bw = (x.cuda(non_blocking=True) for x in (bx, bt, bm, bw))
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                l = crit(m(bx), bt) * bm * bw[:, None]
                l = l.sum() / bm.sum().clamp(min=1.0)
            sc.scale(l).backward(); sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            sc.step(opt); sc.update()
        sch.step()
    m.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(Xall), 1024):
            bx = torch.from_numpy(Xall[s:s + 1024]).cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                out.append(torch.sigmoid(m(bx).float()).cpu().numpy())
    return np.concatenate(out, axis=0)[-n_eval:]


def parse_aln(tsv, qids, idmap, ev=0.001, mx=5):
    idx = np.full((len(qids), mx), -1, np.int64)
    bits = np.zeros((len(qids), mx), np.float32)
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
    n = idx.shape[0]
    valid = idx[:, :tk] >= 0
    lab = np.zeros((n, tk, ys.shape[1]), np.float32)
    for j in range(tk): lab[:, j] = ys[idx[:, j].clip(min=0)]
    lab[~valid] = 0.0
    ww = np.maximum(w[:, :tk], 1e-7); ww[~valid] = 0.0
    return (lab * ww[:, :, None]).sum(1) / (ww.sum(1)[:, None] + 1e-7)


def build_sim(y_tr, n_va, val_pids, train_map, val_map):
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
    return 0.4 * sim2 + 0.6 * kmer


def main():
    t0 = time.time()
    print("=" * 96)
    print("  I087: Does self-trained PLM head transfer to fusion?")
    print("=" * 96)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_tr, n_va = len(idx_tr), len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    prev = y_tr.mean(0)

    df_e = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    df_p = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
    ec = [c for c in df_e.columns if c.startswith("emb_")]
    pc = [c for c in df_p.columns if c.startswith("emb_")]
    X = np.concatenate([df_e[ec].values, df_p[pc].values], axis=1).astype(np.float32)
    Xtr, Xva = X[idx_tr], X[idx_va]
    Xall = np.concatenate([Xtr, Xva], axis=0)

    ones_tr = np.ones((n_tr, N_LABELS), np.float32)
    zeros_va = np.zeros((n_va, N_LABELS), np.float32)
    w_all = np.ones(n_tr + n_va, np.float32)

    # ---- teacher: train-only ----
    T0 = np.concatenate([y_tr.astype(np.float32), zeros_va], axis=0)
    M0 = np.concatenate([ones_tr, zeros_va], axis=0)
    P0 = train_predict(SEEDS[0], Xall, T0, M0, w_all, n_va, "teacher")
    f0 = macro_f1(P0, y_va, search_scut(P0, y_va))
    print(f"  teacher 独立 F1 = {f0:.4f}")

    # ---- 伪标签 (I086 最优配方) ----
    Tv = np.zeros((n_va, N_LABELS), np.float32)
    prec = []
    for j in range(N_LABELS):
        k = min(max(int(prev[j] * FACTOR * n_va), 1), n_va - 1)
        order = np.argsort(-P0[:, j])[:k]
        Tv[order, j] = 1.0
        prec.append(float(y_va[order, j].mean()))
    print(f"  伪正例精度 = {float(np.mean(prec)):.4f} (factor={FACTOR})")

    # ---- student: train + 伪标签 val ----
    T1 = np.concatenate([y_tr.astype(np.float32), Tv], axis=0)
    M1 = np.concatenate([ones_tr, (Tv > 0).astype(np.float32)], axis=0)
    w1 = np.concatenate([np.ones(n_tr), np.full(n_va, LAM)]).astype(np.float32)
    P1 = train_predict(SEEDS[0], Xall, T1, M1, w1, n_va, "student")
    f1s = macro_f1(P1, y_va, search_scut(P1, y_va))
    print(f"  student 独立 F1 = {f1s:.4f} ({f1s - f0:+.4f})")

    # ---- 融合传导 ----
    print("\n  重建融合 ...", flush=True)
    cnn = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1p = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2p = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1p + wc["t002"] * t2p + wc["cnn"] * (0.5 * cnn + 0.5 * c005)
    sim = build_sim(y_tr, n_va, val_pids, train_map, val_map)

    fb, aub, _, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=P0, w=W_PLM)
    ff, auf, _, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=P1, w=W_PLM)
    print(f"  teacher 融合 F1 = {fb:.4f} | student 融合 F1 = {ff:.4f} ({ff - fb:+.4f})")

    out = {"teacher_standalone": f0, "student_standalone": f1s,
           "pseudo_precision": float(np.mean(prec)),
           "teacher_fused": fb, "student_fused": ff, "d_fused": ff - fb,
           "factor": FACTOR, "lam": LAM}
    if ff >= PASS and ff > fb + 0.0005:
        print("\n  过通过线, 3 种子确认 ...")
        probs_t, probs_s = [], []
        for s in SEEDS_CONF:
            pt = train_predict(s, Xall, T0, M0, w_all, n_va, "teacher3")
            ps = train_predict(s, Xall, T1, M1, w1, n_va, "student3")
            probs_t.append(pt); probs_s.append(ps)
        P0m, P1m = np.mean(probs_t, axis=0), np.mean(probs_s, axis=0)
        fbm, _, _, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=P0m, w=W_PLM)
        ffm, _, _, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=P1m, w=W_PLM)
        out.update({"teacher3_fused": fbm, "student3_fused": ffm, "d_fused_3seed": ffm - fbm})
        print(f"  3seed: teacher 融合 {fbm:.4f} | student 融合 {ffm:.4f} ({ffm - fbm:+.4f})")
        verdict = ("传导成功" if ffm > fbm + 0.0005 else "独立增益不传导, 关闭")
        np.savez_compressed(os.path.join(REL_DIR, "i087_selftrain_multiseed_val.npz"), val=P1m)
    else:
        verdict = ("传导失败" if ff <= fb - 0.0005 else
                   f"未过通过线 {PASS} (d={ff - fb:+.4f}), 关闭")
        print(f"\n  判定: {verdict}")

    out["verdict"] = verdict
    with io.open(REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Report: {REPORT}")
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
