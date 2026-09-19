"""
I084 — ESM2-650M 冻结嵌入评估 (前置: 用户完成 I083 提取)
==========================================================
输入: rel/esm2_650m_embeddings.npz (train_last/train_last3 + test 同构)
对照 (单种子 42 筛选, 判据 = 融合 Macro F1):
  base1664  现状 3 种子缓存 (参照, 不重训)
  s_1280    650M 末层单独
  s_2944    640(150M)+1024(ProtBERT)+1280(650M末层)
  s_5504    640+1024+3840(650M末三层拼接, EZpred +2% 依据)
通过线: 融合 F1 >= 0.4485 (+0.001)。过线者用 3 种子(11/22/33)确认后出提交。
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
EMB = os.path.join(REL_DIR, "esm2_650m_embeddings.npz")
REPORT = os.path.join(REL_DIR, "I084_report.json")
N_LABELS = 500
W_PLM = 0.35
BASE = 0.4475
PASS = 0.4485
EPOCHS = 35
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
        return (pos + neg).mean()


def train_predict(seed, Xtr, ytr, Xva, tag):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    tl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))),
        batch_size=512, shuffle=True, pin_memory=True)
    m = MLP(Xtr.shape[1]).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    crit = ASL(); sc = torch.amp.GradScaler("cuda")
    for ep in tqdm(range(EPOCHS), desc=f"  {tag} s{seed}", ncols=90):
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
    m.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(Xva), 1024):
            bx = torch.from_numpy(Xva[s:s + 1024]).cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                out.append(torch.sigmoid(m(bx).float()).cpu().numpy())
    return np.concatenate(out, axis=0)


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
    print("  I084: ESM2-650M frozen embedding evaluation")
    print("=" * 96)
    if not os.path.exists(EMB):
        print(f"  缺少 {EMB} —— 先跑 I083 提取 (见待办命令)。")
        return

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}

    z = np.load(EMB)
    pid_all = z["train_protein_id"]
    pos = {p: i for i, p in enumerate(pid_all)}
    rows_tr = np.array([pos[p] for p in train_df["protein_id"].values[idx_tr]])
    rows_va = np.array([pos[p] for p in val_pids])
    last_tr, last_va = z["train_last"][rows_tr], z["train_last"][rows_va]
    l3_tr, l3_va = z["train_last3"][rows_tr], z["train_last3"][rows_va]

    df_e = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    df_p = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
    ec = [c for c in df_e.columns if c.startswith("emb_")]
    pc = [c for c in df_p.columns if c.startswith("emb_")]
    X640_1024 = np.concatenate([df_e[ec].values, df_p[pc].values], axis=1).astype(np.float32)
    base_tr, base_va = X640_1024[idx_tr], X640_1024[idx_va]

    cnn = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    plm_base = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn + 0.5 * c005)
    sim = build_sim(y_tr, n_va, val_pids, train_map, val_map)

    configs = [
        ("s_1280", last_tr, last_va),
        ("s_2944", np.concatenate([base_tr, last_tr], axis=1), np.concatenate([base_va, last_va], axis=1)),
        ("s_5504", np.concatenate([base_tr, l3_tr], axis=1), np.concatenate([base_va, l3_va], axis=1)),
    ]
    res = {}
    for name, Xtr, Xva in configs:
        P = train_predict(SEEDS[0], Xtr, y_tr, Xva, name)
        sf1 = macro_f1(P, y_va, search_scut(P, y_va))
        f1, auc, _, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=P, w=W_PLM)
        res[name] = {"standalone_f1": sf1, "fused_f1": f1, "auc": auc, "seed": SEEDS[0]}
        print(f"  {name:10s} 独立 {sf1:.4f} | 融合 {f1:.4f} ({f1 - BASE:+.4f}) | AUC {auc:.4f}", flush=True)
        np.savez_compressed(os.path.join(REL_DIR, f"i084_{name}_val.npz"), val=P)

    best = max(res, key=lambda k: res[k]["fused_f1"])
    print(f"\n  单种子最优: {best} 融合 {res[best]['fused_f1']:.4f}")
    if res[best]["fused_f1"] < PASS:
        verdict = f"未过通过线 {PASS}, 关闭 650M 冻结嵌入路线"
        print(f"  判定: {verdict}")
    else:
        print(f"  过线, 3 种子确认 {best} ...")
        name = best
        if name == "s_1280":
            Xtr_c, Xva_c = last_tr, last_va
        elif name == "s_2944":
            Xtr_c, Xva_c = np.concatenate([base_tr, last_tr], axis=1), np.concatenate([base_va, last_va], axis=1)
        else:
            Xtr_c, Xva_c = np.concatenate([base_tr, l3_tr], axis=1), np.concatenate([base_va, l3_va], axis=1)
        probs = [train_predict(s, Xtr_c, y_tr, Xva_c, name) for s in SEEDS_CONF]
        Pm = np.mean(probs, axis=0)
        msf1 = macro_f1(Pm, y_va, search_scut(Pm, y_va))
        f1m, aucm, _, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=Pm, w=W_PLM)
        res[name + "_3seed"] = {"standalone_f1": msf1, "fused_f1": f1m, "auc": aucm}
        print(f"  {name} 3seed 独立 {msf1:.4f} | 融合 {f1m:.4f} ({f1m - BASE:+.4f})")
        verdict = (f"3seed 融合 {f1m:.4f}, {'过线, 建议出提交' if f1m >= PASS else '回退, 未过线'}")
        np.savez_compressed(os.path.join(REL_DIR, f"i084_{name}_multiseed_val.npz"), val=Pm)
        print(f"  判定: {verdict}")

    out = {"results": res, "best_single": best, "verdict": verdict,
           "pass_line": PASS, "base": BASE}
    with io.open(REPORT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {REPORT}")
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
