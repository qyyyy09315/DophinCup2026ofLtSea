"""
I081 — 全量重训 PLM-MLP 头并重建测试提交
==========================================
背景: I065 提交中 deep4 与 PLM-MLP 都只用 80% 训练数据训练,
      而同源支的 test 邻居空间早已用全量 113,796 (TCK002a 已吸收)。
      提交侧唯一没吃的数据增益 = PLM-MLP 用 100% 训练重训。
      配方严格锁 I064 (35 epoch / ASL / AdamW 1e-3 / cosine / bs512 / seeds 11,22,33),
      I064 的最优轮次本来就是最后一轮, 无需早停。
      deep4 不重训 (耗时长且 tck002 权重与 80% 模型耦合, 不动)。

判据: 本地不可验证 (val 无法度量全量重训), 属提交质量改动;
      val 侧链路必须逐位复现 0.4475 作为无偏校验。
输出: submit/submission_I081_fulltrain.csv (0/1) + 桌面副本
      与 I065 提交的逐格差异统计。
"""
import os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = r"F:\FOR_PYTHON\project\26海豚杯"
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
W_PLM = 0.35
SEEDS = [11, 22, 33]
EPOCHS = 35

i064 = SourceFileLoader("i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")).load_module()
MLP, macro_f1, search_scut = i064.MLP, i064.macro_f1, i064.search_scut
ranks, col_norm, sinkhorn = i064.ranks, i064.col_norm, i064.sinkhorn


class ASL(nn.Module):
    def forward(self, lg, tg):
        p = torch.sigmoid(lg.float()).clamp(1e-6, 1 - 1e-6)
        pos = -tg * torch.log(p)
        pm = torch.clamp(p - 0.05, min=0.0)
        neg = -(1 - tg) * (pm ** 4.0) * torch.log(1 - pm)
        return (pos + neg).mean()


def train_full(seed, X, y):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    tl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))),
        batch_size=512, shuffle=True, pin_memory=True)
    m = MLP(X.shape[1]).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    crit = ASL(); sc = torch.amp.GradScaler("cuda")
    for ep in tqdm(range(EPOCHS), desc=f"  train seed{seed}", ncols=90):
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
    return m


def predict(m, X, desc="predict"):
    m.eval(); out = []
    with torch.no_grad():
        for s in tqdm(range(0, len(X), 512), desc=desc, ncols=90, leave=False):
            bx = torch.from_numpy(X[s:s + 512]).cuda(non_blocking=True)
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


def transitive_2hop(sim1, hit, miss, nbr, n):
    out = sim1.copy()
    for qi in range(n):
        if miss[qi] and qi in nbr:
            acc = np.zeros(N_LABELS, np.float32); aw = 0.0
            for ti, bs in nbr[qi]:
                if hit[ti]:
                    acc += sim1[ti] * bs; aw += bs
            if aw > 0: out[qi] = acc / aw
    return out


def build_kmers(y_src, split):
    s = 0.0
    for k, wt in zip([3, 4, 5, 6, 7, 8], [0, 0.1, 0.2, 0.4, 0.2, 0.1]):
        if wt <= 0: continue
        pre = "CK003" if k >= 7 else "CK002"
        if split == "val":
            d = np.load(os.path.join(REL_DIR, f"{pre}_neigh_k{k}.npz"))
            idx = d["idx_val"][:, :5]; sm = d["sim_val"][:, :5].astype(np.float32)
        else:
            f = "TCK002a_neigh_full.npz" if k == 6 else f"CK004_neigh_k{k}_full.npz"
            d = np.load(os.path.join(REL_DIR, f))
            idx = d["idx_test"][:, :5]; sm = d["sim_test"][:, :5].astype(np.float32)
        n = idx.shape[0]
        lab = y_src[idx.ravel()].reshape(n, 5, -1).astype(np.float32)
        ww = np.maximum(sm, 1e-7) ** 2
        s += wt * (lab * ww[:, :, None]).sum(1) / ww.sum(1)[:, None]
    return s


def rrf_terms(a, b, k=30):
    return 1.0 / (k + ranks(a) + 1) + 1.0 / (k + ranks(b) + 1)


def fuse(s_deep, s_sim, y_src, n_rows, t_tmp):
    s = s_deep + 0.75 * s_sim + 0.50 * col_norm(rrf_terms(s_deep, s_sim, 30))
    row_m = np.maximum((s >= t_tmp).sum(1).astype(np.float64), 1.0)
    col_m = np.maximum(y_src.mean(0) * n_rows, 1e-6)
    col_m *= row_m.sum() / col_m.sum()
    ker = np.exp((s - s.max(axis=1, keepdims=True)) / 0.20)
    p = sinkhorn(ker, row_m, col_m)
    return 0.40 * s + 0.60 * p


def main():
    t0 = time.time()
    print("=" * 84)
    print("  I081: Full-train refit of PLM-MLP head + submission rebuild")
    print("=" * 84)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    n_all = len(train_df)
    idx_tr, idx_va = train_test_split(np.arange(n_all), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}
    full_map = {p: i for i, p in enumerate(train_df["protein_id"].values)}
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]

    # ---- 全量 1664D 特征 ----
    df_e = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    df_p = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
    ec = [c for c in df_e.columns if c.startswith("emb_")]
    pc = [c for c in df_p.columns if c.startswith("emb_")]
    X_all = np.concatenate([df_e[ec].values, df_p[pc].values], axis=1).astype(np.float32)
    assert np.array_equal(df_e["protein_id"].values, train_df["protein_id"].values)
    print(f"  全量特征 {X_all.shape}")

    # ---- 测试特征 ----
    df_eT = pd.read_parquet(os.path.join(REL_DIR, "test_esm2_embeddings.parquet"))
    df_pT = pd.read_parquet(os.path.join(REL_DIR, "test_embeddings.parquet"))
    ecT = [c for c in df_eT.columns if c.startswith("emb_")]
    pcT = [c for c in df_pT.columns if c.startswith("emb_")]
    X_test = np.concatenate([df_eT[ecT].values, df_pT[pcT].values], axis=1).astype(np.float32)
    assert np.array_equal(df_eT["protein_id"].values, test_df["protein_id"].values)
    n_te = len(test_df)

    # ---- 3 种子全量重训, 训完即预测 test ----
    probs_t = []
    for s in SEEDS:
        m = train_full(s, X_all, y_full)
        probs_t.append(predict(m, X_test, desc=f"predict seed{s}"))
        torch.cuda.empty_cache()
    multi_t = np.mean(probs_t, axis=0)
    np.savez_compressed(os.path.join(REL_DIR, "plm1664_fulltrain_test.npz"), test=multi_t)
    print(f"  全量重训 PLM test 均值分数已存 (与 80% 版差异格: "
          f"{int((np.load(os.path.join(REL_DIR, 'plm1664_multiseed_test.npz'))['test'] != multi_t).sum())})")

    # ---- val 链路校验 (80% PLM 旧缓存, 必须逐位复现 0.4475) ----
    print("\n  [校验] val 融合链 ...")
    cnn_v = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    c005_v = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1v = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2v = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    deep4_v = wc["t001a"] * t1v + wc["t002"] * t2v + wc["cnn"] * (0.5 * cnn_v + 0.5 * c005_v)
    multi_v = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    deep5_v = (1 - W_PLM) * deep4_v + W_PLM * multi_v
    idx_v, bits_v = parse_aln(os.path.join(CK005A_TMP, "align_val_very-sensitive_e0.001.tsv"),
                              val_pids, train_map)
    hit_v = idx_v[:, 0] >= 0; miss_v = ~hit_v
    sim1_v = agg(idx_v, bits_v, y_tr)
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
    sim2_v = transitive_2hop(sim1_v, hit_v, miss_v, v2v, n_va)
    sim_v = 0.4 * sim2_v + 0.6 * build_kmers(y_tr, "val")
    s_v = deep5_v + 0.75 * sim_v + 0.50 * col_norm(rrf_terms(deep5_v, sim_v, 30))
    t_tmp = search_scut(s_v, y_va)
    mixed_v = fuse(deep5_v, sim_v, y_tr, n_va, t_tmp)
    ths = search_scut(mixed_v, y_va)
    val_f1 = macro_f1(mixed_v, y_va, ths)
    print(f"  val F1 = {val_f1:.4f} (必须 0.4475)")
    assert abs(val_f1 - 0.4475) < 0.0005, f"val 链路漂移: {val_f1}"

    # ---- 测试集融合 (PLM 换全量重训版) ----
    print("\n  [提交] test 融合 ...")
    c005_t = np.load(os.path.join(REL_DIR, "C005_test_probs.npz"))["test"]
    cnn_t = np.load(os.path.join(REL_DIR, "TC003_test_probs.npz"))["cnn"]
    t1t = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_test"]
    t2t = np.load(os.path.join(REL_DIR, "T002_test_probs.npz"))["t002"]
    deep4_t = wc["t001a"] * t1t + wc["t002"] * t2t + wc["cnn"] * (0.5 * cnn_t + 0.5 * c005_t)
    deep5_t = (1 - W_PLM) * deep4_t + W_PLM * multi_t

    idx_t, bits_t = parse_aln(os.path.join(CK005A_TMP, "align_test_very-sensitive_e0.001.tsv"),
                              test_df["protein_id"].values, full_map)
    hit_t = idx_t[:, 0] >= 0; miss_t = ~hit_t
    sim1_t = agg(idx_t, bits_t, y_full)
    test_map = {p: i for i, p in enumerate(test_df["protein_id"].values)}
    te2te = {}
    with open(os.path.join(CK005A_TMP, "align_test_to_test_e0.001.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5: continue
            q, t = p[0], p[1]
            if q == t or q not in test_map or t not in test_map: continue
            qi, ti = test_map[q], test_map[t]
            te2te.setdefault(qi, [])
            if len(te2te[qi]) < 5: te2te[qi].append((ti, float(p[2])))
    sim2_t = transitive_2hop(sim1_t, hit_t, miss_t, te2te, n_te)
    n_rescued = int(np.sum(np.any(sim2_t != sim1_t, axis=1)))
    print(f"  test 2-hop 拯救孤儿 {n_rescued} / {int(miss_t.sum())}")
    sim_t = 0.4 * sim2_t + 0.6 * build_kmers(y_full, "test")

    mixed_t = fuse(deep5_t, sim_t, y_full, n_te, t_tmp)
    y_pred = (mixed_t >= ths).astype(np.int8)

    # 与 I065 提交的逐格差异
    old_path = os.path.join(SUB_DIR, "submission_I065_ensemble.csv")
    if os.path.exists(old_path):
        old = pd.read_csv(old_path)
        ov = old[label_cols].values.astype(np.int8)
        diff = (ov != y_pred).sum()
        turn_on = int(((ov == 0) & (y_pred == 1)).sum())
        turn_off = int(((ov == 1) & (y_pred == 0)).sum())
        print(f"  与 I065 提交差异: {int(diff)} 格 (开 {turn_on} / 关 {turn_off}, "
              f"占 {diff / ov.size * 100:.2f}%)")

    sub = pd.concat([test_df[["protein_id"]], pd.DataFrame(y_pred, columns=label_cols)], axis=1)
    out = os.path.join(SUB_DIR, "submission_I081_fulltrain.csv")
    sub.to_csv(out, index=False)
    mb = os.path.getsize(out) / 1e6
    print(f"  Written: {out} ({mb:.1f} MB, 值域校验: "
          f"{sorted(set(np.unique(y_pred)))})")
    try:
        import shutil
        shutil.copy(out, r"C:\Users\YKSHb\Desktop\submission_I081_fulltrain.csv")
        print("  Desktop: submission_I081_fulltrain.csv")
    except Exception as e:
        print(f"  桌面复制失败: {e}")
    print(f"  预测基数均值: {y_pred.sum(1).mean():.2f} (train 真值 25.42) | 总正格 {y_pred.sum():,d}")

    rep = {"val_f1_check": val_f1, "w_plm": W_PLM, "seeds": SEEDS,
           "test_2hop_rescued": n_rescued, "size_mb": mb,
           "pred_card_mean": float(y_pred.sum(1).mean()), "total_pos": int(y_pred.sum())}
    with Path(os.path.join(REL_DIR, "I081_report.json")).open("w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"\nCompleted in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
