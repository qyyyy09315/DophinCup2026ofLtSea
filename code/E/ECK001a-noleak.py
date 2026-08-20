"""
ECK001a-noleak — ProstT5 邻居检索去泄漏重评（ECK001 衍生）
=========================================================
问题:
  ECK001/E003 的 val 检索库用了全部短序列 short_tr（含验证集），
  短 val 查询会 top-1 命中自己（余弦≈1），把自身标签当邻居标签。
  这能同时解释：本地 F1 0.4558 虚高、提交 43.24 低于 CK005 的 44.84。
方案:
  val 库改为「训练集短序列 only」（idx_train ∩ ≤256aa），与 CK/DIAMOND 防泄漏一致。
  test 库仍用全量 train 短序列（测试集不在库内，无泄漏）。
  其余融合/四路集成/100 档阈值与 ECK001 相同。
用法:
  python code/E/ECK001a-noleak.py [--gen-sub]
输出:
  rel/ECK001a_report.json
  submit/submission_ECK001a_noleak.csv（仅当无泄漏 val 超过 CK005）
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SRC = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("eck001", os.path.join(SRC, "ECK001-fusion.py"))
eck = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eck)

PROJECT_ROOT = eck.PROJECT_ROOT
REL_DIR = eck.REL_DIR
SUB_DIR = eck.SUB_DIR
MAX_AA = eck.MAX_AA
TOP_K = eck.TOP_K
ALIGN_W = eck.ALIGN_W
SIM_W = eck.SIM_W


def knn_agg_cos(q_emb, db_emb, db_y, top_k=3, return_idx=False):
    import torch
    q = torch.from_numpy(q_emb.astype(np.float32))
    db = torch.from_numpy(db_emb.astype(np.float32))
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    db = db / db.norm(dim=1, keepdim=True).clamp(min=1e-8)
    q, db = q.to("cuda"), db.to("cuda")
    n = q.shape[0]
    idx = np.zeros((n, 5), dtype=np.int64)
    sims = np.zeros((n, 5), dtype=np.float32)
    for s in tqdm(range(0, n, 512), desc="    余弦检索", leave=False):
        e = min(s + 512, n)
        top = torch.topk(q[s:e] @ db.T, 5, dim=1)
        idx[s:e] = top.indices.cpu().numpy()
        sims[s:e] = top.values.cpu().numpy()
    lab = db_y[idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    w = np.maximum(sims[:, :top_k], 1e-7)
    agg = (lab * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]
    if return_idx:
        return agg, sims, idx
    return agg, sims


def main():
    gen_sub = "--gen-sub" in sys.argv
    t0 = time.time()
    print("=" * 60)
    print("ECK001a-noleak: ProstT5 去泄漏重评")

    print("[1/5] 数据 + 分割...")
    train_df = pd.read_csv(eck.TRAIN_PATH)
    test_df = pd.read_csv(eck.TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    lens_tr = np.array([len(s) for s in train_df["sequence"].values])
    short_tr = np.where(lens_tr <= MAX_AA)[0]
    val_short = lens_tr[idx_val] <= MAX_AA
    train_short = np.intersect1d(idx_train, short_tr)
    short_te = np.where(test_df["sequence"].str.len().values <= MAX_AA)[0]
    print(f"  短序列: 全 train.csv {len(short_tr)} | train-only 库 {len(train_short)} | "
          f"val 查询 {int(val_short.sum())} | test 查询 {len(short_te)}")
    print(f"  val 短序列落入全量短库比例: {np.isin(idx_val[val_short], short_tr).mean():.3f}（泄漏源）")
    print(f"  val 短序列落入 train-only 库比例: {np.isin(idx_val[val_short], train_short).mean():.3f}（应为 0）")

    print("[2/5] 构建 CK005 sim_base + 两套 pt5 邻居...")
    sim_kmer = eck.build_kmer(idx_train, idx_val, y_train, "val")
    val_ids = train_df["protein_id"].values[idx_val]
    train_db_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values[idx_train])}
    idx_v, bits_v, _ = eck.parse_aln(eck.ALN_VAL, val_ids, train_db_map)
    sim_bit = eck.agg_labels(idx_v, bits_v, y_train, TOP_K)
    sim_base = ALIGN_W * sim_bit + (1 - ALIGN_W) * sim_kmer

    emb = np.load(eck.EMB_PATH)
    emb_s_tr, emb_s_te = emb["emb_s_tr"], emb["emb_s_te"]
    pos_q = np.searchsorted(short_tr, idx_val[val_short])
    emb_q = emb_s_tr[pos_q]
    y_q = y_val[val_short]

    # 泄漏版：全量短序列库（复现 ECK001）
    sim_leak, sims_leak, idx_leak = knn_agg_cos(
        emb_q, emb_s_tr, y_full[short_tr], TOP_K, return_idx=True)
    leak_self = (short_tr[idx_leak[:, 0]] == idx_val[val_short]).mean()
    print(f"  [泄漏] top-1 命中自身 {leak_self:.3f} | top1_sim={sims_leak[:, 0].mean():.4f} | "
          f"F1(0.5)={eck.macro_f1_fixed(sim_leak, y_q):.4f}")

    # 无泄漏版：train-only 短序列库
    pos_db = np.searchsorted(short_tr, train_short)
    emb_db = emb_s_tr[pos_db]
    sim_pt5, sims_ok, idx_ok = knn_agg_cos(
        emb_q, emb_db, y_full[train_short], TOP_K, return_idx=True)
    hit_self = (train_short[idx_ok[:, 0]] == idx_val[val_short]).mean()
    f1_pt5 = eck.macro_f1_fixed(sim_pt5, y_q)
    r_pt5, p_pt5 = eck.macro_rp(sim_pt5, y_q, 0.5)
    auc_pt5 = eck.macro_auc(y_q, sim_pt5)
    f1_kmer_s = eck.macro_f1_fixed(sim_kmer[val_short], y_q)
    corr = float(np.corrcoef(sim_pt5.ravel(), sim_kmer[val_short].ravel())[0, 1])
    print(f"  [无泄漏] top-1 命中自身 {hit_self:.3f} | top1_sim={sims_ok[:, 0].mean():.4f}")
    print(f"  sim_pt5 短序列: F1(0.5)={f1_pt5:.4f} R={r_pt5:.4f} P={p_pt5:.4f} AUC={auc_pt5:.4f}")
    print(f"  kmer 短序列: F1(0.5)={f1_kmer_s:.4f} | corr(pt5,kmer)={corr:.4f}")

    print("[3/5] 融合网格（短行混 pt5，长行保持 sim_base）...")
    fusion = {}
    for wp in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        sim_f = sim_base.copy()
        sim_f[val_short] = wp * sim_pt5 + (1 - wp) * sim_base[val_short]
        fusion[wp] = round(eck.macro_f1_fixed(sim_f, y_val), 6)
    print(f"  融合网格 (w_pt5): {fusion}")
    best_wp = max(fusion, key=fusion.get)
    gain_sim = fusion[best_wp] - fusion[0.0]
    print(f"  最优 w_pt5={best_wp} F1={fusion[best_wp]:.4f} base={fusion[0.0]:.4f} Δ={gain_sim:+.4f}")

    print("[4/5] 四路集成 + 100 档阈值...")
    vp = np.load(f"{REL_DIR}/TC003_val_probs.npz")
    t1 = np.load(f"{REL_DIR}/TCK001_t001a_probs.npz")
    t2v = np.load(f"{REL_DIR}/T002_val_probs.npz")
    with open(eck.TCK002_CONFIG_PATH, "r", encoding="utf-8") as f:
        w = json.load(f)["best_w"]
    sim_final = sim_base.copy()
    sim_final[val_short] = best_wp * sim_pt5 + (1 - best_wp) * sim_base[val_short]
    ens_val = (w["t001a"] * t1["t001a_val"] + w["t002"] * t2v["t002"]
               + w["cnn"] * vp["cnn"] + SIM_W * sim_final)
    f1_05 = eck.macro_f1_fixed(ens_val, y_val, 0.5)
    cands = np.linspace(0.01, 0.99, 100)
    best_t = np.full(500, 0.5)
    for i in tqdm(range(500), desc="    阈值搜索", unit="label"):
        yt = y_val[:, i]
        if yt.sum() == 0:
            continue
        p = ens_val[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (p & ytm).sum(0)
        fp = (p & ~ytm).sum(0)
        fn = (~p & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_t[i] = cands[f1t.argmax()]
    pred100 = ens_val >= best_t[None, :]
    fs = []
    for i in range(500):
        yt = y_val[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        fs.append(f1_score(yt, pred100[:, i], zero_division=0))
    f1_100 = float(np.mean(fs))
    r100, p100 = eck.macro_rp(ens_val, y_val, best_t)
    auc100 = eck.macro_auc(y_val, ens_val)
    print(f"  四路集成: F1(0.5)={f1_05:.4f} | 100档 F1={f1_100:.4f} R={r100:.4f} P={p100:.4f} AUC={auc100:.4f}")
    print(f"  vs CK005 0.4359: Δ={f1_100 - 0.4359:+.4f} | vs 泄漏 ECK001 0.4558: Δ={f1_100 - 0.4558:+.4f}")

    report = {
        "leak_top1_self_hit": round(float(leak_self), 6),
        "leak_top1_sim": round(float(sims_leak[:, 0].mean()), 6),
        "noleak_top1_self_hit": round(float(hit_self), 6),
        "noleak_top1_sim": round(float(sims_ok[:, 0].mean()), 6),
        "sim_pt5_short_f1_05": round(f1_pt5, 6),
        "sim_pt5_short_recall": round(r_pt5, 6),
        "sim_pt5_short_precision": round(p_pt5, 6),
        "sim_pt5_short_auc": round(auc_pt5, 6),
        "kmer_short_f1_05": round(f1_kmer_s, 6),
        "corr_pt5_kmer": round(corr, 6),
        "fusion_w_pt5": fusion,
        "best_w_pt5": best_wp,
        "gain_sim": round(gain_sim, 6),
        "ensemble_f1_05": round(f1_05, 6),
        "ensemble_f1_100": round(f1_100, 6),
        "ensemble_recall_100": round(r100, 6),
        "ensemble_precision_100": round(p100, 6),
        "ensemble_auc": round(auc100, 6),
        "ck005_baseline": 0.4359,
        "delta_vs_ck005": round(f1_100 - 0.4359, 6),
        "delta_vs_leaked_eck001": round(f1_100 - 0.4558, 6),
    }

    beat = f1_100 > 0.4359 + 0.001
    if not gen_sub or not beat:
        report["submission"] = "none" if not beat else "not generated (no --gen-sub)"
        with open(f"{REL_DIR}/ECK001a_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  报告: {REL_DIR}/ECK001a_report.json")
        if not beat:
            print("  无泄漏后未超 CK005 → 不生成提交")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    print("[5/5] test 端（train 全量短序列库，测试集不在其中）...")
    test_ids = test_df["protein_id"].values
    full_db_map = {pid: i for i, pid in enumerate(train_df["protein_id"].values)}
    idx_t, bits_t, _ = eck.parse_aln(eck.ALN_TEST, test_ids, full_db_map)
    sim_bit_test = eck.agg_labels(idx_t, bits_t, y_full, TOP_K)
    sim_kmer_test = eck.build_kmer(idx_train, idx_val, y_train, "test", y_full=y_full)
    sim_base_test = ALIGN_W * sim_bit_test + (1 - ALIGN_W) * sim_kmer_test
    sim_pt5_test, _ = knn_agg_cos(emb_s_te, emb_s_tr, y_full[short_tr], TOP_K)
    sim_test = sim_base_test.copy()
    sim_test[short_te] = best_wp * sim_pt5_test + (1 - best_wp) * sim_base_test[short_te]

    tp = np.load(f"{REL_DIR}/TC003_test_probs.npz")
    t1t = np.load(f"{REL_DIR}/TCK001_t001a_probs.npz")
    t2t = np.load(f"{REL_DIR}/T002_test_probs.npz")
    ens_test = (w["t001a"] * t1t["t001a_test"] + w["t002"] * t2t["t002"]
                + w["cnn"] * tp["cnn"] + SIM_W * sim_test)
    y_pred = (ens_test >= best_t[None, :]).astype(np.int8)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    submission = pd.concat(
        [submission, pd.DataFrame(y_pred, columns=label_cols, index=submission.index)],
        axis=1)
    sub_path = f"{SUB_DIR}/submission_ECK001a_noleak.csv"
    submission.to_csv(sub_path, index=False)
    report["submission"] = sub_path
    with open(f"{REL_DIR}/ECK001a_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  提交: {sub_path} ({len(submission)} 行)")
    print(f"  报告: {REL_DIR}/ECK001a_report.json")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
