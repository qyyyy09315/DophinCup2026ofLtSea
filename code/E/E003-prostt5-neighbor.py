"""
E003-prostt5-neighbor — ProstT5 结构感知嵌入邻居特征 (短序列子集版)
====================================================================
依据 (文献深读):
  - ProstT5 (Heinzinger et al., bioRxiv 2023, doi:10.1101/2023.07.23.550085):
    ProtT5 微调的双语模型, 氨基酸 ↔ Foldseek 3Di 结构字母互译;
    结构感知嵌入 (encoder mean-pooled 1024-D)
  - FuncSeek (bioRxiv 2026, doi:10.64898/2026.08.07.738656) Table 4 硬证据:
    **ProstT5 原始嵌入直接检索 P@1000 61.6% > ESM2 51.6%** (EC4 层级)——
    结构感知嵌入原生有功能区分度, 无需对比训练
  - 本任务 E001 失败原因: ESM2 mean-pooled 嵌入余弦 0.99+ 全体聚集
    (序列统计无功能信号); ProstT5 编码 3Di 折叠拓扑, 结构相似→功能相关
基准与探针 (2026-08-18):
  - 全量推理不可行: 256-512aa 1.4s/条, 512+aa 6s+/条且 OOM (8GB 显存),
    11 万条预估 70-100h → 只做短序列子集 (≤256aa, 实测 0.12s/条)
  - 探针 (E003-probe, ≤256aa 子集): 余弦 mean=0.80 std=0.12 (ESM2 对照 0.94 聚集);
    top-3 邻居 F1=0.111 vs 随机 0.056/先验 0.041; top-1 标签重合 6.09 vs 期望 0.04
    → 结构嵌入有真实功能区分度, 值得进融合
方案:
  - 嵌入: ProstT5 (models/prostt5) encoder mean-pooled 1024-D, 余弦检索
    (<AA2fold> 前缀 + 残基空格分隔, 截断 2000, bfloat16 GPU, 残基预算批处理)
  - 检索: 短序列 query → 短序列 train 库 top-3 加权聚合 (余弦, FuncSeek 同款)
  - 评估: 子集独立 F1/AUC、corr(pt5, kmer)、全局混合融合
    (短序列行 = w*pt5 + (1-w)*kmer, 长序列行 = kmer) → CK005 框架
文献:
  - ProstT5: Heinzinger et al., bioRxiv 2023, doi:10.1101/2023.07.23.550085
  - FuncSeek: Cloete & Patterton, bioRxiv 2026, doi:10.64898/2026.08.07.738656
输出:
  rel/E003_report.json
  rel/E003_pt5_emb_short.npz (短序列嵌入缓存, 免重复推理)
  submit/submission_E003_ensemble.csv (若融合有效)
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SUB_DIR = os.path.join(PROJECT_ROOT, "submit")

TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
REPORT_PATH = f"{REL_DIR}/E003_report.json"
SUB_PATH = f"{SUB_DIR}/submission_E003_ensemble.csv"
EMB_PATH = f"{REL_DIR}/E003_pt5_emb_short.npz"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "prostt5")

NEIGH_K6_PATH = f"{REL_DIR}/CK002_neigh_k6.npz"
TC003_VAL_PROBS_PATH = f"{REL_DIR}/TC003_val_probs.npz"
TC003_TEST_PROBS_PATH = f"{REL_DIR}/TC003_test_probs.npz"
T001A_PROBS_PATH = f"{REL_DIR}/TCK001_t001a_probs.npz"
T002_VAL_PROBS_PATH = f"{REL_DIR}/T002_val_probs.npz"
T002_TEST_PROBS_PATH = f"{REL_DIR}/T002_test_probs.npz"
TCK002_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")

MAX_LEN = 2000
BATCH = 32
TOP_K = 3
VAL_SIZE = 0.2
RANDOM_STATE = 42
MAX_AA = 256          # 子集化: 只嵌入 ≤256aa (全量 70-100h 不可行)


def macro_auc(y_true, y_prob):
    aucs = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, y_prob[:, i]))
    return float(np.mean(aucs))


def macro_f1_fixed(ens, y_true, t=0.5):
    pred = (ens >= t)
    f1s = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i].astype(bool)
        if yt.sum() == 0:
            continue
        p = pred[:, i]
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def knn_aggregate(query_emb, db_emb, db_y, top_k):
    # FuncSeek 用余弦相似度检索; 内积会被嵌入范数主导 (E001 教训)
    q = torch.from_numpy(query_emb.astype(np.float32))
    db = torch.from_numpy(db_emb.astype(np.float32))
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    db = db / db.norm(dim=1, keepdim=True).clamp(min=1e-8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q, db = q.to(dev), db.to(dev)
    n = q.shape[0]
    idx = np.zeros((n, 5), dtype=np.int64)
    sims = np.zeros((n, 5), dtype=np.float32)
    block = 1000 if dev == "cuda" else 200
    for s in tqdm(range(0, n, block), desc="最近邻检索"):
        e = min(s + block, n)
        sim = q[s:e] @ db.T
        top = torch.topk(sim, 5, dim=1)
        idx[s:e] = top.indices.cpu().numpy()
        sims[s:e] = top.values.cpu().numpy()
    labels = db_y[idx[:, :top_k].ravel()].reshape(n, top_k, -1).astype(np.float32)
    w = np.maximum(sims[:, :top_k], 1e-7)
    return (labels * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None], sims[:, 0]


def embed_sequences(model, tokenizer, seqs, dev):
    """ProstT5 嵌入: <AA2fold> 前缀 + 残基空格分隔, mean-pooled。

    按长度分桶 + 每批 token 预算控制 (8GB 显存, 3B bfloat16):
      - 256aa 以下: ~0.1s/条 (快)
      - 256-512aa:  ~1.4s/条 (注意力 O(L²), 计算主导)
      - 512aa 以上: >6s/条 且大 batch 会 OOM, 建议只测短序列
    """
    MAX_LEN = 2000
    TOKEN_BUDGET = 6000   # 每批 token 上限, 防 OOM (bench 验证)
    BATCH_CAP = 32
    lens = np.array([len(s) for s in seqs])
    order = np.argsort(lens)
    emb = np.zeros((len(seqs), 1024), dtype=np.float32)
    pbar = tqdm(total=len(seqs), desc="ProstT5 嵌入")
    i = 0
    while i < len(seqs):
        j = i
        # 累积到 token 预算或 batch 上限 (桶内按长度排序, padding 浪费最小)
        while (j < len(seqs) and (j - i) < BATCH_CAP
               and lens[order[j]] * (j - i + 1) <= TOKEN_BUDGET):
            j += 1
        batch_idx = order[i:j]
        batch = [seqs[k] for k in batch_idx]
        texts = ["<AA2fold> " + " ".join(list(seq[:MAX_LEN])) for seq in batch]
        enc = tokenizer(texts, padding="longest", truncation=True,
                        max_length=MAX_LEN + 2, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        emb[batch_idx] = pooled.cpu().numpy()
        i = j
        pbar.update(len(batch))
    pbar.close()
    return emb


def main():
    gen_sub = "--gen-sub" in sys.argv
    skip_emb = "--skip-emb" in sys.argv
    max_aa = MAX_AA
    for a in sys.argv:
        if a.startswith("--max-aa="):
            max_aa = int(a.split("=")[1])
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"设备: {dev} | 模式: {'含提交' if gen_sub else '评估'} | 短序列上限: {max_aa}aa")

    # transformers 4.46.3 用绝对路径 from_pretrained 返回 bool 的 bug:
    # chdir 后用相对路径加载
    os.chdir(PROJECT_ROOT)
    from transformers import T5Tokenizer, T5EncoderModel
    print(f"[1/6] 加载 ProstT5 (models/prostt5)...")
    tokenizer = T5Tokenizer.from_pretrained("models/prostt5", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("models/prostt5", torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    print("[2/6] 读取数据 + 短序列筛选...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_train, idx_val = train_test_split(np.arange(len(train_df)),
                                          test_size=VAL_SIZE, random_state=RANDOM_STATE)
    y_train, y_val = y_full[idx_train], y_full[idx_val]
    seqs = train_df["sequence"].values
    seqs_test = test_df["sequence"].values
    lens_tr = np.array([len(s) for s in seqs])
    lens_te = np.array([len(s) for s in seqs_test])
    short_tr = np.where(lens_tr <= max_aa)[0]       # train 全局短序列索引
    short_te = np.where(lens_te <= max_aa)[0]       # test 全局短序列索引
    val_short = lens_tr[idx_val] <= max_aa          # val 行内的短序列 mask
    n_val_s = int(val_short.sum())
    print(f"  数据: train {len(idx_train)} / val {len(idx_val)} / test {len(seqs_test)}")
    print(f"  短序列 (≤{max_aa}aa): train {len(short_tr)} / val {n_val_s} / test {len(short_te)}")

    if skip_emb and os.path.exists(EMB_PATH):
        d = np.load(EMB_PATH)
        emb_s_tr, emb_s_te = d["emb_s_tr"], d["emb_s_te"]
        print(f"  嵌入缓存: train {emb_s_tr.shape} / test {emb_s_te.shape}")
    else:
        print(f"[3/6] ProstT5 推理 (短序列子集 {len(short_tr)}+{len(short_te)} 条, 预计 1-2 小时)...")
        seq_s_tr = [seqs[i].replace("U", "X").replace("Z", "X").replace("O", "X")
                    for i in short_tr]
        seq_s_te = [seqs_test[i].replace("U", "X").replace("Z", "X").replace("O", "X")
                    for i in short_te]
        emb_s_tr = embed_sequences(model, tokenizer, seq_s_tr, dev)
        emb_s_te = embed_sequences(model, tokenizer, seq_s_te, dev)
        np.savez_compressed(EMB_PATH, emb_s_tr=emb_s_tr, emb_s_te=emb_s_te)
        print(f"  嵌入已缓存: {EMB_PATH}")

    print("[4/6] 相似度分布诊断 (短序列 val, 对照 E001 ESM2 的 0.94 聚集)...")
    pos_in_short = np.searchsorted(short_tr, idx_val[val_short])
    q = torch.from_numpy(emb_s_tr[pos_in_short][:2000].astype(np.float32))
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sim_diag = (q @ q.T).numpy()
    np.fill_diagonal(sim_diag, -9)
    off = sim_diag[sim_diag > -1]
    print(f"  短 val 内部余弦: mean={off.mean():.4f} std={off.std():.4f} "
          f"p99={np.percentile(off, 99):.4f} (E001 ESM2 对照: mean≈0.94)")

    print("[5/6] val 短序列检索 (短 train 库) + 融合评估...")
    # 短 val 嵌入 (按 val 行序): emb_s_tr 对应 short_tr 全局序, 需映射
    emb_v_s = emb_s_tr[pos_in_short]
    y_v_s = y_val[val_short]
    sim_pt5_s, top1_sim = knn_aggregate(emb_v_s, emb_s_tr, y_full[short_tr], TOP_K)
    f1_pt5 = macro_f1_fixed(sim_pt5_s, y_v_s)
    auc_pt5 = macro_auc(y_v_s, sim_pt5_s)
    print(f"  sim_pt5 独立 (短序列): F1(0.5)={f1_pt5:.4f} AUC={auc_pt5:.4f} top1_sim={top1_sim.mean():.4f}")

    # 对照: kmer 多k 融合 (全量 val 缓存, 取短序列行)
    K_RANGE = [3, 4, 5, 6, 7, 8]
    FUSION = [0, 0.1, 0.2, 0.4, 0.2, 0.1]
    sim_kmer = np.zeros((len(idx_val), 500), dtype=np.float32)
    for k, wt in zip(K_RANGE, FUSION):
        if wt <= 0:
            continue
        prefix = "CK003" if k >= 7 else "CK002"
        d = np.load(f"{REL_DIR}/{prefix}_neigh_k{k}.npz")
        n = len(idx_val)
        lab = y_train[d["idx_val"][:, :5].ravel()].reshape(n, 5, -1).astype(np.float32)
        s = d["sim_val"][:, :5].astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        sim_kmer += wt * (lab * ww[:, :, None]).sum(axis=1) / ww.sum(axis=1)[:, None]
    sim_kmer_s = sim_kmer[val_short]
    f1_kmer = macro_f1_fixed(sim_kmer_s, y_v_s)
    corr_k = float(np.corrcoef(sim_pt5_s.ravel(), sim_kmer_s.ravel())[0, 1])
    print(f"  kmer 多k (短序列): F1(0.5)={f1_kmer:.4f} | corr(pt5, kmer)={corr_k:.4f}")

    # 融合网格: 短序列行融合, 长序列行保持 kmer (提交逻辑的 val 端验证)
    fusion = {}
    f1_global_base = None
    for wp in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        sim_f = sim_kmer.copy()
        sim_f[val_short] = wp * sim_pt5_s + (1 - wp) * sim_kmer_s
        f1g = macro_f1_fixed(sim_f, y_val)
        fusion[wp] = round(f1g, 6)
        if wp == 0.0:
            f1_global_base = round(f1g, 6)
    print(f"  全局融合网格 (w_pt5, 长序列行保持 kmer): {fusion}")
    print(f"  基线 (纯 kmer 全局) F1={f1_global_base}")
    best_wp = max(fusion, key=fusion.get)
    gain = fusion[best_wp] - f1_global_base

    report = {"sim_pt5_alone_short": {"f1_05": round(f1_pt5, 6), "auc": round(auc_pt5, 6)},
              "kmer_multik_short": {"f1_05": round(f1_kmer, 6)},
              "corr_pt5_kmer": corr_k,
              "fusion_w_pt5_global": fusion, "best_w_pt5": best_wp,
              "global_base_kmer": f1_global_base, "global_gain": round(gain, 6),
              "diag_cosine": {"mean": round(float(off.mean()), 4),
                              "std": round(float(off.std()), 4),
                              "p99": round(float(np.percentile(off, 99)), 4)}}

    if not gen_sub:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  报告: {REPORT_PATH}")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    if gain <= 0.001:
        report["submission"] = "none (融合无增益)"
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("  融合无显著增益 → 不生成提交")
        print("=" * 60)
        print(f"完成! 总耗时: {time.time() - t0:.0f}s")
        return

    print("[6/6] test 端检索 + 提交...")
    emb_te_s = emb_s_te
    sim_pt5_test, _ = knn_aggregate(emb_te_s, emb_s_tr, y_full[short_tr], TOP_K)
    # test 端 kmer 多k (全量缓存, 取短序列行)
    sim_kmer_test = np.zeros((len(seqs_test), 500), dtype=np.float32)
    for k, wt in zip(K_RANGE, FUSION):
        if wt <= 0:
            continue
        d = np.load(f"{REL_DIR}/TCK002a_neigh_full.npz" if k == 6
                    else f"{REL_DIR}/CK004_neigh_k{k}_full.npz")
        n2 = d["idx_test"].shape[0]
        lab = y_full[d["idx_test"][:, :5].ravel()].reshape(n2, 5, -1).astype(np.float32)
        s = d["sim_test"][:, :5].astype(np.float32)
        ww = np.maximum(s, 1e-7) ** 2
        sim_kmer_test += wt * (lab * ww[:, :, None]).sum(axis=1) / ww.sum(axis=1)[:, None]
    sim_test = sim_kmer_test.copy()
    sim_test[short_te] = best_wp * sim_pt5_test + (1 - best_wp) * sim_kmer_test[short_te]
    sim_val_final = sim_kmer.copy()
    sim_val_final[val_short] = best_wp * sim_pt5_s + (1 - best_wp) * sim_kmer_s

    vp = np.load(TC003_VAL_PROBS_PATH)
    tp = np.load(TC003_TEST_PROBS_PATH)
    cnn_val, cnn_test = vp["cnn"], tp["cnn"]
    t1 = np.load(T001A_PROBS_PATH)
    t001a_val, t001a_test = t1["t001a_val"], t1["t001a_test"]
    t2v = np.load(T002_VAL_PROBS_PATH)
    t2t = np.load(T002_TEST_PROBS_PATH)
    t002_val, t002_test = t2v["t002"], t2t["t002"]
    with open(TCK002_CONFIG_PATH, "r", encoding="utf-8") as f:
        w = json.load(f)["best_w"]

    sim_val = sim_val_final
    ens_val = (w["t001a"] * t001a_val + w["t002"] * t002_val
               + w["cnn"] * cnn_val + 0.75 * sim_val)
    ens_test = (w["t001a"] * t001a_test + w["t002"] * t002_test
                + w["cnn"] * cnn_test + 0.75 * sim_test)

    cands = np.linspace(0.01, 0.99, 100)
    best_t = np.full(500, 0.5)
    for i in tqdm(range(500), desc="    阈值搜索", unit="label"):
        yt = y_val[:, i]
        if yt.sum() == 0:
            continue
        p = ens_val[:, i][:, None] >= cands[None, :]
        ytm = yt[:, None].astype(bool)
        tp = (p & ytm).sum(0); fp = (p & ~ytm).sum(0); fn = (~p & ytm).sum(0)
        prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
        f1t = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        best_t[i] = cands[f1t.argmax()]
    y_pred = np.zeros_like(ens_test, dtype=np.int8)
    for i, t in enumerate(best_t):
        y_pred[:, i] = (ens_test[:, i] >= t).astype(np.int8)
    os.makedirs(SUB_DIR, exist_ok=True)
    submission = test_df[["protein_id"]].copy()
    label_df = pd.DataFrame(y_pred, columns=label_cols, index=submission.index)
    submission = pd.concat([submission, label_df], axis=1)
    submission.to_csv(SUB_PATH, index=False)
    report["submission"] = SUB_PATH
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  提交: {SUB_PATH} ({len(submission)} 行)")
    print(f"  报告: {REPORT_PATH}")
    print("=" * 60)
    print(f"完成! 总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
