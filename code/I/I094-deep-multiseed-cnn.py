"""
I094 — deep 分支多种子集成（CNN 支试点）+ 中断续跑
====================================================
方案：C002b 架构/超参逐位沿用（MultiScaleCNNv3 + ASL + AdamW + cosine 80ep + early stop），
  训练 SEEDS=[11, 22, 33] 三种子（对齐 PLM 多种子 I064 先例）；每种子概率平均替换 deep4 的
  cnn 槽（0.5·cnn+0.5·c005 → 0.5·cnn_multiseed+0.5·c005），走 canonical 融合（I065 口径）
  对照单种子缓存臂。判据：融合半折 OOF ≥ +0.001（锚 0.4386，I090 同协议）；样内对照锚 0.4475。

中断续跑（对齐 I083 实战机制，断点粒度 = epoch）：
  - 进度文件 rel/i094_progress.json（tmp+os.replace 原子写），记录每种子完成态；
  - 每 epoch 末原子保存 models/i094_cnn_s{seed}_last.pt（model/opt/scheduler/scaler/
    epoch/best_state/best 元数据/patience），重启自动续下一 epoch；
  - 每种子完成即落 rel/i094_cnn_s{seed}_{val,test}.npz 并标记 done，跨次会话可拼接；
  - 全部种子完成后做融合评估并写 rel/I094_report.json（含断点审计）；
  - --reset 清断点重跑；--smoke 走沙箱路径（_smoke 后缀，不污染正式断点），2 epoch 单种子。

用法：
  python code/I/I094-deep-multiseed-cnn.py            # 正式（可随时 Ctrl-C，重跑即续）
  python code/I/I094-deep-multiseed-cnn.py --smoke    # 续跑闭环自测
  python code/I/I094-deep-multiseed-cnn.py --reset    # 清断点从头
输出：rel/I094_report.json、rel/i094_cnn_multiseed_{val,test}.npz、models/i094_cnn_s*.pt
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CK005A_TMP = os.path.join(REL_DIR, "ck005a_tmp")
TCK002_CONFIG = os.path.join(PROJECT_ROOT, "models", "tck002-ensemble.json")
N_LABELS = 500
SEEDS = [11, 22, 33]
N_EPOCHS = 80
GATE_OOF = 0.001
BASELINE_OOF = 0.4386
BASELINE_IN = 0.4475

c002b = SourceFileLoader(
    "c002b", os.path.join(PROJECT_ROOT, "code", "C", "C002b-cnn.py")
).load_module()
i064 = SourceFileLoader(
    "i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")
).load_module()
MultiScaleCNNv3, AsymmetricLoss = c002b.MultiScaleCNNv3, c002b.AsymmetricLoss
ProteinDataset, collate_onehot = c002b.ProteinDataset, c002b.collate_onehot
get_probs = c002b.get_probs
macro_f1, search_scut, full_pipe = i064.macro_f1, i064.search_scut, i064.full_pipe


def train_epoch(model, loader, optimizer, criterion, scaler):
    """单 epoch 训练：无内层 tqdm 换行刷屏，纯批循环，每个 epoch 严格控制在一行输出。"""
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x, y in loader:
        x, y = x.to(c002b.DEVICE), y.to(c002b.DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with c002b.autocast_ctx():
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    return total_loss / max(n_samples, 1)


def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_save(obj, path):
    tmp = path + ".tmp"
    with Path(tmp).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def paths_for(smoke):
    sfx = "_smoke" if smoke else ""
    return {
        "progress": os.path.join(REL_DIR, f"I094_progress{sfx}.json"),
        "ckpt": lambda seed: os.path.join(MODEL_DIR, f"i094_cnn_s{seed}{sfx}_last.pt"),
        "val": lambda seed: os.path.join(REL_DIR, f"i094_cnn_s{seed}{sfx}_val.npz"),
        "test": lambda seed: os.path.join(REL_DIR, f"i094_cnn_s{seed}{sfx}_test.npz"),
        "multi_val": os.path.join(REL_DIR, f"i094_cnn_multiseed{sfx}_val.npz"),
        "multi_test": os.path.join(REL_DIR, f"i094_cnn_multiseed{sfx}_test.npz"),
        "report": os.path.join(REL_DIR, f"I094_report{sfx}.json"),
    }


def load_progress(P):
    if os.path.exists(P["progress"]):
        return json.load(open(P["progress"], encoding="utf-8"))
    return {"seeds": {}, "fusion_done": False}


def build_sim(y_tr, n_va, val_pids, train_map, val_map):
    def parse_aln(tsv, qids, idmap, ev=0.001, mx=5):
        idx = np.full((len(qids), mx), -1, np.int64)
        bits = np.zeros((len(qids), mx), np.float32)
        qp = {q: i for i, q in enumerate(qids)}
        with open(tsv) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 5:
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


def train_one_seed(seed, sequences, y_full, idx_tr, idx_va, test_seqs, P, args):
    """单种子训练：epoch 粒度断点，完成即落盘 val/test 概率。"""
    ckpt_path = P["ckpt"](seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_ds = ProteinDataset(sequences[idx_tr], y_full[idx_tr], 1500)
    val_ds = ProteinDataset(sequences[idx_va], y_full[idx_va], 1500)
    test_ds = ProteinDataset(test_seqs, np.zeros((len(test_seqs), N_LABELS), dtype=np.int8), 1500)
    tl = DataLoader(train_ds, batch_size=256, shuffle=True,
                    collate_fn=collate_onehot, pin_memory=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=256, shuffle=False,
                    collate_fn=collate_onehot, pin_memory=True, num_workers=0)
    tsel = DataLoader(test_ds, batch_size=256, shuffle=False,
                      collate_fn=collate_onehot, pin_memory=True, num_workers=0)

    model = MultiScaleCNNv3(n_labels=N_LABELS).to(c002b.DEVICE)
    crit = AsymmetricLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if c002b.DEVICE.type == "cuda" else None

    start_epoch, best_f1, best_epoch, patience_counter = 1, 0.0, 0, 0
    best_state = None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=c002b.DEVICE, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sched"])
        if scaler is not None and ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_f1, best_epoch, patience_counter = ck["best_f1"], ck["best_epoch"], ck["patience"]
        best_state = ck["best_state"]
        print(f"  [断点] seed {seed}: 自 epoch {start_epoch} 续（best {best_f1:.4f}@ep{best_epoch}）")

    for epoch in range(start_epoch, args.epochs + 1):
        t_ep = time.time()
        loss = train_epoch(model, tl, opt, crit, scaler)
        val_prob, val_true = get_probs(model, vl)
        _, vf1, _, _ = c002b.evaluate_at_thresholds(val_prob, val_true,
                                                    np.full(N_LABELS, 0.5))
        sch.step()
        if vf1 > best_f1:
            best_f1, best_epoch, patience_counter = vf1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(),
                           "sched": sch.state_dict(),
                           "scaler": scaler.state_dict() if scaler else None,
                           "epoch": epoch, "best_f1": best_f1, "best_epoch": best_epoch,
                           "patience": patience_counter, "best_state": best_state,
                           "seed": seed}, ckpt_path)
        print(f"  seed {seed} ep{epoch:02d}/{args.epochs} | loss={loss:.4f} | "
              f"val_f1(0.5)={vf1:.4f} | best={best_f1:.4f}@ep{best_epoch:02d} "
              f"({time.time() - t_ep:.0f}s)", flush=True)
        if patience_counter >= 10:
            print(f"  seed {seed} early stop at ep{epoch}")
            break

    assert best_state is not None, f"seed {seed}: 无任何有效 epoch"
    model.load_state_dict(best_state)
    val_prob, _ = get_probs(model, vl)
    test_prob, _ = get_probs(model, tsel)
    np.savez_compressed(P["val"](seed), val=val_prob)
    np.savez_compressed(P["test"](seed), test=test_prob)
    os.remove(ckpt_path)  # 种子完成：释放断点文件
    print(f"  seed {seed} 完成（best {best_f1:.4f}@ep{best_epoch}），概率已落盘")


def fusion_eval(P, args):
    """多种子平均进 deep4 → canonical 融合对照（样内 + 半折 OOF）。"""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2, random_state=42)
    y_tr, y_va = y_full[idx_tr], y_full[idx_va]
    n_va = len(idx_va)
    val_pids = train_df["protein_id"].values[idx_va]
    train_map = {p: i for i, p in enumerate(train_df["protein_id"].values[idx_tr])}
    val_map = {p: i for i, p in enumerate(val_pids)}

    cnn0 = np.load(os.path.join(REL_DIR, "TC003_val_probs.npz"))["cnn"]
    cnn_m = np.mean([np.load(P["val"](s))["val"] for s in args.seeds], axis=0)
    c005 = np.load(os.path.join(REL_DIR, "C005_val_probs.npz"))["val"]
    t1 = np.load(os.path.join(REL_DIR, "TCK001_t001a_probs.npz"))["t001a_val"]
    t2 = np.load(os.path.join(REL_DIR, "T002_val_probs.npz"))["t002"]
    with open(TCK002_CONFIG, encoding="utf-8") as f:
        wc = json.load(f)["best_w"]
    plm = np.load(os.path.join(REL_DIR, "plm1664_multiseed_val.npz"))["val"]
    print("  building sim ...")
    sim = build_sim(y_tr, n_va, val_pids, train_map, val_map)

    def arm(cnn_probs, tag):
        deep4 = wc["t001a"] * t1 + wc["t002"] * t2 + wc["cnn"] * (0.5 * cnn_probs + 0.5 * c005)
        f1, auc, mixed, _, _ = full_pipe(deep4, sim, y_tr, y_va, n_va, plm=plm, w=0.35)
        a, b = train_test_split(np.arange(n_va), test_size=0.5, random_state=42)
        preds = np.zeros_like(y_va, dtype=bool)
        for fi, ei in ((a, b), (b, a)):
            ths = search_scut(mixed[fi], y_va[fi])
            preds[ei] = mixed[ei] >= ths
        yb = y_va.astype(bool)
        tp = (preds & yb).sum(0); fp = (preds & ~yb).sum(0); fn = (~preds & yb).sum(0)
        p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        f1_oof = float(np.mean(np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)))
        print(f"  [{tag}] in-sample F1={f1:.4f} AUC={auc:.4f} | half-fold OOF={f1_oof:.4f}")
        return {"f1_in": f1, "auc": auc, "f1_oof": f1_oof}, mixed

    rep_base, _ = arm(cnn0, "baseline 单种子")
    rep_multi, mixed_m = arm(cnn_m, "multiseed 平均")
    np.savez_compressed(P["multi_val"], val=mixed_m)
    np.savez_compressed(P["multi_test"],
                        test=np.mean([np.load(P["test"](s))["test"] for s in args.seeds], axis=0))
    d_oof = rep_multi["f1_oof"] - rep_base["f1_oof"]
    d_in = rep_multi["f1_in"] - rep_base["f1_in"]
    rep = {
        "experiment": "I094",
        "seeds": args.seeds, "n_epochs": args.epochs,
        "baseline": rep_base, "multiseed": rep_multi,
        "delta_oof": d_oof, "delta_in": d_in,
        "oof_anchor": BASELINE_OOF,
        "conclusion": ("有效，进入提交评估" if d_oof >= GATE_OOF
                       else f"试点未过 +{GATE_OOF} 换锁线（ΔOOF {d_oof:+.4f}）"),
    }
    atomic_json_save(rep, P["report"])
    print(f"\n  ΔOOF={d_oof:+.4f} Δin={d_in:+.4f} → {rep['conclusion']}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="沙箱自测（2ep×单种子×10%数据）")
    ap.add_argument("--reset", action="store_true", help="清除断点从头跑")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = [11]
        args.epochs = 2
    else:
        args.seeds = SEEDS
        args.epochs = N_EPOCHS
    P = paths_for(args.smoke)
    t0 = time.time()

    if args.reset and os.path.exists(P["progress"]):
        for s in args.seeds:
            for f in (P["ckpt"](s), P["val"](s), P["test"](s)):
                if os.path.exists(f):
                    os.remove(f)
        os.remove(P["progress"])
        print("--reset: 断点已清")

    progress = load_progress(P)
    print("=" * 80)
    print(f"  I094: deep multi-seed CNN ({'SMOKE' if args.smoke else '正式'}) "
          f"seeds={args.seeds} epochs={args.epochs}")
    print("=" * 80)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.int8)
    idx_all = np.arange(len(train_df))
    idx_tr, idx_va = train_test_split(idx_all, test_size=0.2, random_state=42)
    sequences = train_df["sequence"].values
    test_seqs = test_df["sequence"].values
    if args.smoke:  # 沙箱：10% 训练行加速闭环自测
        rng = np.random.RandomState(0)
        idx_tr = rng.choice(idx_tr, max(len(idx_tr) // 10, 512), replace=False)
        idx_va = idx_va[:2048]
        test_seqs = test_seqs[:512]

    for s_idx, seed in enumerate(args.seeds, 1):
        if progress["seeds"].get(str(seed), {}).get("done"):
            print(f"\n===== [Seed {seed} ({s_idx}/{len(args.seeds)})] 已完成，跳过 =====", flush=True)
            continue
        print(f"\n===== [Seed {seed} ({s_idx}/{len(args.seeds)})] =====", flush=True)
        train_one_seed(seed, sequences, y_full, idx_tr, idx_va, test_seqs, P, args)
        progress["seeds"][str(seed)] = {"done": True, "ts": time.time()}
        atomic_json_save(progress, P["progress"])

    if not args.smoke:
        if progress.get("fusion_done") and os.path.exists(P["report"]):
            print("  融合评估已完成，跳过")
        else:
            fusion_eval(P, args)
            progress["fusion_done"] = True
            atomic_json_save(progress, P["progress"])
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
