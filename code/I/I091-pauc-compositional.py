"""
I091 — 末层 pAUC 组合式微调（LibAUC 完整配方 vs 加权和简化版）
================================================================
背景：I073 的 ZLPR / IA-F1 是「加权和」排序损失（−0.02 ~ −0.04），I080 的难负例 ASL 是
「近似 pAUC」（−0.0047）——两次都是简化近似。文献正式答案是组合式交替（Zhu et al., ICLR
2022 CompositionalAUCLoss；LibAUC 官方 recipe：两阶段 + 重置末层 + MultiLabelpAUCLoss
+ SOPAs + DualSampler）。

设计（对照消融，全部同起点：ASL 阶段一权重 + 重置末层 fc）：
  L0  阶段一基线：ASL 训练 35ep（对齐 I073-L0，独立 SCut F1 ≈ 0.2629 锚）
  A   组合式（完整版）：SOPAs + MultiLabelpAUCLoss(mode='1w', margin=0.6, gamma=0.1,
      beta=0.07) + DualSampler(1:1)，末层重置后 15ep
  B   加权和（简化版对照）：Adam 同时优化 ASL + 1.0·pAUC 损失，无 DualSampler，15ep

指标：独立 SCut F1 / AUC / pAUC[0,0.1] / R@FPR0.069（I072 口径：逐标签宏平均）。
判据：① A 的独立 SCut F1 > 0.2716（PLM 头家族现最优 L1）才接融合；② A 相对 B 的差异 =
组合式是否是关键（完整 vs 简化的直接对照）；③ pAUC↑≠Macro F1↑ 风险，双门槛。
对象限 PLM 头（1664D），绝不对 A/B 两支同施（152 轮 §2 警告）。
文献：Zhu et al., ICLR 2022（CompositionalAUCLoss/PDSCA）；LibAUC docs recipe。
输出：rel/I091_report.json
"""
import os
import io
import json
import time
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from importlib.machinery import SourceFileLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REL_DIR = os.path.join(PROJECT_ROOT, "rel")
SEED = 42
N_LABELS = 500
GATE_F1 = 0.2716  # I073-L1 独立 SCut F1（PLM 头家族现最优）

i064 = SourceFileLoader(
    "i064", os.path.join(PROJECT_ROOT, "code", "I", "I064-diversity-branches.py")
).load_module()
MLP, ASL = i064.MLP, i064.ASL
macro_f1, search_scut = i064.macro_f1, i064.search_scut


def eval_probs(p, y):
    ths = search_scut(p, y)
    f1 = macro_f1(p, y, ths)
    auc = roc_auc_score(y, p, average="macro")
    paucs, recs = [], []
    for j in range(y.shape[1]):
        fpr, tpr, _ = roc_curve(y[:, j].astype(int), p[:, j])
        if len(fpr) < 2:
            continue
        m = fpr <= 0.10
        if m.sum() >= 2:
            paucs.append(float(np.trapezoid(tpr[m], fpr[m]) / 0.10))
        idx = np.searchsorted(fpr, 0.069, side="right") - 1
        recs.append(float(tpr[max(idx, 0)]))
    return {"f1": f1, "auc": float(auc),
            "pauc01": float(np.mean(paucs)) if paucs else 0.0,
            "r_at_069": float(np.mean(recs)) if recs else 0.0}


def predict(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for bx, _ in loader:
            with torch.amp.autocast("cuda"):
                out.append(torch.sigmoid(model(bx.cuda(non_blocking=True)).float()).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_stage1(Xtr, ytr, Xva, yva, epochs=35):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=512, shuffle=True, pin_memory=True)
    vl = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                    batch_size=512, shuffle=False, pin_memory=True)
    model = MLP(Xtr.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = ASL()
    sc = torch.amp.GradScaler("cuda")
    best = (-1, None)
    for ep in range(epochs):
        model.train()
        for bx, by in tl:
            bx, by = bx.cuda(non_blocking=True), by.cuda(non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(bx), by)
            sc.scale(loss).backward()
            sc.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sc.step(opt)
            sc.update()
        sch.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            m = eval_probs(predict(model, vl), yva)
            print(f"    s1 ep{ep + 1:02d} F1={m['f1']:.4f} AUC={m['auc']:.4f} "
                  f"pAUC={m['pauc01']:.4f} R@.069={m['r_at_069']:.4f}")
            if m["f1"] > best[0]:
                best = (m["f1"], copy.deepcopy(model.state_dict()))
    return model, best


def reset_last_fc(model):
    torch.manual_seed(SEED + 1)
    model.head.weight.data.normal_(0.0, 0.02)
    model.head.bias.data.zero_()
    return model


class IdxDS(torch.utils.data.Dataset):
    """带样本索引的数据集（libauc 组合式损失 forward 需要 index）。"""

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i], i


def train_stage2(Xtr, ytr, Xva, yva, base_state, arm):
    from libauc.losses import MultiLabelpAUCLoss
    from libauc.optimizers import SOPAs
    from libauc.sampler import DualSampler

    model = MLP(Xtr.shape[1]).cuda()
    model.load_state_dict(copy.deepcopy(base_state))
    reset_last_fc(model)
    vl = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                    batch_size=512, shuffle=False, pin_memory=True)
    ds = IdxDS(Xtr, ytr)
    epochs = 10  # 配方 15ep 的预算压缩版（标签子采样后每步信息量下降，A/B 同预算对照仍成立）
    # libauc v2 契约：pAUC_DRO_Loss(data_len, gamma=EMA, margin, Lambda=KL 权≈旧 beta 角色)
    lam = {"A": 1.0, "A2": 0.1, "B": 1.0}[arm]
    lr2 = 3e-4 if arm == "A2" else 1e-3  # Λ=0.1 的 LSE 倾斜更尖，降 lr 防发散
    pl = MultiLabelpAUCLoss("1w", num_labels=N_LABELS, data_len=int(Xtr.shape[0]),
                            margin=0.6, gamma=0.9, Lambda=lam)
    crit_asl = ASL()

    if arm in ("A", "A2"):
        sampler = DualSampler(ds, batch_size=1024, labels=ytr, sampling_rate=0.5)
        try:
            loader = DataLoader(ds, batch_size=1024, sampler=sampler, pin_memory=True)
            _probe = next(iter(loader))
            assert len(_probe) == 3
        except Exception:
            loader = DataLoader(ds, batch_sampler=sampler, pin_memory=True)
        opt = SOPAs(model.parameters(), lr=lr2, mode="adam", weight_decay=2e-4)
    else:
        loader = DataLoader(ds, batch_size=1024, shuffle=True, pin_memory=True)
        opt = torch.optim.Adam(model.parameters(), lr=lr2, weight_decay=2e-4)
    n_skip = 0

    best = (-1, None)
    for ep in range(epochs):
        model.train()
        for batch in tqdm(loader, desc=f"s2-{arm} ep{ep + 1:02d}", leave=False):
            bx, by, bidx = batch
            bx = bx.cuda(non_blocking=True)
            by = by.cuda(non_blocking=True)
            bidx = bidx.cuda(non_blocking=True)  # u_pos 在 cuda，index 须同设备
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(bx)
                task = (by.sum(0) > 0).nonzero(as_tuple=False).flatten()  # 批内零正例标签跳过
                if task.numel() > 64:  # 无偏标签子采样（×n/B 缩放），规避逐标签循环 500 次/批
                    sel = task[torch.randperm(task.numel(), device=task.device)[:64]]
                    scale = task.numel() / 64.0
                else:
                    sel, scale = task, 1.0
                l_pa = pl(logits.float(), by.float(), bidx, task_id=sel) * scale
                loss = l_pa if arm in ("A", "A2") else (crit_asl(logits, by) + 1.0 * l_pa)
            if not torch.isfinite(loss):
                opt.zero_grad()
                n_skip += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            opt.step()
        if arm in ("A", "A2") and ep == epochs // 2 and hasattr(opt, "update_lr"):
            opt.update_lr(decay_factor=10)
        if (ep + 1) % 3 == 0 or ep == epochs - 1:
            preds = predict(model, vl)
            if not np.isfinite(preds).all():
                print(f"    s2-{arm} ep{ep + 1:02d} DIVERGED (non-finite preds, skip={n_skip})")
                break
            m = eval_probs(preds, yva)
            print(f"    s2-{arm} ep{ep + 1:02d} F1={m['f1']:.4f} AUC={m['auc']:.4f} "
                  f"pAUC={m['pauc01']:.4f} R@.069={m['r_at_069']:.4f} (skip={n_skip})")
            if m["f1"] > best[0]:
                best = (m["f1"], copy.deepcopy(model.state_dict()))
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, best


def main():
    t0 = time.time()
    print("=" * 88)
    print("  I091: compositional pAUC fine-tune (LibAUC full recipe) vs weighted-sum")
    print("=" * 88)

    train_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train.csv"))
    label_cols = [c for c in train_df.columns if c.startswith("label_")]
    y_full = train_df[label_cols].values.astype(np.float32)
    idx_tr, idx_va = train_test_split(np.arange(len(train_df)), test_size=0.2,
                                      random_state=42)
    df_e = pd.read_parquet(os.path.join(REL_DIR, "train_esm2_embeddings.parquet"))
    df_p = pd.read_parquet(os.path.join(REL_DIR, "train_embeddings.parquet"))
    ec = [c for c in df_e.columns if c.startswith("emb_")]
    pc = [c for c in df_p.columns if c.startswith("emb_")]
    X = np.concatenate([df_e[ec].values, df_p[pc].values], axis=1).astype(np.float32)
    del df_e, df_p
    Xtr, Xva = X[idx_tr], X[idx_va]
    ytr, yva = y_full[idx_tr], y_full[idx_va]
    yva_i8 = yva.astype(np.int8)
    print(f"  data: X{Xtr.shape} / X{Xva.shape}")

    print("\n  [L0] stage-1 ASL (anchor ~0.2629)")
    s1_path = os.path.join(PROJECT_ROOT, "models", "i091_stage1.pt")
    vl = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
                    batch_size=512, shuffle=False, pin_memory=True)
    if os.path.exists(s1_path):
        state_b = torch.load(s1_path, map_location="cuda", weights_only=True)
        model = MLP(X.shape[1]).cuda()
        model.load_state_dict(state_b)
        print("  [L0] loaded cached stage-1 weights")
    else:
        model, (f1b, state_b) = train_stage1(Xtr, ytr, Xva, yva_i8, epochs=35)
        torch.save(state_b, s1_path)
    rep = {"experiment": "I091",
           "recipe": "libauc MultiLabelpAUCLoss('1w',margin=0.6,gamma=0.1,beta=0.07) + SOPAs + DualSampler(1:1)",
           "gate_f1": GATE_F1, "arms": {}}
    m0 = eval_probs(predict(model, vl), yva_i8)
    rep["arms"]["L0_stage1_ASL"] = m0
    print(f"  [L0] {m0}")

    for arm, key in (("A", "A_compositional"), ("A2", "A2_compositional_Lam01"),
                     ("B", "B_weighted_sum")):
        arm_json = os.path.join(REL_DIR, f"I091_arm_{arm}.json")
        if os.path.exists(arm_json):
            rep["arms"][key] = json.load(open(arm_json, encoding="utf-8"))
            print(f"\n  [arm {arm}] loaded cached: {rep['arms'][key]}")
            continue
        print(f"\n  [arm {arm}] " +
              ("compositional (full)" if arm != "B" else "weighted-sum (simplified)"))
        m2, _ = train_stage2(Xtr, ytr, Xva, yva_i8, state_b, arm)
        mm = eval_probs(predict(m2, vl), yva_i8)
        rep["arms"][key] = mm
        json.dump(mm, Path(arm_json).open("w", encoding="utf-8"), indent=2)
        print(f"  [arm {arm}] {mm}")

    a, b = rep["arms"]["A_compositional"], rep["arms"]["B_weighted_sum"]
    rep["contrast_A_minus_B"] = {k: a[k] - b[k] for k in a}
    rep["gate_pass"] = bool(a["f1"] > GATE_F1)
    rep["conclusion"] = ("有效，接融合" if rep["gate_pass"]
                         else f"未过独立门槛 {GATE_F1}，不接融合")
    with io.open(os.path.join(REL_DIR, "I091_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print("\n  contrast A-B:", rep["contrast_A_minus_B"])
    print("  conclusion:", rep["conclusion"])
    print(f"Completed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
